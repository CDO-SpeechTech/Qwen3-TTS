# coding=utf-8
# Continual Pre-Training (CPT) script for Qwen3-TTS-12Hz-Base model.
#
# Key differences from sft_12hz.py:
#   [Bug Fix 1] text_projection applied to text embeddings (missing in sft_12hz.py)
#   [Bug Fix 2] Double label shifting removed (HF ForCausalLM handles shift internally)
#   - No .detach() on speaker_encoder (jointly trained with backbone)
#   - Self-reference x-vector: speaker embedding from target audio, not ref_audio
#   - Korean conditioning: think_id(4202) + lang_id(2064), NOT nothink_id(4203)
#   - Checkpoint saved as tts_model_type="base" with speaker_encoder included
#   - Variable instruct_len: per-sample speaker injection instead of hardcoded pos 6
#   - LR scheduler: linear warmup + cosine decay
#   - Supports --save_steps, --max_steps, --non_streaming_ratio

import argparse
import json
import math
import os
import random
import shutil

import torch
from accelerate import Accelerator
from cpt_dataset import CPTDataset
from functools import partial
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoConfig


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """Linear warmup then cosine decay to min_lr_ratio * base_lr."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(accelerator, model, config, init_model_path, output_dir):
    """Save Base model checkpoint (keeps speaker_encoder, tts_model_type stays 'base')."""
    os.makedirs(output_dir, exist_ok=True)
    shutil.copytree(init_model_path, output_dir, dirs_exist_ok=True)

    # Keep tts_model_type="base"
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config_dict["tts_model_type"] = "base"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # Save all weights including speaker_encoder
    unwrapped = accelerator.unwrap_model(model)
    state_dict = {k: v.detach().cpu() for k, v in unwrapped.state_dict().items()}
    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output_cpt")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N optimizer steps (0 = epoch-only)")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Stop training after N optimizer steps (0 = no limit)")
    parser.add_argument("--non_streaming_ratio", type=float, default=0.0,
                        help="Fraction of samples to train with Pattern B (interleaved). "
                             "0.0 = all Pattern A (sequential). 0.5 = 50/50 mix.")
    parser.add_argument("--log_steps", type=int, default=10)
    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="tensorboard",
    )

    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    train_data = [json.loads(l) for l in open(args.train_jsonl) if l.strip()]

    # 길이 기반 배칭: audio_codes 길이로 정렬하여 배치 내 패딩 낭비 최소화.
    # 가장 긴 메가배치를 맨 앞에 배치하여 OOM 발생 시 즉시 감지 가능.
    lengths = [len(item["audio_codes"]) for item in train_data]
    sorted_indices = sorted(range(len(train_data)), key=lambda i: lengths[i])

    mega_size = max(args.batch_size * 50, 1)
    mega_batches = [sorted_indices[i:i+mega_size]
                    for i in range(0, len(sorted_indices), mega_size)]

    longest_mega = mega_batches.pop()
    random.Random(42).shuffle(mega_batches)
    mega_batches.insert(0, longest_mega)

    reordered = [idx for mb in mega_batches for idx in mb]
    train_data = [train_data[i] for i in reordered]

    accelerator.print(
        f"길이 기반 배칭 적용: 최단 audio_codes={min(lengths)}, "
        f"최장={max(lengths)}, 메가배치 수={len(mega_batches)}"
    )

    dataset = CPTDataset(train_data, qwen3tts.processor, config)
    collate = partial(dataset.collate_fn, non_streaming_ratio=args.non_streaming_ratio)
    train_dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
        num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True,
    )

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    total_steps = args.num_epochs * num_update_steps_per_epoch
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader, scheduler
    )

    num_code_groups = config.talker_config.num_code_groups
    global_step = 0
    model.train()

    # DDP 래핑 후 model.talker 등 서브모듈에 직접 접근 불가.
    # unwrap하여 원본 모델 참조를 유지한다.
    unwrapped = accelerator.unwrap_model(model)

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch["input_ids"]            # (B, T, 2)
                codec_ids = batch["codec_ids"]            # (B, T, G)
                ref_mels = batch["ref_mels"]              # (B, T_mel, 128)
                text_embedding_mask = batch["text_embedding_mask"]    # (B, T, 1)
                codec_embedding_mask = batch["codec_embedding_mask"]  # (B, T, 1)
                attention_mask = batch["attention_mask"]  # (B, T)
                codec_0_labels = batch["codec_0_labels"]  # (B, T)
                codec_mask = batch["codec_mask"]          # (B, T)
                speaker_positions = batch["speaker_positions"]  # (B,)

                B = input_ids.shape[0]

                # ── Speaker embedding (self-reference, jointly trained) ───────────
                speaker_embedding = unwrapped.speaker_encoder(
                    ref_mels.to(unwrapped.device).to(unwrapped.dtype)
                )  # (B, 1024) — NO .detach(), jointly trained

                # ── Text embeddings with text_projection [Bug Fix 1] ─────────────
                # Bug Fix 1: apply text_projection to match codec_embedding dimension
                # text_embedding: (vocab) → text_hidden_size (2048 for 1.7B)
                # text_projection: text_hidden_size → hidden_size (1024)
                input_text_ids = input_ids[:, :, 0]
                input_text_embedding = unwrapped.talker.text_projection(
                    unwrapped.talker.model.text_embedding(input_text_ids)
                ) * text_embedding_mask  # (B, T, 1024)

                # ── Codec embeddings ─────────────────────────────────────────────
                input_codec_ids = input_ids[:, :, 1]
                input_codec_embedding = (
                    unwrapped.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                )  # (B, T, 1024); spk_pos is zeroed by codec_embedding_mask=False

                # Inject speaker embedding at per-sample speaker positions
                for b_idx in range(B):
                    input_codec_embedding[b_idx, speaker_positions[b_idx], :] = (
                        speaker_embedding[b_idx]
                    )

                # ── Fused dual-track embedding ───────────────────────────────────
                input_embeddings = input_text_embedding + input_codec_embedding

                # Sub-codec groups 1-15 are summed into input_embeddings at codec positions.
                # This matches inference: at each autoregressive step the talker receives
                # codec_hiddens.sum(1) = codec_0_embed + groups_1_15_embeds (modeling line ~1694).
                for grp in range(1, num_code_groups):
                    codec_grp_emb = unwrapped.talker.code_predictor.get_input_embeddings()[grp - 1](
                        codec_ids[:, :, grp]
                    )
                    input_embeddings = input_embeddings + codec_grp_emb * codec_mask.unsqueeze(-1)

                # ── Talker forward (teacher-forcing) [Bug Fix 2] ─────────────────
                # Bug Fix 2: pass full sequence without manual shift.
                # HF ForCausalLM handles the shift internally (logits[:-1] vs labels[1:]).
                outputs = unwrapped.talker(
                    inputs_embeds=input_embeddings,
                    attention_mask=attention_mask,
                    labels=codec_0_labels,
                    output_hidden_states=True,
                )

                # ── Subtalker loss (groups 1..G-1) ───────────────────────────────
                # Need hidden state at position t-1 for codec token at position t.
                # codec_mask True at codec[0..T-1] positions.
                # codec_mask[:, 1:] True at one-before each codec position.
                hidden_states = outputs.hidden_states[0][-1]  # (B, T, D)
                talker_hidden = hidden_states[:, :-1][codec_mask[:, 1:]]  # (N_valid, D)
                talker_codec_ids = codec_ids[codec_mask]                   # (N_valid, G)

                _, sub_talker_loss = unwrapped.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden
                )

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Count optimizer steps
            if accelerator.sync_gradients:
                global_step += 1

                if global_step % args.log_steps == 0:
                    accelerator.print(
                        f"Epoch {epoch} | Step {global_step} "
                        f"| Loss: {loss.item():.4f} "
                        f"| LR: {scheduler.get_last_lr()[0]:.2e}"
                    )

                # Step-based checkpoint
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        ckpt_dir = os.path.join(
                            args.output_model_path, f"checkpoint-step-{global_step}"
                        )
                        save_checkpoint(accelerator, model, config, MODEL_PATH, ckpt_dir)
                        accelerator.print(f"Saved checkpoint: {ckpt_dir}")

                # Early stop
                if args.max_steps > 0 and global_step >= args.max_steps:
                    accelerator.print(f"Reached max_steps={args.max_steps}, stopping.")
                    if accelerator.is_main_process:
                        ckpt_dir = os.path.join(
                            args.output_model_path, f"checkpoint-step-{global_step}"
                        )
                        save_checkpoint(accelerator, model, config, MODEL_PATH, ckpt_dir)
                    return

        # Epoch-based checkpoint
        if accelerator.is_main_process:
            ckpt_dir = os.path.join(
                args.output_model_path, f"checkpoint-epoch-{epoch}"
            )
            save_checkpoint(accelerator, model, config, MODEL_PATH, ckpt_dir)
            accelerator.print(f"Saved epoch checkpoint: {ckpt_dir}")


if __name__ == "__main__":
    train()
