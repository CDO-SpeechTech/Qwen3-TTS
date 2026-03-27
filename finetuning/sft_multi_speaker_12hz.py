# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import shutil
from collections import defaultdict

import torch
from accelerate import Accelerator
from dataset_multi_speaker import MultiSpeakerTTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig


def precompute_speaker_embeddings(data_list, config, processor, model, device):
    """화자별 ref_audio를 한 번씩 처리해 speaker embedding을 사전 계산한다.

    권장 사용 케이스처럼 같은 화자의 모든 샘플이 동일한 ref_audio를 사용하면,
    중복 제거 후 화자당 speaker encoder가 딱 한 번만 실행된다.
    (단일화자 SFT에서 첫 배치에 한 번 계산하고 재사용하는 것과 동일한 원리)

    만약 화자당 ref_audio가 여러 개인 경우에는 각각의 임베딩을 추출한 뒤
    평균을 취한다.

    Returns:
        dict[str, torch.Tensor]: speaker_id(소문자) → (1024,) bfloat16 CPU 텐서
    """
    # 화자별 ref_audio 경로 수집 (set으로 중복 제거)
    speaker_refs: dict = defaultdict(set)
    for item in data_list:
        spk = item["speaker_id"].lower()
        speaker_refs[spk].add(item["ref_audio"])

    # extract_mels 메서드 재사용을 위해 빈 데이터셋 인스턴스 생성
    tmp_dataset = MultiSpeakerTTSDataset([], processor, config)

    cache = {}
    model.speaker_encoder.eval()
    with torch.inference_mode():
        for spk_name, ref_paths in speaker_refs.items():
            embs = []
            for path in sorted(ref_paths):  # sorted: 재현성 보장
                wav, sr = tmp_dataset._load_audio_to_np(path)
                mel = tmp_dataset.extract_mels(wav, sr).to(device).to(model.dtype)
                emb = model.speaker_encoder(mel)  # (1, 1024)
                embs.append(emb.cpu())
            # 권장 케이스(ref_audio 1개)에서는 mean이 그 값 자체와 동일
            cache[spk_name] = torch.cat(embs, dim=0).mean(dim=0).to(torch.bfloat16)

    return cache


def save_checkpoint(accelerator, model, args, speaker_emb_cache, epoch):
    """에폭 체크포인트를 저장하고 config.json을 다화자용으로 업데이트한다."""
    output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
    shutil.copytree(args.init_model_path, output_dir, dirs_exist_ok=True)

    # config.json 업데이트
    input_config_file = os.path.join(args.init_model_path, "config.json")
    output_config_file = os.path.join(output_dir, "config.json")
    with open(input_config_file, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    config_dict["tts_model_type"] = "custom_voice"

    # 화자 슬롯 할당 (알파벳 정렬 → 재현성 보장)
    spk_id_dict = {}
    for idx, spk_name in enumerate(sorted(speaker_emb_cache.keys())):
        spk_id_dict[spk_name] = args.speaker_slot_start + idx

    talker_config = config_dict.get("talker_config", {})
    talker_config["spk_id"] = spk_id_dict
    talker_config["spk_is_dialect"] = {k: False for k in spk_id_dict}
    config_dict["talker_config"] = talker_config

    with open(output_config_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # 모델 가중치 저장
    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

    # speaker_encoder 가중치 제거 (custom_voice 추론에 불필요)
    drop_prefix = "speaker_encoder"
    for k in [k for k in state_dict if k.startswith(drop_prefix)]:
        del state_dict[k]

    # 화자별 임베딩을 codec_embedding.weight에 삽입
    weight = state_dict['talker.model.codec_embedding.weight']
    for spk_name, slot in spk_id_dict.items():
        emb = speaker_emb_cache[spk_name].to(weight.device).to(weight.dtype)
        state_dict['talker.model.codec_embedding.weight'][slot] = emb

    save_path = os.path.join(output_dir, "model.safetensors")
    save_file(state_dict, save_path)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_slot_start", type=int, default=3000,
                        help="codec_embedding 슬롯 시작 위치. 여러 fine-tuned 모델을 합칠 때 충돌 방지용.")
    parser.add_argument("--max_seq_length", type=int, default=0,
                        help="이 값을 초과하는 audio_codes 길이의 샘플 제외. 0=제한 없음.")
    parser.add_argument("--max_tokens", type=int, default=0,
                        help="Dynamic batching 토큰 예산. 0이면 --batch_size 고정 크기 사용. "
                             "활성화 시 --batch_size는 배치 내 최대 샘플 수 상한으로 동작.")
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")

    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_3",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]

    # 고유 화자 목록 출력
    unique_speakers = sorted({item["speaker_id"].lower() for item in train_data})
    accelerator.print(f"발견된 화자 수: {len(unique_speakers)}, 화자 목록: {unique_speakers}")

    # 슬롯 오버플로우 검증
    vocab_size = config.talker_config.vocab_size
    assert args.speaker_slot_start + len(unique_speakers) <= vocab_size, (
        f"codec_embedding 슬롯 부족: {len(unique_speakers)}명의 화자가 필요하지만 "
        f"슬롯 {args.speaker_slot_start}~{vocab_size-1} 중 "
        f"{vocab_size - args.speaker_slot_start}개만 가용 가능."
    )

    # 긴 시퀀스 필터링
    if args.max_seq_length > 0:
        before = len(train_data)
        train_data = [item for item in train_data if len(item["audio_codes"]) <= args.max_seq_length]
        if len(train_data) < before:
            accelerator.print(f"{before - len(train_data)}개 샘플 제외 (audio_codes > {args.max_seq_length})")

    lengths = [len(item["audio_codes"]) for item in train_data]
    dataset = MultiSpeakerTTSDataset(train_data, qwen3tts.processor, config)

    if args.max_tokens > 0:
        from dynamic_batch_sampler import DynamicBatchSampler
        batch_sampler = DynamicBatchSampler(
            lengths=lengths, max_tokens=args.max_tokens,
            max_batch_size=args.batch_size, shuffle=True, seed=42,
        )
        batch_sizes = [len(b) for b in batch_sampler._batches]
        accelerator.print(
            f"Dynamic batching 적용: max_tokens={args.max_tokens}, "
            f"batch_size 상한={args.batch_size}, "
            f"audio_codes 최단={min(lengths)}, 최장={max(lengths)}, "
            f"배치 수={len(batch_sampler)}, "
            f"배치 크기 범위={min(batch_sizes)}~{max(batch_sizes)}"
        )
        train_dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=dataset.collate_fn,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )
    else:
        batch_sampler = None
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

    # speaker_encoder는 학습 루프 forward pass에서 호출되지 않으므로 freeze.
    # - DDP는 requires_grad=True 파라미터만 gradient sync를 기다리므로,
    #   freeze하지 않으면 멀티 GPU에서 DDP가 hang됨.
    # - Adam state 메모리도 절약됨.
    qwen3tts.model.speaker_encoder.requires_grad_(False)

    # speaker_encoder 파라미터를 optimizer에서 제외 (Adam state 불필요)
    trainable_params = [p for n, p in qwen3tts.model.named_parameters()
                        if not n.startswith("speaker_encoder")]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    # 학습 시작 전 화자 임베딩 사전 계산.
    # accelerator.prepare() 이후 모델이 각 프로세스의 GPU에 올라간 상태이므로
    # unwrap_model을 통해 base 모델에 접근.
    # (같은 화자의 ref_audio가 동일하다면 화자당 speaker encoder 단 1회 실행)
    accelerator.print("화자 임베딩 사전 계산 중...")
    speaker_emb_cache = precompute_speaker_embeddings(
        train_data, config, qwen3tts.processor,
        accelerator.unwrap_model(model), accelerator.device
    )
    accelerator.print(f"사전 계산 완료: {list(speaker_emb_cache.keys())}")

    # speaker_encoder GPU 메모리 해제 — 학습 루프에서 사용하지 않으므로 CPU로 이동.
    # (requires_grad=False 상태이므로 DDP gradient sync 대상 아님, 체크포인트 저장 시에도 제거됨)
    accelerator.unwrap_model(model).speaker_encoder.cpu()
    torch.cuda.empty_cache()
    accelerator.print("speaker_encoder → CPU 이동, GPU 캐시 해제 완료.")

    num_epochs = args.num_epochs
    model.train()

    # DDP 래핑 후 model.talker 등 서브모듈에 직접 접근 불가 (.module 필요).
    # accelerator.unwrap_model()로 원본 모델 참조를 유지한다.
    # 이 참조를 통한 forward는 DDP의 gradient sync를 우회하지 않음 —
    # 실제 backward는 DDP-wrapped `model`에서 수행되므로 정상 동작.
    unwrapped = accelerator.unwrap_model(model)

    for epoch in range(num_epochs):
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids            = batch['input_ids']
                codec_ids            = batch['codec_ids']
                speaker_ids          = batch['speaker_ids']         # list[str], len=B
                text_embedding_mask  = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask       = batch['attention_mask']
                codec_0_labels       = batch['codec_0_labels']
                codec_mask           = batch['codec_mask']

                input_text_ids  = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                # [Bug Fix 1] text_projection 적용.
                # 미적용 시 0.6B: RuntimeError(2048 vs 1024 차원 불일치),
                # 1.7B: training-inference mismatch (추론은 항상 projection 적용).
                input_text_embedding = unwrapped.talker.text_projection(
                    unwrapped.talker.model.text_embedding(input_text_ids)
                ) * text_embedding_mask
                input_codec_embedding = unwrapped.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask

                # per-sample 화자 임베딩 주입 (위치 6, SFT 방식)
                # speaker_emb_cache는 학습 전 계산된 CPU 텐서이므로 gradient graph 없음
                B = input_ids.shape[0]
                for b_idx in range(B):
                    spk_emb = speaker_emb_cache[speaker_ids[b_idx]].to(
                        input_codec_embedding.device, dtype=input_codec_embedding.dtype
                    )
                    input_codec_embedding[b_idx, 6, :] = spk_emb

                input_embeddings = input_text_embedding + input_codec_embedding

                # Sub-codec groups 1-15를 codec 위치에 합산.
                # 추론 시 codec_hiddens.sum(1) = codec_0_embed + groups_1_15_embeds와 일치.
                for i in range(1, 16):
                    codec_i_embedding = unwrapped.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                # [Bug Fix 2] 수동 shift 없이 전체 시퀀스 전달.
                # HuggingFace ForCausalLM이 내부적으로 shift 처리.
                # 이전 코드: 수동 shift + HF 내부 shift = 2번 shift
                # → temporal misalignment → 학습할수록 발화가 빨라지는 현상 (Issue #179).
                outputs = unwrapped.talker(
                    inputs_embeds=input_embeddings,
                    attention_mask=attention_mask,
                    labels=codec_0_labels,
                    output_hidden_states=True
                )

                # codec 토큰 직전 hidden state를 sub-talker 입력으로 사용
                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[:, :-1][codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                # [Bug Fix 3] forward_finetune 내부에서 F.cross_entropy 직접 사용
                # (모델 코드에 이미 적용됨 — 학습 스크립트에서 별도 처리 불필요)
                sub_talker_logits, sub_talker_loss = unwrapped.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        # 모든 프로세스가 에폭을 완료할 때까지 대기 후 rank 0만 저장.
        # 이 barrier가 없으면 rank 0가 먼저 끝나고 저장을 시작하는 동안
        # 다른 rank가 아직 마지막 배치를 처리 중일 수 있음.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_checkpoint(accelerator, model, args, speaker_emb_cache, epoch)
            accelerator.print(f"Epoch {epoch} 체크포인트 저장 완료.")


if __name__ == "__main__":
    train()
