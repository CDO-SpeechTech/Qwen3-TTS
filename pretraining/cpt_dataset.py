# coding=utf-8
# CPT dataset for Korean continual pre-training of Qwen3-TTS Base model.
#
# Key differences from dataset.py (SFT):
#   - No ref_audio: uses target audio itself as speaker reference (self-reference x-vector)
#   - Handles optional 'instruct' field (prepended to sequence in text channel)
#   - Korean conditioning: think_id(4202) + lang_id(2064) instead of nothink_id(4203)
#   - speaker_pos shifted from 6 (SFT/Auto) to instruct_len+7 (CPT/Korean)
#   - Supports Pattern A (non_streaming=True, sequential) and Pattern B (non_streaming=False, interleaved)
#   - Pattern selection per sample controlled by non_streaming_ratio hyperparameter

import random

import librosa
import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from torch.utils.data import Dataset


class CPTDataset(Dataset):
    def __init__(self, data_list, processor, config: Qwen3TTSConfig):
        self.data_list = data_list
        self.processor = processor
        self.config = config

    def __len__(self):
        return len(self.data_list)

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _tokenize_texts(self, text) -> torch.Tensor:
        inp = self.processor(text=text, return_tensors="pt", padding=True)
        ids = inp["input_ids"]
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return ids

    @torch.inference_mode()
    def _extract_mels(self, audio, sr=24000):
        assert sr == 24000, "Speaker encoder requires 24kHz audio"
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        return mels  # (1, T_mel, 128)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        audio_path = item["audio"]
        text = item["text"]
        audio_codes = torch.tensor(item["audio_codes"], dtype=torch.long)  # (T, G)

        # Text tokenization: wrap and remove last 5 tokens (im_end \n im_start assistant \n)
        text_str = self._build_assistant_text(text)
        text_ids = self._tokenize_texts(text_str)[:, :-5]  # (1, 3+L)

        # Self-reference: extract mel from target audio at 24kHz
        audio_24k, _ = librosa.load(audio_path, sr=24000)
        ref_mel = self._extract_mels(audio_24k, sr=24000)  # (1, T_mel, 128)

        # Instruct handling (optional field)
        instruct_ids = None
        instruct = item.get("instruct", None)
        if instruct:
            instruct_str = f"<|im_start|>user\n{instruct}<|im_end|>\n"
            instruct_ids = self._tokenize_texts(instruct_str)  # (1, instruct_len)

        return {
            "text_ids": text_ids,        # (1, 3+L)
            "audio_codes": audio_codes,  # (T, G)
            "ref_mel": ref_mel,          # (1, T_mel, 128)
            "instruct_ids": instruct_ids,  # (1, I) or None
        }

    def collate_fn(self, batch, non_streaming_ratio: float = 0.0):
        """
        Build batched training tensors for CPT.

        Pattern A (non_streaming=True, sequential):
          - Common prefix: instruct(I) + role(3) + conditioning(6) = I+9 positions
          - Text block: tts_bos + text[0..L-1] + tts_eos + codec_pad phase
          - Codec block: codec_bos + codec[0..T-1] + codec_eos
          - Total length: I + 12 + L + T

        Pattern B (non_streaming=False, interleaved):
          - Common prefix: instruct(I) + role(3) + conditioning(6) = I+9 positions
          - Boundary: pos I+9 = text[0] + codec_bos  (no loss)
          - Interleaved: pos I+10+k = text[k+1] + codec[k]  (k=0..T-1, loss on codec)
          - Total length: I + 11 + T

        Conditioning layout (positions I+3 to I+8, Korean):
          I+3: tts_pad | think_id(4202)
          I+4: tts_pad | think_bos_id(4204)
          I+5: tts_pad | lang_id(2064)
          I+6: tts_pad | think_eos_id(4205)
          I+7: tts_pad | spk_embed_slot  [codec_embedding_mask=False here]
          I+8: tts_bos | codec_pad
        """
        cfg = self.config
        tcfg = cfg.talker_config
        num_code_groups = tcfg.num_code_groups

        # Korean language ID
        lang_id = tcfg.codec_language_id["korean"]

        # Pre-compute per-sample info and choose pattern
        info = []
        for data in batch:
            text_ids = data["text_ids"]        # (1, 3+L)
            audio_codes = data["audio_codes"]  # (T, G)
            instruct_ids = data["instruct_ids"]  # (1, I) or None

            L = text_ids.shape[1] - 3
            T = audio_codes.shape[0]
            I = instruct_ids.shape[1] if instruct_ids is not None else 0

            use_pattern_b = (random.random() < non_streaming_ratio)
            seq_len = (I + 11 + T) if use_pattern_b else (I + 12 + L + T)

            info.append({
                "text_ids": text_ids,
                "audio_codes": audio_codes,
                "instruct_ids": instruct_ids,
                "L": L, "T": T, "I": I,
                "pattern_b": use_pattern_b,
                "seq_len": seq_len,
            })

        B = len(info)
        max_len = max(s["seq_len"] for s in info)

        # Allocate tensors
        input_ids = torch.zeros((B, max_len, 2), dtype=torch.long)
        codec_ids = torch.zeros((B, max_len, num_code_groups), dtype=torch.long)
        text_embedding_mask = torch.zeros((B, max_len), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((B, max_len), dtype=torch.bool)
        codec_mask = torch.zeros((B, max_len), dtype=torch.bool)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        codec_0_labels = torch.full((B, max_len), -100, dtype=torch.long)
        speaker_positions = torch.zeros(B, dtype=torch.long)

        for i, s in enumerate(info):
            text_ids = s["text_ids"]          # (1, 3+L)
            audio_codes = s["audio_codes"]    # (T, G)
            instruct_ids = s["instruct_ids"]
            L, T, I = s["L"], s["T"], s["I"]
            use_b = s["pattern_b"]
            seq_len = s["seq_len"]

            spk_pos = I + 7  # speaker embedding injection position
            speaker_positions[i] = spk_pos

            # ── Instruct prefix (positions 0..I-1) ──────────────────────────────
            if instruct_ids is not None:
                input_ids[i, :I, 0] = instruct_ids[0]
                # codec channel stays 0; codec_embedding_mask stays False

            # ── Role prefix (positions I..I+2) ──────────────────────────────────
            input_ids[i, I:I+3, 0] = text_ids[0, :3]

            # ── Korean conditioning (positions I+3..I+8) ─────────────────────────
            # text channel: tts_pad at I+3..I+7, tts_bos at I+8
            input_ids[i, I+3:I+8, 0] = cfg.tts_pad_token_id
            input_ids[i, I+8, 0] = cfg.tts_bos_token_id

            # codec channel: think, think_bos, lang_id, think_eos at I+3..I+6
            #                0 at I+7 (spk_slot, codec_embedding_mask=False)
            #                codec_pad at I+8
            input_ids[i, I+3, 1] = tcfg.codec_think_id
            input_ids[i, I+4, 1] = tcfg.codec_think_bos_id
            input_ids[i, I+5, 1] = lang_id
            input_ids[i, I+6, 1] = tcfg.codec_think_eos_id
            # input_ids[i, I+7, 1] = 0  (already 0, spk_slot)
            input_ids[i, I+8, 1] = tcfg.codec_pad_id

            if not use_b:
                # ── Pattern A: Sequential ────────────────────────────────────────
                # text channel: text[0..L-1] at I+9..I+8+L, tts_eos at I+9+L,
                #               tts_pad at I+10+L..I+11+L+T
                input_ids[i, I+9:I+9+L, 0] = text_ids[0, 3:]          # text[0..L-1]
                input_ids[i, I+9+L, 0] = cfg.tts_eos_token_id
                input_ids[i, I+10+L:I+12+L+T, 0] = cfg.tts_pad_token_id  # codec phase

                # codec channel: codec_pad at I+9..I+9+L, codec_bos at I+10+L,
                #                codec[0..T-1] at I+11+L..I+10+L+T, codec_eos at I+11+L+T
                input_ids[i, I+9:I+10+L, 1] = tcfg.codec_pad_id
                input_ids[i, I+10+L, 1] = tcfg.codec_bos_id
                input_ids[i, I+11+L:I+11+L+T, 1] = audio_codes[:, 0]
                input_ids[i, I+11+L+T, 1] = tcfg.codec_eos_token_id

                # codec_ids (all groups)
                codec_ids[i, I+11+L:I+11+L+T, :] = audio_codes

                # labels
                codec_0_labels[i, I+11+L:I+11+L+T] = audio_codes[:, 0]
                codec_0_labels[i, I+11+L+T] = tcfg.codec_eos_token_id

                # codec_mask for subtalker (codec[0..T-1] only, NOT eos)
                codec_mask[i, I+11+L:I+11+L+T] = True

            else:
                # ── Pattern B: Interleaved ───────────────────────────────────────
                # text channel:
                #   I+9: text[0]
                #   I+9+k (k=1..min(L-1, T+1)): text[k]
                #   I+9+L: tts_eos (if L <= T+1)
                #   I+10+L..I+10+T: tts_pad
                text_seq = torch.full((T+2,), cfg.tts_pad_token_id, dtype=torch.long)
                n_text = min(L, T + 2)
                if n_text > 0:
                    text_seq[:n_text] = text_ids[0, 3:3+n_text]
                if L < T + 2:
                    text_seq[L] = cfg.tts_eos_token_id
                input_ids[i, I+9:I+11+T, 0] = text_seq

                # codec channel:
                #   I+9: codec_bos (boundary)
                #   I+10+k (k=0..T-1): codec[k]
                #   I+10+T: codec_eos
                input_ids[i, I+9, 1] = tcfg.codec_bos_id
                input_ids[i, I+10:I+10+T, 1] = audio_codes[:, 0]
                input_ids[i, I+10+T, 1] = tcfg.codec_eos_token_id

                # codec_ids (all groups)
                codec_ids[i, I+10:I+10+T, :] = audio_codes

                # labels (codec[0..T-1] and eos)
                codec_0_labels[i, I+10:I+10+T] = audio_codes[:, 0]
                codec_0_labels[i, I+10+T] = tcfg.codec_eos_token_id

                # codec_mask for subtalker (codec[0..T-1] only, NOT eos, NOT boundary)
                codec_mask[i, I+10:I+10+T] = True

            # ── Masks ────────────────────────────────────────────────────────────
            # text_embedding_mask: True for all non-padding positions
            text_embedding_mask[i, :seq_len] = True

            # codec_embedding_mask: True for I+3..seq_len-1 except spk_pos(I+7)
            codec_embedding_mask[i, I+3:seq_len] = True
            codec_embedding_mask[i, spk_pos] = False  # speaker injection position

            # attention_mask
            attention_mask[i, :seq_len] = 1

        ref_mels = torch.cat([d["ref_mel"] for d in batch], dim=0)  # (B, T_mel, 128)

        return {
            "input_ids": input_ids,                                   # (B, T, 2)
            "ref_mels": ref_mels,                                     # (B, T_mel, 128)
            "attention_mask": attention_mask,                         # (B, T)
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1), # (B, T, 1)
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),# (B, T, 1)
            "codec_0_labels": codec_0_labels,                         # (B, T)
            "codec_ids": codec_ids,                                   # (B, T, G)
            "codec_mask": codec_mask,                                 # (B, T)
            "speaker_positions": speaker_positions,                   # (B,)
        }
