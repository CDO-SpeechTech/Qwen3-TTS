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
import random
from typing import Any, List, Tuple, Union

import librosa
import numpy as np
import torch
import torchaudio
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from torch.utils.data import Dataset

AudioLike = Union[
    str,                     # wav path, URL, base64
    np.ndarray,              # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

MaybeList = Union[Any, List[Any]]


class TTSDataset(Dataset):
    """SFT용 데이터셋 (단일화자/다화자 통합).

    각 샘플에서 speaker_id를 반환한다.
    JSONL에 speaker_id 필드가 없으면 default_speaker_id를 사용한다.

    ref_audio 처리는 precompute_speaker_embeddings 함수에서
    학습 시작 전에 수행되므로 이 클래스에서는 ref_mel을 반환하지 않는다.

    입력 JSONL 필드:
        필수: text, audio_codes, ref_audio
        선택: speaker_id (없으면 default_speaker_id 사용)
    """

    def __init__(self, data_list, processor, config: Qwen3TTSConfig,
                 lag_num=-1, default_speaker_id="default"):
        self.data_list = data_list
        self.processor = processor
        self.lag_num = lag_num
        self.config = config
        self.default_speaker_id = default_speaker_id.lower()

    def __len__(self):
        return len(self.data_list)

    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        audio, sr = librosa.load(x, sr=None, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]

        out: List[Tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        return out

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _ensure_list(self, x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]

    def _tokenize_texts(self, text) -> List[torch.Tensor]:
        input = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = input["input_ids"]
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        return input_id

    @torch.inference_mode()
    def extract_mels(self, audio, sr):
        target_sr = 24000
        audio = torch.from_numpy(audio)
        if sr != target_sr:
            audio = torchaudio.functional.resample(audio, sr, target_sr).unsqueeze(0)
        mels = mel_spectrogram(
            audio,
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000
        ).transpose(1, 2)
        return mels

    def __getitem__(self, idx):
        item = self.data_list[idx]

        text        = item["text"]
        audio_codes = item["audio_codes"]
        speaker_id  = item.get("speaker_id", self.default_speaker_id).lower()

        text = self._build_assistant_text(text)
        text_ids = self._tokenize_texts(text)

        audio_codes = torch.tensor(audio_codes, dtype=torch.long)

        return {
            "text_ids": text_ids[:, :-5],   # 1, t
            "audio_codes": audio_codes,      # t, 16
            "speaker_id": speaker_id,
        }

    def collate_fn(self, batch, non_streaming_ratio: float = 0.0):
        """Build batched training tensors.

        Pattern A (sequential, non-streaming input): text block → codec block.
            seq_len = 11 + L + T (L = text len, T = codec len)
            Layout: role(3) + cond(5) + text(L) + tts_eos(1) + codec_bos(1) + codec(T) + codec_eos(1)

        Pattern B (interleaved, streaming input): text[k+1] aligned with codec[k].
            seq_len = T + 10
            Layout: role(3) + cond(5, identical to A) + boundary(1: text[0]|codec_bos)
                    + interleaved(T: text[1..T]|codec[0..T-1]) + terminal(1: tts_pad|codec_eos)
            Pos 0..7 are bit-identical to Pattern A; only pos 8 onwards differs.

        Per-sample pattern selection: each sample uses Pattern B with probability
        non_streaming_ratio, else Pattern A. Default 0.0 keeps backward-compatible
        Pattern A behavior bit-exact.
        """
        assert self.lag_num == -1

        # 1) Per-sample pattern selection + seq_len computation
        info = []
        for data in batch:
            L = data['text_ids'].shape[1] - 3
            T = data['audio_codes'].shape[0]
            use_b = (random.random() < non_streaming_ratio)
            seq_len = (T + 10) if use_b else (11 + L + T)
            info.append({
                'text_ids': data['text_ids'],
                'audio_codes': data['audio_codes'],
                'speaker_id': data['speaker_id'],
                'L': L, 'T': T,
                'use_b': use_b,
                'seq_len': seq_len,
            })

        b = len(info)
        max_len = max(s['seq_len'] for s in info)

        input_ids               = torch.zeros((b, max_len, 2), dtype=torch.long)
        codec_ids               = torch.zeros((b, max_len, 16), dtype=torch.long)
        text_embedding_mask     = torch.zeros((b, max_len), dtype=torch.bool)
        codec_embedding_mask    = torch.zeros((b, max_len), dtype=torch.bool)
        codec_mask              = torch.zeros((b, max_len), dtype=torch.bool)
        attention_mask          = torch.zeros((b, max_len), dtype=torch.long)
        codec_0_labels          = torch.full((b, max_len), -100, dtype=torch.long)

        # 2) Per-sample fill — dispatch on pattern
        for i, s in enumerate(info):
            tensors = (input_ids, codec_ids, text_embedding_mask, codec_embedding_mask,
                       codec_mask, attention_mask, codec_0_labels)
            if s['use_b']:
                self._fill_pattern_b(i, s, *tensors)
            else:
                self._fill_pattern_a(i, s, *tensors)

        speaker_ids = [s['speaker_id'] for s in info]

        return {
            'input_ids': input_ids,
            'speaker_ids': speaker_ids,
            'attention_mask': attention_mask,
            'text_embedding_mask': text_embedding_mask.unsqueeze(-1),
            'codec_embedding_mask': codec_embedding_mask.unsqueeze(-1),
            'codec_0_labels': codec_0_labels,
            'codec_ids': codec_ids,
            'codec_mask': codec_mask,
        }

    def _fill_pattern_a(self, i, s, input_ids, codec_ids, text_embedding_mask,
                        codec_embedding_mask, codec_mask, attention_mask, codec_0_labels):
        """Pattern A (sequential, non-streaming). Layout unchanged from original collate_fn."""
        text_ids      = s['text_ids']
        audio_codecs  = s['audio_codes']
        audio_codec_0 = audio_codecs[:, 0]

        text_ids_len  = text_ids.shape[1]   # = 3 + L
        codec_ids_len = audio_codec_0.shape[0]  # = T

        # text channel
        input_ids[i,  :3, 0] = text_ids[0, :3]
        input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
        input_ids[i,   7, 0] = self.config.tts_bos_token_id
        input_ids[i, 8:8+text_ids_len-3, 0] = text_ids[0, 3:]
        input_ids[i,   8+text_ids_len-3, 0] = self.config.tts_eos_token_id
        input_ids[i, 8+text_ids_len-2:8+text_ids_len+codec_ids_len, 0] = self.config.tts_pad_token_id
        text_embedding_mask[i, :8+text_ids_len+codec_ids_len] = True

        # codec channel
        input_ids[i, 3:8, 1] = torch.tensor(
            [
                self.config.talker_config.codec_nothink_id,
                self.config.talker_config.codec_think_bos_id,
                self.config.talker_config.codec_think_eos_id,
                0,      # speaker embedding slot
                self.config.talker_config.codec_pad_id
            ]
        )
        input_ids[i, 8:8+text_ids_len-3, 1] = self.config.talker_config.codec_pad_id
        input_ids[i, 8+text_ids_len-3, 1]   = self.config.talker_config.codec_pad_id
        input_ids[i, 8+text_ids_len-2, 1]   = self.config.talker_config.codec_bos_id
        input_ids[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len, 1] = audio_codec_0
        input_ids[i, 8+text_ids_len-1+codec_ids_len, 1] = self.config.talker_config.codec_eos_token_id

        codec_0_labels[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len] = audio_codec_0
        codec_0_labels[i, 8+text_ids_len-1+codec_ids_len] = self.config.talker_config.codec_eos_token_id

        codec_ids[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len, :] = audio_codecs

        codec_embedding_mask[i, 3:8+text_ids_len+codec_ids_len] = True
        codec_embedding_mask[i, 6] = False      # speaker embedding slot

        codec_mask[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len] = True
        attention_mask[i, :8+text_ids_len+codec_ids_len] = True

    def _fill_pattern_b(self, i, s, input_ids, codec_ids, text_embedding_mask,
                        codec_embedding_mask, codec_mask, attention_mask, codec_0_labels):
        """Pattern B (interleaved, streaming input).

        Layout (L = pure text len, T = codec len):
          pos 0..2:   role         | 0
          pos 3..6:   tts_pad      | [nothink, think_bos, think_eos, spk_slot]
          pos 7:      tts_bos      | codec_pad
          pos 8:      text[0]      | codec_bos        (boundary, no codec loss)
          pos 9..8+T: text[1..T]   | codec[0..T-1]    (interleaved, codec loss)
          pos 9+T:    tts_pad/eos  | codec_eos        (terminal, codec loss)

        Total seq_len = T + 10. Pos 0..7 identical to Pattern A.
        Edge case L < T+2: text ends early, tts_eos placed at pos 8+L.
        Edge case L > T+2: text truncated to first T+2 tokens (no tts_eos).
        """
        text_ids      = s['text_ids']
        audio_codecs  = s['audio_codes']
        audio_codec_0 = audio_codecs[:, 0]
        L, T = s['L'], s['T']
        seq_len = T + 10

        # ── Common prefix (pos 0..7, identical to Pattern A) ──────────────
        # text channel
        input_ids[i,  :3, 0] = text_ids[0, :3]
        input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
        input_ids[i,   7, 0] = self.config.tts_bos_token_id

        # codec channel
        input_ids[i, 3:8, 1] = torch.tensor(
            [
                self.config.talker_config.codec_nothink_id,
                self.config.talker_config.codec_think_bos_id,
                self.config.talker_config.codec_think_eos_id,
                0,      # speaker embedding slot
                self.config.talker_config.codec_pad_id
            ]
        )

        # ── Pattern B body (pos 8..9+T) ───────────────────────────────────
        # text channel: text_seq covers pos 8..9+T (length T+2)
        text_seq = torch.full((T + 2,), self.config.tts_pad_token_id, dtype=torch.long)
        n_text = min(L, T + 2)
        if n_text > 0:
            text_seq[:n_text] = text_ids[0, 3:3+n_text]
        if L < T + 2:
            text_seq[L] = self.config.tts_eos_token_id
        input_ids[i, 8:10+T, 0] = text_seq

        # codec channel: codec_bos at pos 8 (boundary), codec[0..T-1] at pos 9..8+T,
        #                codec_eos at pos 9+T
        input_ids[i, 8, 1]       = self.config.talker_config.codec_bos_id
        input_ids[i, 9:9+T, 1]   = audio_codec_0
        input_ids[i, 9+T, 1]     = self.config.talker_config.codec_eos_token_id

        # codec_ids (all 16 groups, sub-talker training)
        codec_ids[i, 9:9+T, :] = audio_codecs

        # labels (HF internal shift: logits[p] → labels[p+1])
        codec_0_labels[i, 9:9+T] = audio_codec_0
        codec_0_labels[i, 9+T]   = self.config.talker_config.codec_eos_token_id

        # masks
        text_embedding_mask[i, :seq_len] = True
        codec_embedding_mask[i, 3:seq_len] = True
        codec_embedding_mask[i, 6] = False        # spk_slot (same as Pattern A)
        codec_mask[i, 9:9+T] = True               # codec[0..T-1] only (excl. boundary/eos)
        attention_mask[i, :seq_len] = 1
