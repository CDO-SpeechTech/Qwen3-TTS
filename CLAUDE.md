# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installation
```bash
pip install -e .
```

### Run example scripts
```bash
python examples/test_model_12hz_custom_voice.py
python examples/test_model_12hz_base.py
python examples/test_tokenizer_12hz.py
```

### Launch web demo
```bash
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --port 8000
```

### Fine-tuning
```bash
# Step 1: Tokenize audio in training data
python finetuning/prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl

# Step 2: SFT training
python finetuning/sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 32 --lr 2e-6 --num_epochs 10 \
  --speaker_name my_speaker
```

## Architecture

Qwen3-TTS is a TTS system with three model variants that all share the same 12Hz speech tokenizer codec.

### Core package: `qwen_tts/`

- **`inference/qwen3_tts_model.py`** — High-level user-facing API (`Qwen3TTSModel`). Routes to three generation methods depending on `tts_model_type` in config.
- **`inference/qwen3_tts_tokenizer.py`** — Wraps the 12Hz speech codec; encodes audio → discrete tokens and decodes tokens → waveforms.
- **`core/models/modeling_qwen3_tts.py`** — Core transformer model (`Qwen3TTSForConditionalGeneration`): Qwen2 text encoder + ECAPA-TDNN speaker encoder + talker code predictor (causal transformer).
- **`core/models/configuration_qwen3_tts.py`** — Config dataclasses for all sub-components.
- **`core/models/processing_qwen3_tts.py`** — Text tokenization via Qwen2Tokenizer.
- **`core/tokenizer_12hz/`** — 12Hz speech codec (current, used by all models).
- **`core/tokenizer_25hz/`** — Legacy 25Hz codec (not used by current models).
- **`cli/demo.py`** — Gradio web UI entry point.

### Three model types (set via `tts_model_type` in config.json)

| Type | Method | Control mechanism |
|------|--------|-------------------|
| `custom_voice` | `generate_custom_voice()` | Pre-trained speaker name string |
| `voice_design` | `generate_voice_design()` | Natural language instruction |
| `base` | `generate_voice_clone()` | Reference audio + transcript |

### Voice cloning modes (Base model only)
- `x_vector_only_mode=True` — Uses only speaker embedding (faster, lower quality)
- `x_vector_only_mode=False` — ICL mode using reference audio tokens (better quality, requires `ref_text`)

### Local model paths
Pre-trained models are stored under `Qwen/` in the repo root:
- `Qwen/Qwen3-TTS-Tokenizer-12Hz` — required by all models
- `Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-{Base,CustomVoice}`
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`

### Fine-tuning data format
Input JSONL for `prepare_data.py`:
```json
{"audio": "./data/utt0001.wav", "text": "transcript", "ref_audio": "./data/ref.wav"}
```

### Key dependencies
- `transformers==4.57.3` (pinned)
- `accelerate==1.12.0` (pinned)
- `onnxruntime` — used by the 12Hz tokenizer
- `einops`, `librosa`, `torchaudio`, `soundfile`

---

## Confirmed Bug Fixes (검증 완료)

아래 수정 사항들은 실제 SFT 학습을 수행하여 효과가 확인되었음 ("기존에 비해 알려졌던 이슈들이 전부 해결되었음. 효과가 있음").

### Bug Fix 1: Missing `text_projection` in training (Issue #120, #198)

**파일**: `finetuning/sft_12hz.py`, `finetuning/cpt_12hz.py`

`text_embedding` 출력에 `text_projection`을 적용하지 않으면 training-inference 불일치 발생.
(0.6B 모델: 2048 vs 1024 차원 불일치로 RuntimeError; 1.7B 모델: silent mismatch)

```python
# 수정 전 (버그)
input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask

# 수정 후 (Bug Fix 1)
input_text_embedding = model.talker.text_projection(
    model.talker.model.text_embedding(input_text_ids)
) * text_embedding_mask
```

### Bug Fix 2: Double label shifting (Issue #179)

**파일**: `finetuning/sft_12hz.py`, `finetuning/cpt_12hz.py`

수동 shift (`[:, :-1]`/`[:, 1:]`) + HuggingFace ForCausalLM 내부 shift = 2번 shift → temporal misalignment → 학습할수록 발화가 빨라지는 현상.

```python
# 수정 전 (버그) — 수동 shift
outputs = model.talker(
    inputs_embeds=input_embeddings[:, :-1],
    labels=codec_0_labels[:, 1:],
    ...
)
hidden_states[:, 1:][codec_mask[:, :-1]]  # 잘못된 hidden state 선택

# 수정 후 (Bug Fix 2) — HF 내부 shift에만 의존
outputs = model.talker(
    inputs_embeds=input_embeddings,
    labels=codec_0_labels,
    ...
)
hidden_states[:, :-1][codec_mask[:, 1:]]  # 올바른 hidden state 선택
```

### Bug Fix 3: Sub-talker `forward_finetune` loss shifting

**파일**: `qwen_tts/core/models/modeling_qwen3_tts.py` — `Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.forward_finetune()`

`logits[:, i]`는 이미 `labels[:, i]`와 직접 정렬되어 있음. HuggingFace `self.loss_function`을 사용하면 내부에서 다시 shift가 일어나 1 group 오프셋 + 마지막 group 미학습 발생.

```python
# 수정 전 (버그)
loss = self.loss_function(logits=logits, labels=labels, ...)

# 수정 후 (Bug Fix 3)
loss = F.cross_entropy(
    logits.reshape(-1, self.config.vocab_size),
    labels.reshape(-1),
    ignore_index=-100,
)
```

### 관련 GitHub Issues / PRs
- Issue #120: text_projection missing in SFT training
- Issue #179: double shift bug causing speech speedup over epochs
- Issue #198: training-inference mismatch for 0.6B model
- PR #178: sub-codec groups removal (적용 **금지** — inference의 `codec_hiddens.sum(1)` 패턴과 불일치하여 음성이 빨라지는 부작용 보고됨)

---

## CPT (Continual Pre-Training) — 한국어 특화

### 파일 구성
- `finetuning/prepare_data_cpt.py` — 오디오 tokenization (resume 지원, ref_audio 불필요)
- `finetuning/cpt_dataset.py` — CPTDataset (Pattern A/B, Korean lang_id, instruct prefix)
- `finetuning/cpt_12hz.py` — CPT 학습 스크립트 (Bug Fix 1-3 모두 적용)

### SFT와의 핵심 차이
| 항목 | SFT | CPT |
|------|-----|-----|
| 언어 conditioning | `nothink_id(4203)` | `think_id(4202) + lang_id(2064)` |
| Speaker x-vector | ref_audio (별도 파일) | target audio self-reference |
| Speaker encoder | `.detach()` (gradient 차단) | jointly train (gradient 흘림) |
| 출력 모델 타입 | `custom_voice` | `base` 유지 |
| ref_audio | 필수 | 불필요 |

### CPT 실행
```bash
# Step 1: 오디오 tokenization
python finetuning/prepare_data_cpt.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl ko_raw.jsonl \
  --output_jsonl ko_with_codes.jsonl

# Step 2: CPT 학습
python finetuning/cpt_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output_cpt \
  --train_jsonl ko_with_codes.jsonl \
  --batch_size 2 --lr 1e-5 --num_epochs 3 \
  --non_streaming_ratio 0.0
```

### CPT 입력 데이터 형식
```json
{"audio": "./ko/utt001.wav", "text": "안녕하세요.", "speaker_id": "SPK001", "instruct": "밝게"}
{"audio": "./ko/utt002.wav", "text": "감사합니다.", "speaker_id": "SPK001"}
```
