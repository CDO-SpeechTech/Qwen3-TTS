# Multi-Speaker Fine-Tuning (다화자 SFT)

단일화자 SFT(`sft_12hz.py`)를 N명의 화자로 확장한 버전입니다.
학습이 완료되면 `generate_custom_voice(speaker="alice")` 형태로 여러 화자를 선택해 추론할 수 있는 `custom_voice` 모델이 생성됩니다.

> **검증 상태**: 코드 초안 단계. 실제 학습 결과 확인 후 `README.md`와 병합 예정.

---

## 1. 입력 데이터 형식

단일화자 SFT와 동일하되 `speaker_id` 필드가 추가됩니다.

```jsonl
{"audio": "./alice/utt001.wav", "text": "안녕하세요.", "ref_audio": "./alice_ref.wav", "speaker_id": "alice"}
{"audio": "./alice/utt002.wav", "text": "반갑습니다.", "ref_audio": "./alice_ref.wav", "speaker_id": "alice"}
{"audio": "./bob/utt001.wav",   "text": "Hello world.", "ref_audio": "./bob_ref.wav",   "speaker_id": "bob"}
{"audio": "./bob/utt002.wav",   "text": "How are you.", "ref_audio": "./bob_ref.wav",   "speaker_id": "bob"}
```

**`ref_audio` 권장 사항**:
- 같은 `speaker_id`의 모든 샘플에서 **동일한 `ref_audio` 파일**을 사용할 것을 강력히 권장합니다.
- 화자당 하나의 대표 음성 파일을 사용하면 화자 일관성이 높아집니다.

**슬롯 제한**:
- 기본 설정(`--speaker_slot_start 3000`, `vocab_size=3072`) 기준 최대 **72명**까지 등록 가능합니다.

---

## 2. 데이터 준비 (`audio_codes` 추출)

기존 `prepare_data.py`를 그대로 사용합니다. `speaker_id` 등 추가 필드는 그대로 pass-through됩니다.

```bash
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```

---

## 3. 다화자 SFT 학습

### 싱글 GPU

```bash
python sft_multi_speaker_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output_multi \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 2 \
  --lr 2e-5 \
  --num_epochs 3
```

### 멀티 GPU (싱글 노드)

```bash
torchrun --nproc_per_node=4 sft_multi_speaker_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output_multi \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 2 \
  --lr 2e-5 \
  --num_epochs 3
```

`--nproc_per_node`는 사용할 GPU 수로 설정합니다.
`--batch_size`는 GPU당 배치 사이즈입니다. GPU 4장이면 실질적인 배치 사이즈 = `batch_size × nproc_per_node × gradient_accumulation_steps(4)`.

> `torchrun`으로 실행하면 `Accelerator`가 자동으로 DDP 모드로 동작합니다. 체크포인트 저장은 rank 0 프로세스에서만 수행됩니다.

### 인자 설명

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--init_model_path` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 초기 모델 경로 |
| `--output_model_path` | `output` | 체크포인트 저장 경로 |
| `--train_jsonl` | (필수) | 학습 데이터 JSONL |
| `--batch_size` | `2` | 배치 사이즈 |
| `--lr` | `2e-5` | 학습률 |
| `--num_epochs` | `3` | 에폭 수 |
| `--speaker_slot_start` | `3000` | codec_embedding 슬롯 시작 위치 |

**`--speaker_slot_start`**: 여러 fine-tuned 모델의 화자를 합칠 때 슬롯 충돌을 피하기 위해 사용합니다. 예를 들어 첫 번째 모델은 3000번부터, 두 번째 모델은 3010번부터 시작하도록 설정할 수 있습니다.

체크포인트는 에폭마다 저장됩니다:
```
output_multi/
├── checkpoint-epoch-0/
├── checkpoint-epoch-1/
└── checkpoint-epoch-2/
```

---

## 4. 추론

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "output_multi/checkpoint-epoch-2",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# 화자 alice로 생성
wavs, sr = tts.generate_custom_voice(
    text="안녕하세요.",
    speaker="alice",
)
sf.write("output_alice.wav", wavs[0], sr)

# 화자 bob으로 생성
wavs, sr = tts.generate_custom_voice(
    text="Hello world.",
    speaker="bob",
)
sf.write("output_bob.wav", wavs[0], sr)

# 등록된 화자 목록 확인
print(tts.model.get_supported_speakers())
```

---

## 5. 단일화자 SFT와의 차이점

| 항목 | 단일화자 (`sft_12hz.py`) | 다화자 (`sft_multi_speaker_12hz.py`) |
|------|--------------------------|--------------------------------------|
| `speaker_id` 필드 | 불필요 | 필수 |
| `--speaker_name` 인자 | 필요 | 불필요 (JSONL에서 자동 탐색) |
| Speaker encoder | 첫 배치에서 1회 계산, 재사용 | 학습 전 화자당 1회 사전 계산, 재사용 |
| Speaker embedding 주입 | 배치 전체에 동일하게 broadcast | 배치 내 샘플별 개별 주입 |
| 저장 슬롯 | `codec_embedding[3000]` 고정 | `codec_embedding[3000+i]` (화자 수만큼) |
| `spk_id` dict | `{"speaker_name": 3000}` | `{"alice": 3000, "bob": 3001, ...}` |

두 방식 모두 **Bug Fix 1-3**이 적용되어 있습니다.

---

## 6. 관련 파일

| 파일 | 역할 |
|------|------|
| `sft_multi_speaker_12hz.py` | 다화자 SFT 학습 스크립트 |
| `dataset_multi_speaker.py` | 다화자 데이터셋 (`MultiSpeakerTTSDataset`) |
| `prepare_data.py` | audio_codes 추출 (단일화자와 공유) |
| `sft_12hz.py` | 단일화자 SFT (참고용) |
