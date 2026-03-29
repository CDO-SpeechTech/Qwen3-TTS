## Qwen3-TTS-12Hz-1.7B/0.6B-Base 파인튜닝

단일화자와 다화자 SFT를 동일한 스크립트(`sft_12hz.py`)로 수행합니다.

사전 준비: `pip install qwen-tts` 실행 후 아래 명령어로 저장소를 클론합니다.

```
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS/finetuning
```

### 1) 입력 JSONL 형식

학습 데이터를 JSONL 파일로 준비합니다 (한 줄에 JSON 객체 하나). 각 줄에 다음 필드가 필요합니다:

- `audio`: 학습용 음성 파일 경로 (wav)
- `text`: `audio`에 대응하는 텍스트 전사
- `ref_audio`: 참조 화자 음성 파일 경로 (wav)
- `speaker_id` (선택): 다화자 학습 시 화자 식별자

단일화자 예시:
```jsonl
{"audio":"./data/utt0001.wav","text":"안녕하세요, 반갑습니다.","ref_audio":"./data/ref.wav"}
{"audio":"./data/utt0002.wav","text":"오늘 날씨가 좋네요.","ref_audio":"./data/ref.wav"}
```

다화자 예시:
```jsonl
{"audio":"./data/alice/utt001.wav","text":"안녕하세요.","ref_audio":"./data/alice/ref.wav","speaker_id":"alice"}
{"audio":"./data/bob/utt001.wav","text":"감사합니다.","ref_audio":"./data/bob/ref.wav","speaker_id":"bob"}
```

`ref_audio` 권장사항:
- 같은 화자의 모든 샘플에 동일한 `ref_audio`를 사용하는 것을 강력 권장합니다.
- 화자별로 `ref_audio`를 통일하면 합성 시 화자 일관성과 안정성이 향상됩니다.


### 2) 데이터 전처리 (`audio_codes` 추출)

`train_raw.jsonl`에서 `audio_codes`가 포함된 학습용 JSONL을 생성합니다:

```bash
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```


### 3) 파인튜닝

**단일화자** (JSONL에 `speaker_id` 없을 때 `--speaker_name` 필수):

```bash
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 32 \
  --lr 2e-6 \
  --num_epochs 10 \
  --speaker_name speaker_test
```

**다화자** (JSONL에 `speaker_id` 필드 필요):

```bash
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output_multi \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 2 \
  --lr 2e-5 \
  --num_epochs 3
```

추가 옵션:
- `--speaker_slot_start`: codec_embedding 슬롯 시작 위치 (기본값: 3000)
- `--max_seq_length`: 이 값보다 긴 audio_codes를 가진 샘플 제외 (0=제한 없음)
- `--max_tokens`: 동적 배칭 토큰 예산 (0=`--batch_size` 고정 크기 사용)

체크포인트 저장 경로:
- `output/checkpoint-epoch-0`
- `output/checkpoint-epoch-1`
- `output/checkpoint-epoch-2`
- ...


### 4) 추론 테스트

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "output/checkpoint-epoch-2",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = tts.generate_custom_voice(
    text="안녕하세요, 반갑습니다.",
    speaker="speaker_test",
)
sf.write("output.wav", wavs[0], sr)
```

### 원클릭 쉘 스크립트 예시

```bash
#!/usr/bin/env bash
set -e

DEVICE="cuda:0"
TOKENIZER_MODEL_PATH="Qwen/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="Qwen/Qwen3-TTS-12Hz-1.7B-Base"

RAW_JSONL="train_raw.jsonl"
TRAIN_JSONL="train_with_codes.jsonl"
OUTPUT_DIR="output"

BATCH_SIZE=2
LR=2e-5
EPOCHS=3
SPEAKER_NAME="speaker_1"

python prepare_data.py \
  --device ${DEVICE} \
  --tokenizer_model_path ${TOKENIZER_MODEL_PATH} \
  --input_jsonl ${RAW_JSONL} \
  --output_jsonl ${TRAIN_JSONL}

python sft_12hz.py \
  --init_model_path ${INIT_MODEL_PATH} \
  --output_model_path ${OUTPUT_DIR} \
  --train_jsonl ${TRAIN_JSONL} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
  --speaker_name ${SPEAKER_NAME}
```
