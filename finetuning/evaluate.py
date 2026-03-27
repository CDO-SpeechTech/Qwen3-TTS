"""다화자 SFT 체크포인트 평가 스크립트.

메트릭:
  - CER: faster-whisper large-v3-turbo ASR → 구두점/공백 제거 후 문자 edit distance
  - SECS: speechbrain WavLM-Large SV → cosine similarity (speaker identity)
  - SpeechBERTScore: discrete_speech_metrics WavLM-Large layer 14 (content + style)

추가 설치:
  pip install faster-whisper          # CER (ASR)
  pip install --no-deps git+https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics.git  # SpeechBERTScore
  pip install pysptk pyworld fastdtw jellyfish Levenshtein nltk  # DiscreteSpeechMetrics 종속성 (pypesq 제외)
  # SECS: speechbrain 불필요 — transformers의 WavLM 직접 사용

사용법:
  python finetuning/evaluate.py \
    --checkpoint_path output_multi/checkpoint-epoch-2 \
    --test_jsonl test_with_codes.jsonl \
    --output_dir eval_results/ \
    --device cuda:0
"""

import argparse
import json
import os
import re
import unicodedata
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch


# ---------------------------------------------------------------------------
# CER 계산
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """구두점 제거 → 공백 제거 → 소문자."""
    text = text.lower()
    # Unicode 구두점 + ASCII 구두점 제거
    text = re.sub(r"[^\w]", "", text, flags=re.UNICODE)
    # 숫자와 문자만 남기고 공백도 제거
    text = text.replace(" ", "")
    return text


def edit_distance(ref: str, hyp: str) -> int:
    """문자 단위 Levenshtein distance."""
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def compute_cer(ref: str, hyp: str) -> float:
    ref_norm = normalize_text(ref)
    hyp_norm = normalize_text(hyp)
    if len(ref_norm) == 0:
        return 0.0 if len(hyp_norm) == 0 else 1.0
    return edit_distance(ref_norm, hyp_norm) / len(ref_norm)


# ---------------------------------------------------------------------------
# TTS 합성
# ---------------------------------------------------------------------------

def synthesize_samples(model, test_data, output_dir, language, max_new_tokens, device):
    """테스트 샘플별로 합성 후 wav 저장. 반환: [{...sample_info, synth_path, ref_audio_path}]"""
    wavs_dir = os.path.join(output_dir, "wavs")
    results = []

    for idx, item in enumerate(test_data):
        speaker_id = item.get("speaker_id", "default").lower()
        text = item["text"]
        ref_audio = item["audio"]  # 원음 경로

        spk_dir = os.path.join(wavs_dir, speaker_id)
        os.makedirs(spk_dir, exist_ok=True)
        synth_path = os.path.join(spk_dir, f"{idx}.wav")

        # 이미 합성된 파일이 있으면 스킵 (재실행 시 시간 절약)
        if not os.path.exists(synth_path):
            wavs, sr = model.generate_custom_voice(
                text=text,
                speaker=speaker_id,
                language=language,
                max_new_tokens=max_new_tokens,
            )
            sf.write(synth_path, wavs[0], sr)
        else:
            print(f"  [skip] {synth_path} already exists")

        results.append({
            "idx": idx,
            "speaker_id": speaker_id,
            "text": text,
            "ref_audio": ref_audio,
            "synth_path": synth_path,
        })

        if (idx + 1) % 10 == 0 or idx == len(test_data) - 1:
            print(f"  합성 진행: {idx + 1}/{len(test_data)}")

    return results


# ---------------------------------------------------------------------------
# CER 측정 (faster-whisper)
# ---------------------------------------------------------------------------

def measure_cer(samples, asr_model_size="large-v3-turbo", device="cuda:0"):
    """합성음을 ASR로 transcribe하여 CER 계산."""
    from faster_whisper import WhisperModel

    compute_type = "float16" if "cuda" in device else "float32"
    asr = WhisperModel(asr_model_size, device=device.split(":")[0],
                       device_index=int(device.split(":")[1]) if ":" in device else 0,
                       compute_type=compute_type)

    for sample in samples:
        segments, _ = asr.transcribe(sample["synth_path"], language="ko")
        hyp = "".join(seg.text for seg in segments)
        sample["asr_hyp"] = hyp
        sample["cer"] = compute_cer(sample["text"], hyp)

    return samples


# ---------------------------------------------------------------------------
# SECS 측정 (WavLM-Large + x-vector head via transformers)
# ---------------------------------------------------------------------------

def measure_secs(samples, device="cuda:0"):
    """원음 vs 합성음의 speaker embedding cosine similarity.

    microsoft/wavlm-large-sv (speaker verification finetuned)를 사용.
    speechbrain 없이 transformers AutoModelForAudioXVector로 직접 로드.
    """
    from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
    import torchaudio

    model_name = "microsoft/wavlm-base-plus-sv"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    sv_model = AutoModelForAudioXVector.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def get_embedding(audio_path):
        wav, sr = torchaudio.load(audio_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        inputs = feature_extractor(wav.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        emb = sv_model(**inputs).embeddings
        return emb.squeeze()

    for sample in samples:
        emb_ref = get_embedding(sample["ref_audio"])
        emb_syn = get_embedding(sample["synth_path"])
        cos_sim = torch.nn.functional.cosine_similarity(emb_ref, emb_syn, dim=0)
        sample["secs"] = cos_sim.item()

    del sv_model
    torch.cuda.empty_cache()

    return samples


# ---------------------------------------------------------------------------
# SpeechBERTScore 측정 (discrete-speech-metrics)
# ---------------------------------------------------------------------------

def measure_speech_bert_score(samples, device="cuda:0"):
    """WavLM-Large layer 14 기반 SpeechBERTScore."""
    from discrete_speech_metrics import SpeechBERTScore

    scorer = SpeechBERTScore(
        sr=16000,
        model_type="wavlm-large",
        layer=14,
        use_gpu=("cuda" in device),
    )

    import librosa

    for sample in samples:
        ref_wav, _ = librosa.load(sample["ref_audio"], sr=16000, mono=True)
        syn_wav, _ = librosa.load(sample["synth_path"], sr=16000, mono=True)

        precision, recall, f1 = scorer.score(ref_wav, syn_wav)
        sample["speech_bert_score_precision"] = float(precision)
        sample["speech_bert_score_recall"] = float(recall)
        sample["speech_bert_score_f1"] = float(f1)

    return samples


# ---------------------------------------------------------------------------
# 결과 집계
# ---------------------------------------------------------------------------

def aggregate_results(samples):
    """화자별 + 전체 평균 계산."""
    per_speaker = defaultdict(list)
    for s in samples:
        per_speaker[s["speaker_id"]].append(s)

    metrics = ["cer", "secs", "speech_bert_score_f1"]

    per_speaker_summary = {}
    for spk, spk_samples in sorted(per_speaker.items()):
        summary = {"num_samples": len(spk_samples)}
        for m in metrics:
            vals = [s[m] for s in spk_samples if m in s]
            summary[m] = np.mean(vals) if vals else None
        per_speaker_summary[spk] = summary

    overall = {}
    for m in metrics:
        vals = [s[m] for s in samples if m in s]
        overall[m] = float(np.mean(vals)) if vals else None

    return {"overall": overall, "per_speaker": per_speaker_summary}


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="다화자 SFT 체크포인트 평가")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="SFT 체크포인트 경로")
    parser.add_argument("--test_jsonl", type=str, required=True,
                        help="평가용 JSONL (학습 데이터와 동일 포맷)")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--language", type=str, default="Korean")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--skip_cer", action="store_true", help="CER 측정 스킵")
    parser.add_argument("--skip_secs", action="store_true", help="SECS 측정 스킵")
    parser.add_argument("--skip_sbs", action="store_true", help="SpeechBERTScore 측정 스킵")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 테스트 데이터 로드
    with open(args.test_jsonl) as f:
        test_data = [json.loads(line) for line in f]
    print(f"테스트 샘플 수: {len(test_data)}")

    # 2. 모델 로드
    print(f"체크포인트 로드: {args.checkpoint_path}")
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        args.checkpoint_path,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
    )
    speakers = model.get_supported_speakers()
    print(f"지원 화자: {speakers}")

    # 3. 합성
    print("\n=== 합성 시작 ===")
    samples = synthesize_samples(
        model, test_data, args.output_dir,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )

    # 합성 후 모델 메모리 해제
    del model
    torch.cuda.empty_cache()

    # 4. CER 측정
    if not args.skip_cer:
        print("\n=== CER 측정 (faster-whisper) ===")
        samples = measure_cer(samples, device=args.device)
        cer_vals = [s["cer"] for s in samples]
        print(f"  평균 CER: {np.mean(cer_vals):.4f}")

        # 불일치 샘플 출력
        mismatches = [s for s in samples if s["cer"] > 0.0]
        if mismatches:
            print(f"\n  ASR 불일치 샘플 ({len(mismatches)}개):")
            print("  " + "-" * 70)
            for s in mismatches:
                print(f"  [{s['speaker_id']}] CER={s['cer']:.4f}")
                print(f"    정답: {s['text']}")
                print(f"    인식: {s['asr_hyp']}")
                print(f"    파일: {s['synth_path']}")
                print()

    # 5. SECS 측정
    if not args.skip_secs:
        print("\n=== SECS 측정 (WavLM SV) ===")
        samples = measure_secs(samples, device=args.device)
        secs_vals = [s["secs"] for s in samples]
        print(f"  평균 SECS: {np.mean(secs_vals):.4f}")

    # 6. SpeechBERTScore 측정
    if not args.skip_sbs:
        print("\n=== SpeechBERTScore 측정 ===")
        samples = measure_speech_bert_score(samples, device=args.device)
        sbs_vals = [s["speech_bert_score_f1"] for s in samples]
        print(f"  평균 SpeechBERTScore F1: {np.mean(sbs_vals):.4f}")

    # 7. 결과 집계 및 저장
    results = aggregate_results(samples)

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    details_path = os.path.join(args.output_dir, "details.jsonl")
    with open(details_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 콘솔 출력
    print("\n" + "=" * 60)
    print("평가 결과")
    print("=" * 60)
    overall = results["overall"]
    for k, v in overall.items():
        if v is not None:
            print(f"  {k}: {v:.4f}")

    print("\n화자별:")
    for spk, summary in results["per_speaker"].items():
        parts = [f"n={summary['num_samples']}"]
        for m in ["cer", "secs", "speech_bert_score_f1"]:
            if summary.get(m) is not None:
                parts.append(f"{m}={summary[m]:.4f}")
        print(f"  {spk}: {', '.join(parts)}")

    print(f"\n결과 저장: {results_path}")
    print(f"상세 결과: {details_path}")


if __name__ == "__main__":
    main()
