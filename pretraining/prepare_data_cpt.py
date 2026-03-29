# coding=utf-8
# CPT version of prepare_data.py
# Differences:
#   - Pass-through all fields (text, instruct, speaker_id, etc.)
#   - Resume support via a progress file (skips already-processed lines)
#   - ref_audio field is NOT required

import argparse
import json
import os

from qwen_tts import Qwen3TTSTokenizer

BATCH_INFER_NUM = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    args = parser.parse_args()

    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )

    # Resume: count already-written lines
    n_done = 0
    if os.path.exists(args.output_jsonl):
        with open(args.output_jsonl, "r") as f:
            n_done = sum(1 for line in f if line.strip())
        print(f"Resuming: {n_done} lines already processed, skipping them.")

    total_lines = open(args.input_jsonl).readlines()
    total_lines = [json.loads(line.strip()) for line in total_lines if line.strip()]

    # Skip already-done lines
    remaining = total_lines[n_done:]
    print(f"Total: {len(total_lines)}, remaining: {len(remaining)}")

    out_f = open(args.output_jsonl, "a", encoding="utf-8")

    batch_lines = []
    batch_audios = []
    for line in remaining:
        batch_lines.append(line)
        batch_audios.append(line["audio"])

        if len(batch_lines) >= BATCH_INFER_NUM:
            enc_res = tokenizer_12hz.encode(batch_audios)
            for code, item in zip(enc_res.audio_codes, batch_lines):
                item["audio_codes"] = code.cpu().tolist()
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            batch_lines.clear()
            batch_audios.clear()

    if batch_lines:
        enc_res = tokenizer_12hz.encode(batch_audios)
        for code, item in zip(enc_res.audio_codes, batch_lines):
            item["audio_codes"] = code.cpu().tolist()
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        batch_lines.clear()
        batch_audios.clear()

    out_f.close()
    print("Done.")


if __name__ == "__main__":
    main()
