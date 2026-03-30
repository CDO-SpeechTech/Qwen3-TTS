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
import random
from collections import defaultdict

from qwen_tts import Qwen3TTSTokenizer

BATCH_INFER_NUM = 32


def write_jsonl(path, data_list):
    with open(path, 'w') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def split_test(data_list, test_per_speaker, seed):
    """화자별 N개를 hold-out하여 train/test 분리."""
    speaker_indices = defaultdict(list)
    for i, item in enumerate(data_list):
        spk = item.get("speaker_id", "default").lower()
        speaker_indices[spk].append(i)

    rng = random.Random(seed)
    test_indices = set()

    for spk, indices in speaker_indices.items():
        n = min(test_per_speaker, len(indices))
        selected = rng.sample(indices, n)
        test_indices.update(selected)
        print(f"  {spk}: {len(indices)}개 중 {n}개 test hold-out")

    train_data = [item for i, item in enumerate(data_list) if i not in test_indices]
    test_data = [item for i, item in enumerate(data_list) if i in test_indices]
    return train_data, test_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--test_jsonl", type=str, default=None,
                        help="지정 시 화자별 hold-out으로 test split 생성")
    parser.add_argument("--test_per_speaker", type=int, default=5,
                        help="화자별 hold-out할 샘플 수 (--test_jsonl 필요)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )

    total_lines = open(args.input_jsonl).readlines()
    total_lines = [json.loads(line.strip()) for line in total_lines]

    final_lines = []
    batch_lines = []
    batch_audios = []
    for line in total_lines:

        batch_lines.append(line)
        batch_audios.append(line['audio'])

        if len(batch_lines) >= BATCH_INFER_NUM:
            enc_res = tokenizer_12hz.encode(batch_audios)
            for code, line in zip(enc_res.audio_codes, batch_lines):
                line['audio_codes'] = code.cpu().tolist()
                final_lines.append(line)
            batch_lines.clear()
            batch_audios.clear()

    if len(batch_audios) > 0:
        enc_res = tokenizer_12hz.encode(batch_audios)
        for code, line in zip(enc_res.audio_codes, batch_lines):
            line['audio_codes'] = code.cpu().tolist()
            final_lines.append(line)
        batch_lines.clear()
        batch_audios.clear()

    # train/test 분리
    if args.test_jsonl:
        train_data, test_data = split_test(final_lines, args.test_per_speaker, args.seed)
        write_jsonl(args.output_jsonl, train_data)
        write_jsonl(args.test_jsonl, test_data)
        print(f"\n총 {len(final_lines)}개 → train {len(train_data)}개 + test {len(test_data)}개")
        print(f"  train: {args.output_jsonl}")
        print(f"  test:  {args.test_jsonl}")
    else:
        write_jsonl(args.output_jsonl, final_lines)

if __name__ == "__main__":
    main()
