"""학습 JSONL에서 화자별 N개를 hold-out하여 train/test 분리."""

import argparse
import json
import random
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--train_jsonl", type=str, default="train_split.jsonl")
    parser.add_argument("--test_jsonl", type=str, default="test_split.jsonl")
    parser.add_argument("--test_per_speaker", type=int, default=5,
                        help="화자별 hold-out할 샘플 수")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input_jsonl) as f:
        data = [json.loads(line) for line in f]

    # 화자별 인덱스 수집
    speaker_indices = defaultdict(list)
    for i, item in enumerate(data):
        spk = item.get("speaker_id", "default").lower()
        speaker_indices[spk].append(i)

    rng = random.Random(args.seed)
    test_indices = set()

    for spk, indices in speaker_indices.items():
        n = min(args.test_per_speaker, len(indices))
        selected = rng.sample(indices, n)
        test_indices.update(selected)
        print(f"  {spk}: {len(indices)}개 중 {n}개 test hold-out")

    train_data = [item for i, item in enumerate(data) if i not in test_indices]
    test_data = [item for i, item in enumerate(data) if i in test_indices]

    with open(args.train_jsonl, "w") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(args.test_jsonl, "w") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n총 {len(data)}개 -> train {len(train_data)}개 + test {len(test_data)}개")
    print(f"  train: {args.train_jsonl}")
    print(f"  test:  {args.test_jsonl}")


if __name__ == "__main__":
    main()
