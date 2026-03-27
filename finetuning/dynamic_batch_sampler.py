import random

import torch.utils.data


class DynamicBatchSampler(torch.utils.data.Sampler):
    """max_tokens 예산 기반 동적 배치 구성.

    길이순 정렬 후 메가배치 단위로 셔플하여 데이터 다양성 확보.
    메가배치 내에서 max_tokens 기준으로 가변 크기 배치를 생성.
    (짧은 시퀀스 → 큰 배치, 긴 시퀀스 → 작은 배치 → GPU 메모리 사용량 거의 일정)

    Args:
        lengths: 각 샘플의 시퀀스 길이 리스트.
        max_tokens: 배치 내 최대 토큰 예산 (batch_size × max_len_in_batch ≤ max_tokens).
        shuffle: 메가배치 순서 셔플 여부.
        seed: 셔플 시드.
        mega_batch_size: 메가배치 내 샘플 수 (기본 256).
    """

    def __init__(self, lengths, max_tokens, shuffle=True, seed=42,
                 mega_batch_size=256):
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.mega_batch_size = mega_batch_size
        self.epoch = 0
        # accelerate BatchSamplerShard 호환을 위한 속성
        self.batch_size = max_tokens
        self._batches = self._build_batches()

    def set_epoch(self, epoch):
        """DDP epoch 간 셔플 시드 변경용."""
        self.epoch = epoch
        self._batches = self._build_batches()

    def _build_batches(self):
        # 1) 길이순 정렬
        sorted_indices = sorted(range(len(self.lengths)),
                                key=lambda i: self.lengths[i])

        # 2) 메가배치 단위로 분할
        ms = self.mega_batch_size
        mega_batches = [sorted_indices[i:i + ms]
                        for i in range(0, len(sorted_indices), ms)]

        # 3) 메가배치 셔플 (에폭별 다른 시드)
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(mega_batches)

        # 4) 메가배치 내에서 max_tokens 기준 동적 배치 구성
        batches = []
        for mega in mega_batches:
            batch = []
            max_len_in_batch = 0
            for idx in mega:
                new_max = max(max_len_in_batch, self.lengths[idx])
                # 이 샘플을 추가하면 예산 초과 → 현재 배치를 확정하고 새 배치 시작
                if len(batch) > 0 and new_max * (len(batch) + 1) > self.max_tokens:
                    batches.append(batch)
                    batch = [idx]
                    max_len_in_batch = self.lengths[idx]
                else:
                    batch.append(idx)
                    max_len_in_batch = new_max
            if batch:
                batches.append(batch)

        # 5) 배치 순서도 셔플 (메가배치 내 순서 깨기)
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch + 1000)
            rng.shuffle(batches)

        return batches

    def __iter__(self):
        for batch in self._batches:
            yield batch

    def __len__(self):
        return len(self._batches)
