import random

import torch.utils.data


class DynamicBatchSampler(torch.utils.data.Sampler):
    """max_tokens 예산 기반 동적 배치 구성.

    길이순 정렬 후 메가배치(256개) 단위로 셔플하여 데이터 다양성 확보.
    메가배치 내에서 max_tokens 기준으로 가변 크기 배치를 생성.
    (짧은 시퀀스 → 큰 배치, 긴 시퀀스 → 작은 배치 → GPU 메모리 사용량 거의 일정)

    Args:
        lengths: 각 샘플의 시퀀스 길이 리스트.
        max_tokens: 배치 내 최대 토큰 예산 (batch_size × max_len_in_batch ≤ max_tokens).
        max_batch_size: 배치 내 최대 샘플 수 상한. 짧은 시퀀스에서 배치 크기가
            지나치게 커지는 것을 방지. 0이면 제한 없음 (기본 32).
        shuffle: 메가배치 순서 셔플 여부.
        seed: 셔플 시드.
        mega_batch_size: 메가배치 내 샘플 수 (기본 256).
    """

    def __init__(self, lengths, max_tokens, max_batch_size=32,
                 shuffle=True, seed=42, mega_batch_size=256):
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size if max_batch_size > 0 else float('inf')
        self.shuffle = shuffle
        self.seed = seed
        self.mega_batch_size = mega_batch_size
        self.epoch = 0
        self._batches = self._build_batches()
        # accelerate BatchSamplerShard 호환: 실제 최대 배치 크기 반영
        self.batch_size = max(len(b) for b in self._batches) if self._batches else 1

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
                tokens_if_added = new_max * (len(batch) + 1)
                batch_full = (
                    len(batch) > 0
                    and (tokens_if_added > self.max_tokens
                         or len(batch) + 1 > self.max_batch_size)
                )
                if batch_full:
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
