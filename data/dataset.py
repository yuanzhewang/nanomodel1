"""
Memory-mapped dataset that reads tokenized shards for training.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, data_dir: str, split: str, seq_len: int):
        self.seq_len = seq_len
        shard_dir = os.path.join(data_dir, split)

        shard_files = sorted(
            f for f in os.listdir(shard_dir) if f.endswith(".npy")
        )
        assert len(shard_files) > 0, f"No shards found in {shard_dir}"

        # memory-map all shards and compute total length
        self.shards = []
        self.shard_offsets = []  # cumulative token offsets
        offset = 0
        for fname in shard_files:
            path = os.path.join(shard_dir, fname)
            mmap = np.load(path, mmap_mode="r")
            self.shards.append(mmap)
            self.shard_offsets.append(offset)
            offset += len(mmap)

        self.total_tokens = offset
        self.n_samples = (self.total_tokens - 1) // self.seq_len

        print(f"[{split}] {len(shard_files)} shards, {self.total_tokens:,} tokens, {self.n_samples:,} samples")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for target shift

        # find which shard(s) this spans
        tokens = self._read_range(start, end)
        x = torch.from_numpy(tokens[:-1].astype(np.int64))
        y = torch.from_numpy(tokens[1:].astype(np.int64))
        return x, y

    def _read_range(self, start: int, end: int):
        # binary search for starting shard
        shard_idx = 0
        for i, offset in enumerate(self.shard_offsets):
            if offset <= start:
                shard_idx = i
            else:
                break

        chunks = []
        pos = start
        while pos < end:
            local_start = pos - self.shard_offsets[shard_idx]
            shard_end = (
                self.shard_offsets[shard_idx + 1]
                if shard_idx + 1 < len(self.shards)
                else self.total_tokens
            )
            local_end = min(end - self.shard_offsets[shard_idx], len(self.shards[shard_idx]))
            chunks.append(np.array(self.shards[shard_idx][local_start:local_end]))
            pos = shard_end
            shard_idx += 1

        return np.concatenate(chunks)
