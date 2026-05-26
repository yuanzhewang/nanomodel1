"""
SFT dataset that loads pre-tokenized conversations with loss masks.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(self, data_dir: str, split: str):
        tokens = np.load(os.path.join(data_dir, f"{split}_tokens.npy"))
        masks = np.load(os.path.join(data_dir, f"{split}_masks.npy"))

        self.tokens = tokens
        self.masks = masks
        print(f"[SFT {split}] {len(self.tokens)} examples, seq_len={self.tokens.shape[1]}")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = torch.from_numpy(self.tokens[idx].astype(np.int64))
        mask = torch.from_numpy(self.masks[idx].astype(np.int64))
        x = tokens[:-1]
        y = tokens[1:]
        loss_mask = mask[1:]  # shifted to align with targets
        return x, y, loss_mask
