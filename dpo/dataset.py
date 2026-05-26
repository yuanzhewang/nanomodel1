"""
DPO dataset: loads chosen/rejected preference pairs with loss masks.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class DPODataset(Dataset):
    def __init__(self, data_dir: str, split: str):
        self.chosen_tokens = np.load(os.path.join(data_dir, f"{split}_chosen_tokens.npy"))
        self.chosen_masks = np.load(os.path.join(data_dir, f"{split}_chosen_masks.npy"))
        self.rejected_tokens = np.load(os.path.join(data_dir, f"{split}_rejected_tokens.npy"))
        self.rejected_masks = np.load(os.path.join(data_dir, f"{split}_rejected_masks.npy"))

        print(f"[DPO {split}] {len(self.chosen_tokens)} pairs, seq_len={self.chosen_tokens.shape[1]}")

    def __len__(self):
        return len(self.chosen_tokens)

    def __getitem__(self, idx):
        chosen_tok = torch.from_numpy(self.chosen_tokens[idx].astype(np.int64))
        chosen_mask = torch.from_numpy(self.chosen_masks[idx].astype(np.int64))
        rejected_tok = torch.from_numpy(self.rejected_tokens[idx].astype(np.int64))
        rejected_mask = torch.from_numpy(self.rejected_masks[idx].astype(np.int64))

        return (
            chosen_tok[:-1], chosen_tok[1:], chosen_mask[1:],
            rejected_tok[:-1], rejected_tok[1:], rejected_mask[1:],
        )
