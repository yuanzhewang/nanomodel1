"""
Prepare DPO preference pairs from UltraFeedback.

Downloads chosen/rejected pairs, formats with chat template, and tokenizes.

Usage:
    python -m dpo.prepare_dpo_data
"""

import os
import json

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = "/lambda/nfs/nanomodel1/data/dpo"

USER_START = "<|user|>"
ASSISTANT_START = "<|assistant|>"
TURN_END = "<|end|>"


def format_and_tokenize(prompt: str, response: str, enc: tiktoken.Encoding, max_seq_len: int):
    """Format a single (prompt, response) into tokens with loss mask on response."""
    text = f"{USER_START}\n{prompt.strip()}{TURN_END}\n{ASSISTANT_START}\n{response.strip()}{TURN_END}\n"
    tokens = enc.encode(text, allowed_special="all")

    # Build mask: 1 for assistant response tokens only
    prefix = f"{USER_START}\n{prompt.strip()}{TURN_END}\n{ASSISTANT_START}\n"
    prefix_len = len(enc.encode(prefix, allowed_special="all"))
    mask = [0] * prefix_len + [1] * (len(tokens) - prefix_len)

    if len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]
        mask = mask[:max_seq_len]

    pad_len = max_seq_len - len(tokens)
    tokens.extend([enc.eot_token] * pad_len)
    mask.extend([0] * pad_len)

    return tokens, mask


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    max_seq_len = 1024
    enc = tiktoken.get_encoding("gpt2")

    print("Loading UltraFeedback dataset...")
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

    chosen_tokens_list = []
    chosen_masks_list = []
    rejected_tokens_list = []
    rejected_masks_list = []
    skipped = 0

    print("Processing preference pairs...")
    for row in tqdm(dataset):
        prompt = row["prompt"]

        chosen_msgs = row["chosen"]
        rejected_msgs = row["rejected"]

        # Extract assistant response from message list
        chosen_text = ""
        rejected_text = ""
        for msg in chosen_msgs:
            if msg["role"] == "assistant":
                chosen_text = msg["content"]
        for msg in rejected_msgs:
            if msg["role"] == "assistant":
                rejected_text = msg["content"]

        if not chosen_text or not rejected_text:
            skipped += 1
            continue

        c_tok, c_mask = format_and_tokenize(prompt, chosen_text, enc, max_seq_len)
        r_tok, r_mask = format_and_tokenize(prompt, rejected_text, enc, max_seq_len)

        if sum(c_mask) < 10 or sum(r_mask) < 10:
            skipped += 1
            continue

        chosen_tokens_list.append(c_tok)
        chosen_masks_list.append(c_mask)
        rejected_tokens_list.append(r_tok)
        rejected_masks_list.append(r_mask)

    print(f"  {len(chosen_tokens_list)} pairs, {skipped} skipped")

    # Shuffle and split 95/5
    indices = np.random.default_rng(42).permutation(len(chosen_tokens_list))
    split = int(0.95 * len(indices))
    train_idx = indices[:split]
    val_idx = indices[split:]

    for name, idx_set in [("train", train_idx), ("val", val_idx)]:
        ct = np.array([chosen_tokens_list[i] for i in idx_set], dtype=np.uint16)
        cm = np.array([chosen_masks_list[i] for i in idx_set], dtype=np.uint8)
        rt = np.array([rejected_tokens_list[i] for i in idx_set], dtype=np.uint16)
        rm = np.array([rejected_masks_list[i] for i in idx_set], dtype=np.uint8)

        np.save(os.path.join(DATA_DIR, f"{name}_chosen_tokens.npy"), ct)
        np.save(os.path.join(DATA_DIR, f"{name}_chosen_masks.npy"), cm)
        np.save(os.path.join(DATA_DIR, f"{name}_rejected_tokens.npy"), rt)
        np.save(os.path.join(DATA_DIR, f"{name}_rejected_masks.npy"), rm)

    print(f"\nSaved to {DATA_DIR}:")
    print(f"  Train: {len(train_idx)} pairs")
    print(f"  Val:   {len(val_idx)} pairs")

    meta = {"max_seq_len": max_seq_len, "num_train": len(train_idx), "num_val": len(val_idx)}
    with open(os.path.join(DATA_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
