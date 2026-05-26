"""
Download and tokenize FineWeb-Edu (sample-10BT) into memory-mapped shards.

Saves tokenized data as uint16 numpy arrays (GPT-2 vocab fits in uint16).
Each shard holds ~100M tokens. Train/val split is 99/1.

Usage:
    python -m data.prepare_fineweb
    python -m data.prepare_fineweb --num_tokens 1e9  # smaller subset for testing
"""

import argparse
import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = "/lambda/nfs/nanomodel1/data/fineweb-edu"
SHARD_SIZE = 50_000_000  # 50M tokens per shard


def tokenize_and_save(num_tokens: int = None):
    os.makedirs(os.path.join(DATA_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "val"), exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    token_buf = np.empty(SHARD_SIZE, dtype=np.uint16)
    buf_idx = 0
    shard_idx = 0
    total_tokens = 0
    val_every = 100  # every 100th document goes to val

    doc_count = 0
    pbar = tqdm(desc="Tokenizing", unit=" tokens")

    for doc in dataset:
        tokens = enc.encode_ordinary(doc["text"])
        tokens.append(eot)

        if len(tokens) + buf_idx > SHARD_SIZE:
            # flush current shard
            split = "val" if shard_idx % val_every == 0 else "train"
            path = os.path.join(DATA_DIR, split, f"shard_{shard_idx:05d}.npy")
            np.save(path, token_buf[:buf_idx])
            pbar.set_postfix(shard=shard_idx, split=split)
            shard_idx += 1
            buf_idx = 0

        # write tokens to buffer
        for tok in tokens:
            if buf_idx >= SHARD_SIZE:
                split = "val" if shard_idx % val_every == 0 else "train"
                path = os.path.join(DATA_DIR, split, f"shard_{shard_idx:05d}.npy")
                np.save(path, token_buf)
                pbar.set_postfix(shard=shard_idx, split=split)
                shard_idx += 1
                buf_idx = 0
            token_buf[buf_idx] = tok
            buf_idx += 1

        total_tokens += len(tokens)
        doc_count += 1
        pbar.update(len(tokens))

        if num_tokens and total_tokens >= num_tokens:
            break

    # flush remaining
    if buf_idx > 0:
        split = "val" if shard_idx % val_every == 0 else "train"
        path = os.path.join(DATA_DIR, split, f"shard_{shard_idx:05d}.npy")
        np.save(path, token_buf[:buf_idx])
        shard_idx += 1

    pbar.close()
    print(f"\nDone: {total_tokens:,} tokens in {shard_idx} shards ({doc_count:,} documents)")
    print(f"Saved to: {DATA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tokens", type=float, default=None,
                        help="Max tokens to process (e.g. 1e9 for 1B). Default: all ~10B.")
    args = parser.parse_args()
    tokenize_and_save(int(args.num_tokens) if args.num_tokens else None)
