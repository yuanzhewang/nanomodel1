"""
Prepare SFT data from OpenAssistant Conversations (oasst1).

Downloads the dataset, extracts top-rated conversation trees,
formats into chat turns, and tokenizes with proper masking.

Usage:
    python -m sft.prepare_sft_data
"""

import os
import json

import tiktoken
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = "/lambda/nfs/nanomodel1/data/sft"

# Chat template tokens
SYSTEM_START = "<|system|>"
USER_START = "<|user|>"
ASSISTANT_START = "<|assistant|>"
TURN_END = "<|end|>"


def build_chat_text(conversation: list[dict]) -> str:
    """Format a conversation into chat template string."""
    text = ""
    for turn in conversation:
        role = turn["role"]
        content = turn["content"].strip()
        if role == "user":
            text += f"{USER_START}\n{content}{TURN_END}\n"
        elif role == "assistant":
            text += f"{ASSISTANT_START}\n{content}{TURN_END}\n"
    return text


def tokenize_with_mask(text: str, enc: tiktoken.Encoding) -> tuple[list[int], list[int]]:
    """
    Tokenize chat text and create loss mask.
    Only compute loss on assistant responses (not user prompts or special tokens).
    Returns (token_ids, loss_mask) where loss_mask[i]=1 means compute loss on token i.
    """
    tokens = []
    mask = []

    parts = text.split(ASSISTANT_START)
    # First part is everything before first assistant response
    prefix_tokens = enc.encode(parts[0], allowed_special="all")
    tokens.extend(prefix_tokens)
    mask.extend([0] * len(prefix_tokens))

    for part in parts[1:]:
        # Add the assistant start marker
        ast_tokens = enc.encode(ASSISTANT_START, allowed_special="all")
        tokens.extend(ast_tokens)
        mask.extend([0] * len(ast_tokens))

        # Split on TURN_END to separate assistant content from next user turn
        if TURN_END in part:
            assistant_content, remainder = part.split(TURN_END, 1)
            # Assistant content: compute loss
            content_tokens = enc.encode(assistant_content, allowed_special="all")
            tokens.extend(content_tokens)
            mask.extend([1] * len(content_tokens))
            # Turn end + remainder: no loss
            end_tokens = enc.encode(TURN_END + remainder, allowed_special="all")
            tokens.extend(end_tokens)
            mask.extend([0] * len(end_tokens))
        else:
            # No turn end found (shouldn't happen, but handle gracefully)
            content_tokens = enc.encode(part, allowed_special="all")
            tokens.extend(content_tokens)
            mask.extend([1] * len(content_tokens))

    return tokens, mask


def extract_conversations(dataset) -> list[list[dict]]:
    """Extract top-rated conversation threads from oasst1."""
    from collections import defaultdict

    messages = {}
    children_map = defaultdict(list)
    roots = []

    for row in dataset:
        messages[row["message_id"]] = row
        if row["parent_id"] is None:
            roots.append(row)
        else:
            children_map[row["parent_id"]].append(row)

    conversations = []
    for root in roots:
        conv = []
        current = root
        while current is not None:
            conv.append({
                "role": current["role"],
                "content": current["text"],
            })
            children = children_map.get(current["message_id"], [])
            if not children:
                break
            children.sort(key=lambda x: (x.get("rank", 999) or 999))
            current = children[0]

        if len(conv) >= 2:
            conversations.append(conv)

    return conversations


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Loading OpenAssistant dataset...")
    dataset = load_dataset("OpenAssistant/oasst1", split="train")

    print("Extracting conversations...")
    conversations = extract_conversations(dataset)
    print(f"  {len(conversations)} conversations extracted")

    enc = tiktoken.get_encoding("gpt2")
    max_seq_len = 1024

    all_tokens = []
    all_masks = []
    skipped = 0

    print("Tokenizing...")
    for conv in tqdm(conversations):
        text = build_chat_text(conv)
        tokens, mask = tokenize_with_mask(text, enc)

        # Truncate to max_seq_len
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
            mask = mask[:max_seq_len]

        # Skip if too short or no assistant tokens
        if len(tokens) < 32 or sum(mask) == 0:
            skipped += 1
            continue

        # Pad to max_seq_len
        pad_len = max_seq_len - len(tokens)
        tokens.extend([enc.eot_token] * pad_len)
        mask.extend([0] * pad_len)

        all_tokens.append(tokens)
        all_masks.append(mask)

    print(f"  {len(all_tokens)} examples, {skipped} skipped")

    # Shuffle and split 95/5
    indices = np.random.default_rng(42).permutation(len(all_tokens))
    split = int(0.95 * len(indices))

    train_idx = indices[:split]
    val_idx = indices[split:]

    train_tokens = np.array([all_tokens[i] for i in train_idx], dtype=np.uint16)
    train_masks = np.array([all_masks[i] for i in train_idx], dtype=np.uint8)
    val_tokens = np.array([all_tokens[i] for i in val_idx], dtype=np.uint16)
    val_masks = np.array([all_masks[i] for i in val_idx], dtype=np.uint8)

    np.save(os.path.join(DATA_DIR, "train_tokens.npy"), train_tokens)
    np.save(os.path.join(DATA_DIR, "train_masks.npy"), train_masks)
    np.save(os.path.join(DATA_DIR, "val_tokens.npy"), val_tokens)
    np.save(os.path.join(DATA_DIR, "val_masks.npy"), val_masks)

    print(f"\nSaved to {DATA_DIR}:")
    print(f"  Train: {len(train_idx)} examples")
    print(f"  Val:   {len(val_idx)} examples")

    # Save chat template info for reference
    meta = {
        "template_tokens": {
            "system_start": SYSTEM_START,
            "user_start": USER_START,
            "assistant_start": ASSISTANT_START,
            "turn_end": TURN_END,
        },
        "max_seq_len": max_seq_len,
        "num_train": len(train_idx),
        "num_val": len(val_idx),
    }
    with open(os.path.join(DATA_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
