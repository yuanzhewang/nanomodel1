"""
Evaluate perplexity of a checkpoint on the validation set.

Usage:
    python -m scripts.eval_perplexity
    python -m scripts.eval_perplexity --checkpoint /path/to/step_010000.pt
    python -m scripts.eval_perplexity --num_batches 100
"""

import argparse
import math

import torch
from torch.utils.data import DataLoader

from configs.config import ModelConfig, TrainConfig
from data.dataset import PretrainDataset
from model.transformer import NanoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/lambda/nfs/nanomodel1/checkpoints/pretrain/latest.pt")
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    train_cfg = TrainConfig()
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)

    if "model_config" in checkpoint:
        model_cfg = ModelConfig(**checkpoint["model_config"])
    else:
        model_cfg = ModelConfig()

    model = NanoModel(model_cfg).cuda()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    data_dir = f"{train_cfg.data_dir}/fineweb-edu"
    val_ds = PretrainDataset(data_dir, "val", model_cfg.max_seq_len)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    total_loss = 0.0
    total_tokens = 0
    print(f"Evaluating {args.checkpoint}")
    print(f"  {args.num_batches} batches x {args.batch_size} x {model_cfg.max_seq_len} tokens")

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= args.num_batches:
                break
            x, y = x.cuda(), y.cuda()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            total_loss += loss.item() * x.shape[0] * model_cfg.max_seq_len
            total_tokens += x.shape[0] * model_cfg.max_seq_len

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    print(f"\nResults:")
    print(f"  Avg loss:    {avg_loss:.4f}")
    print(f"  Perplexity:  {perplexity:.2f}")
    print(f"  Tokens eval: {total_tokens:,}")


if __name__ == "__main__":
    main()
