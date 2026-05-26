"""
Supervised Fine-Tuning (SFT) on instruction-following data.

Loads a pre-trained checkpoint and fine-tunes on chat conversations,
computing loss only on assistant responses.

Usage:
    python -m sft.train_sft
    python -m sft.train_sft --base_checkpoint /path/to/step_010000.pt
    python -m sft.train_sft --wandb
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.config import ModelConfig, TrainConfig
from model.transformer import NanoModel
from sft.dataset import SFTDataset


def get_lr(step: int, max_steps: int, max_lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")
    loss = loss.view_as(targets)
    loss = (loss * mask).sum() / mask.sum().clamp(min=1)
    return loss


def save_checkpoint(model, optimizer, step, loss, path):
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "train_loss": loss,
        "model_config": vars(ModelConfig()),
    }
    torch.save(checkpoint, os.path.join(path, f"step_{step:06d}.pt"))
    torch.save(checkpoint, os.path.join(path, "latest.pt"))
    print(f"  Saved SFT checkpoint at step {step}")


@torch.no_grad()
def evaluate(model, val_loader, max_batches: int = 20):
    model.eval()
    total_loss = 0.0
    count = 0
    for i, (x, y, mask) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y, mask = x.cuda(), y.cuda(), mask.cuda()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = masked_cross_entropy(logits, y, mask)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / count


def train(args):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # SFT hyperparameters (lower LR, fewer steps than pre-training)
    max_lr = 2e-5
    min_lr = 2e-6
    warmup_steps = 100
    max_steps = args.max_steps
    batch_size = 16
    grad_accum = 2
    log_interval = 10
    eval_interval = 200
    save_interval = 500

    # Load base model
    base_path = args.base_checkpoint
    print(f"Loading base model: {base_path}")
    checkpoint = torch.load(base_path, map_location="cuda", weights_only=False)

    if "model_config" in checkpoint:
        model_cfg = ModelConfig(**checkpoint["model_config"])
    else:
        model_cfg = ModelConfig()

    model = NanoModel(model_cfg).cuda()
    model.load_state_dict(checkpoint["model"])
    print(f"  Loaded from step {checkpoint.get('step', '?')}, loss {checkpoint.get('train_loss', '?'):.4f}")

    # Data
    data_dir = "/lambda/nfs/nanomodel1/data/sft"
    train_ds = SFTDataset(data_dir, "train")
    val_ds = SFTDataset(data_dir, "val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=1, pin_memory=True)

    # Compile
    print("Compiling model...")
    model = torch.compile(model)

    # Optimizer
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if p.dim() >= 2], "weight_decay": 0.1},
        {"params": [p for n, p in model.named_parameters() if p.dim() < 2], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=max_lr, betas=(0.9, 0.95), fused=True)

    # Wandb
    if args.wandb:
        import wandb
        wandb.init(project="nanomodel", name="sft", config={
            "max_lr": max_lr, "max_steps": max_steps, "batch_size": batch_size,
            "grad_accum": grad_accum, "base_checkpoint": base_path,
        })

    print(f"\nSFT config:")
    print(f"  Steps: {max_steps}, LR: {max_lr} -> {min_lr}")
    print(f"  Batch: {batch_size} x {grad_accum} accum")
    print(f"  Train examples: {len(train_ds)}, Val examples: {len(val_ds)}")
    print()

    model.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    t0 = time.time()
    ckpt_dir = os.path.join(TrainConfig.checkpoint_dir, "sft")

    for step in range(max_steps):
        lr = get_lr(step, max_steps, max_lr, min_lr, warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(grad_accum):
            try:
                x, y, mask = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y, mask = next(train_iter)

            x, y, mask = x.cuda(), y.cuda(), mask.cuda()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = masked_cross_entropy(logits, y, mask) / grad_accum

            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        running_loss += accum_loss

        if (step + 1) % log_interval == 0:
            dt = time.time() - t0
            avg_loss = running_loss / log_interval
            print(f"step {step+1:5d} | loss {avg_loss:.4f} | lr {lr:.2e} | {dt:.1f}s")
            if args.wandb:
                import wandb
                wandb.log({"sft/loss": avg_loss, "sft/lr": lr}, step=step+1)
            running_loss = 0.0
            t0 = time.time()

        if (step + 1) % eval_interval == 0:
            val_loss = evaluate(model, val_loader)
            print(f"  val loss: {val_loss:.4f}")
            if args.wandb:
                import wandb
                wandb.log({"sft/val_loss": val_loss}, step=step+1)
            t0 = time.time()

        if (step + 1) % save_interval == 0:
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            save_checkpoint(raw_model, optimizer, step + 1, accum_loss, ckpt_dir)
            t0 = time.time()

    # Final save
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    save_checkpoint(raw_model, optimizer, max_steps, accum_loss, ckpt_dir)

    if args.wandb:
        import wandb
        wandb.finish()

    print("\nSFT complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_checkpoint", type=str,
                        default="/lambda/nfs/nanomodel1/checkpoints/pretrain/latest.pt")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    train(args)
