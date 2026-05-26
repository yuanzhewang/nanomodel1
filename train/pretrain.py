"""
Pre-training loop for NanoModel.

Usage:
    python -m train.pretrain
    python -m train.pretrain --resume                    # resume from latest checkpoint
    python -m train.pretrain --max_steps 20000 --wandb   # override config + enable wandb
"""

import argparse
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from configs.config import ModelConfig, TrainConfig
from data.dataset import PretrainDataset
from model.transformer import NanoModel


def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def save_checkpoint(model, optimizer, step, train_loss, cfg: TrainConfig):
    path = os.path.join(cfg.checkpoint_dir, "pretrain")
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "train_loss": train_loss,
        "model_config": vars(ModelConfig()),
    }
    torch.save(checkpoint, os.path.join(path, f"step_{step:06d}.pt"))
    torch.save(checkpoint, os.path.join(path, "latest.pt"))
    print(f"  Saved checkpoint at step {step}")


def load_checkpoint(model, optimizer, cfg: TrainConfig):
    path = os.path.join(cfg.checkpoint_dir, "pretrain", "latest.pt")
    if not os.path.exists(path):
        print("No checkpoint found, starting from scratch")
        return 0
    checkpoint = torch.load(path, map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    print(f"Resumed from step {step} (loss {checkpoint['train_loss']:.4f})")
    return step


@torch.no_grad()
def evaluate(model, val_loader, cfg: TrainConfig):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= cfg.eval_steps:
            break
        x, y = x.cuda(), y.cuda()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(args):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    if args.max_steps:
        train_cfg.max_steps = args.max_steps

    # Data
    data_dir = os.path.join(train_cfg.data_dir, "fineweb-edu")
    train_ds = PretrainDataset(data_dir, "train", model_cfg.max_seq_len)
    val_ds = PretrainDataset(data_dir, "val", model_cfg.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True, drop_last=True)

    # Model
    model = NanoModel(model_cfg).cuda()
    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if p.dim() >= 2], "weight_decay": train_cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if p.dim() < 2], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=train_cfg.max_lr, betas=(0.9, 0.95), fused=True)

    # Resume (before compile so state_dict keys match)
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(model, optimizer, train_cfg)

    if train_cfg.compile:
        print("Compiling model...")
        model = torch.compile(model)

    # Wandb
    if args.wandb:
        import wandb
        wandb.init(project="nanomodel", config={**vars(model_cfg), **vars(train_cfg)})

    # Training
    tokens_per_step = train_cfg.batch_size * train_cfg.gradient_accumulation_steps * model_cfg.max_seq_len
    print(f"\nTraining config:")
    print(f"  Steps: {train_cfg.max_steps:,}, Tokens/step: {tokens_per_step:,}")
    print(f"  Total tokens: {train_cfg.max_steps * tokens_per_step / 1e9:.1f}B")
    print(f"  LR: {train_cfg.max_lr} -> {train_cfg.min_lr}, Warmup: {train_cfg.warmup_steps}")
    print(f"  Batch: {train_cfg.batch_size} x {train_cfg.gradient_accumulation_steps} accum x {model_cfg.max_seq_len} seq")
    print()

    model.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    t0 = time.time()

    for step in range(start_step, train_cfg.max_steps):
        lr = get_lr(step, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0.0
        for micro_step in range(train_cfg.gradient_accumulation_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.cuda(), y.cuda()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / train_cfg.gradient_accumulation_steps

            loss.backward()
            accum_loss += loss.item()

        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

        optimizer.step()
        torch.cuda.synchronize()
        running_loss += accum_loss

        # Logging
        if (step + 1) % train_cfg.log_interval == 0:
            dt = time.time() - t0
            avg_loss = running_loss / train_cfg.log_interval
            tokens_sec = tokens_per_step * train_cfg.log_interval / dt
            print(f"step {step+1:6d} | loss {avg_loss:.4f} | lr {lr:.2e} | {tokens_sec:,.0f} tok/s | {dt:.1f}s")
            if args.wandb:
                import wandb
                wandb.log({"train/loss": avg_loss, "train/lr": lr, "train/tokens_per_sec": tokens_sec}, step=step+1)
            running_loss = 0.0
            t0 = time.time()

        # Eval
        if (step + 1) % train_cfg.eval_interval == 0:
            val_loss = evaluate(model, val_loader, train_cfg)
            print(f"  val loss: {val_loss:.4f}")
            if args.wandb:
                import wandb
                wandb.log({"val/loss": val_loss}, step=step+1)
            t0 = time.time()

        # Save
        if (step + 1) % train_cfg.save_interval == 0:
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            save_checkpoint(raw_model, optimizer, step + 1, accum_loss, train_cfg)
            t0 = time.time()

    # Final save
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    save_checkpoint(raw_model, optimizer, train_cfg.max_steps, accum_loss, train_cfg)

    if args.wandb:
        import wandb
        wandb.finish()

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()
    train(args)
