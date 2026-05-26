"""
Direct Preference Optimization (DPO) training.

Loads an SFT checkpoint as both reference (frozen) and policy (trainable).
Optimizes the policy to prefer chosen over rejected responses.

Usage:
    python -m dpo.train_dpo
    python -m dpo.train_dpo --base_checkpoint /path/to/sft/latest.pt
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
from dpo.dataset import DPODataset


def get_lr(step: int, max_steps: int, max_lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def compute_log_probs(model, input_ids, targets, mask):
    """Compute per-token log probabilities, masked and summed."""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(input_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather log probs for actual target tokens
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # Apply mask and sum per sequence
    return (token_log_probs * mask).sum(dim=-1)


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta: float = 0.1):
    """
    DPO loss: -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
    where log_ratio = log_pi(y|x) - log_ref(y|x)
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    # Metrics
    with torch.no_grad():
        chosen_better = (chosen_rewards > rejected_rewards).float().mean()
        reward_margin = (chosen_rewards - rejected_rewards).mean()

    return loss, chosen_better.item(), reward_margin.item()


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
    print(f"  Saved DPO checkpoint at step {step}")


@torch.no_grad()
def evaluate(policy_model, ref_model, val_loader, beta: float, max_batches: int = 20):
    policy_model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        c_x, c_y, c_m, r_x, r_y, r_m = [t.cuda() for t in batch]

        policy_chosen = compute_log_probs(policy_model, c_x, c_y, c_m)
        policy_rejected = compute_log_probs(policy_model, r_x, r_y, r_m)
        ref_chosen = compute_log_probs(ref_model, c_x, c_y, c_m)
        ref_rejected = compute_log_probs(ref_model, r_x, r_y, r_m)

        loss, acc, _ = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
        total_loss += loss.item()
        total_acc += acc
        count += 1

    policy_model.train()
    return total_loss / count, total_acc / count


def train(args):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # DPO hyperparameters
    max_lr = 5e-6
    min_lr = 5e-7
    warmup_steps = 50
    max_steps = args.max_steps
    batch_size = 4
    grad_accum = 4
    beta = 0.1
    log_interval = 10
    eval_interval = 200
    save_interval = 500

    # Load SFT model
    base_path = args.base_checkpoint
    print(f"Loading SFT model: {base_path}")
    checkpoint = torch.load(base_path, map_location="cuda", weights_only=False)

    if "model_config" in checkpoint:
        model_cfg = ModelConfig(**checkpoint["model_config"])
    else:
        model_cfg = ModelConfig()

    # Policy model (trainable)
    policy_model = NanoModel(model_cfg).cuda()
    policy_model.load_state_dict(checkpoint["model"])

    # Reference model (frozen copy of SFT)
    ref_model = NanoModel(model_cfg).cuda()
    ref_model.load_state_dict(checkpoint["model"])
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    print(f"  Loaded from step {checkpoint.get('step', '?')}")
    print(f"  Policy params: {sum(p.numel() for p in policy_model.parameters()):,}")

    # Data
    data_dir = "/lambda/nfs/nanomodel1/data/dpo"
    train_ds = DPODataset(data_dir, "train")
    val_ds = DPODataset(data_dir, "val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=1, pin_memory=True)

    # Compile both models
    print("Compiling models...")
    policy_model = torch.compile(policy_model)
    ref_model = torch.compile(ref_model)

    # Optimizer (only policy model)
    param_groups = [
        {"params": [p for n, p in policy_model.named_parameters() if p.dim() >= 2], "weight_decay": 0.1},
        {"params": [p for n, p in policy_model.named_parameters() if p.dim() < 2], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=max_lr, betas=(0.9, 0.95), fused=True)

    if args.wandb:
        import wandb
        wandb.init(project="nanomodel", name="dpo", config={
            "max_lr": max_lr, "max_steps": max_steps, "beta": beta,
            "batch_size": batch_size, "grad_accum": grad_accum,
        })

    print(f"\nDPO config:")
    print(f"  Steps: {max_steps}, Beta: {beta}, LR: {max_lr}")
    print(f"  Batch: {batch_size} x {grad_accum} accum")
    print(f"  Train pairs: {len(train_ds)}, Val pairs: {len(val_ds)}")
    print()

    policy_model.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    running_acc = 0.0
    t0 = time.time()
    ckpt_dir = os.path.join(TrainConfig.checkpoint_dir, "dpo")

    for step in range(max_steps):
        lr = get_lr(step, max_steps, max_lr, min_lr, warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        accum_loss = 0.0
        accum_acc = 0.0

        for _ in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            c_x, c_y, c_m, r_x, r_y, r_m = [t.cuda() for t in batch]

            # Forward through both models
            policy_chosen = compute_log_probs(policy_model, c_x, c_y, c_m)
            policy_rejected = compute_log_probs(policy_model, r_x, r_y, r_m)

            with torch.no_grad():
                ref_chosen = compute_log_probs(ref_model, c_x, c_y, c_m)
                ref_rejected = compute_log_probs(ref_model, r_x, r_y, r_m)

            loss, acc, margin = dpo_loss(
                policy_chosen, policy_rejected,
                ref_chosen, ref_rejected, beta
            )
            (loss / grad_accum).backward()
            accum_loss += loss.item() / grad_accum
            accum_acc += acc / grad_accum

        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        running_loss += accum_loss
        running_acc += accum_acc

        if (step + 1) % log_interval == 0:
            dt = time.time() - t0
            avg_loss = running_loss / log_interval
            avg_acc = running_acc / log_interval
            print(f"step {step+1:5d} | loss {avg_loss:.4f} | acc {avg_acc:.3f} | lr {lr:.2e} | {dt:.1f}s")
            if args.wandb:
                import wandb
                wandb.log({"dpo/loss": avg_loss, "dpo/accuracy": avg_acc, "dpo/lr": lr}, step=step+1)
            running_loss = 0.0
            running_acc = 0.0
            t0 = time.time()

        if (step + 1) % eval_interval == 0:
            val_loss, val_acc = evaluate(policy_model, ref_model, val_loader, beta)
            print(f"  val loss: {val_loss:.4f}, val acc: {val_acc:.3f}")
            if args.wandb:
                import wandb
                wandb.log({"dpo/val_loss": val_loss, "dpo/val_accuracy": val_acc}, step=step+1)
            t0 = time.time()

        if (step + 1) % save_interval == 0:
            raw_model = policy_model._orig_mod if hasattr(policy_model, "_orig_mod") else policy_model
            save_checkpoint(raw_model, optimizer, step + 1, accum_loss, ckpt_dir)
            t0 = time.time()

    # Final save
    raw_model = policy_model._orig_mod if hasattr(policy_model, "_orig_mod") else policy_model
    save_checkpoint(raw_model, optimizer, max_steps, accum_loss, ckpt_dir)

    if args.wandb:
        import wandb
        wandb.finish()

    print("\nDPO training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_checkpoint", type=str,
                        default="/lambda/nfs/nanomodel1/checkpoints/sft/latest.pt")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    train(args)
