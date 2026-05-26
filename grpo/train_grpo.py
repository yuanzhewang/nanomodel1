"""
Group Relative Policy Optimization (GRPO) training.

DeepSeek-style RL alignment without a separate reward model.
For each prompt, generates G completions, scores them with a reward function,
computes group-relative advantages, and updates the policy with clipped surrogate loss.

Usage:
    python -m grpo.train_grpo
    python -m grpo.train_grpo --base_checkpoint /path/to/sft/latest.pt
"""

import argparse
import math
import os
import time

import tiktoken
import torch
import torch.nn.functional as F

from configs.config import ModelConfig, TrainConfig
from model.transformer import NanoModel


USER_START = "<|user|>"
ASSISTANT_START = "<|assistant|>"
TURN_END = "<|end|>"


def get_lr(step: int, max_steps: int, max_lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def load_prompts(path: str) -> list[str]:
    """Load prompts from a text file (one per line)."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def format_prompt(prompt: str) -> str:
    return f"{USER_START}\n{prompt}{TURN_END}\n{ASSISTANT_START}\n"


def reward_fn(prompt: str, completion: str) -> float:
    """
    Simple reward function combining multiple signals.
    In production you'd use a trained reward model; here we use heuristics
    that encourage helpful, well-structured responses.
    """
    reward = 0.0

    # Length: prefer responses that aren't too short or too long
    words = completion.split()
    if len(words) < 5:
        reward -= 1.0
    elif len(words) < 20:
        reward += 0.5
    elif len(words) < 100:
        reward += 1.0
    elif len(words) < 200:
        reward += 0.5
    else:
        reward -= 0.5

    # Coherence: penalize repetition
    sentences = completion.split(".")
    if len(sentences) > 2:
        unique_ratio = len(set(sentences)) / len(sentences)
        reward += unique_ratio * 0.5

    # Format: reward structured responses
    if any(c in completion for c in ["1.", "2.", "- ", "* "]):
        reward += 0.3

    # Penalize empty or degenerate
    if len(completion.strip()) < 10:
        reward -= 2.0

    # Penalize excessive repetition of same word
    if words:
        most_common_ratio = max(words.count(w) for w in set(words)) / len(words)
        if most_common_ratio > 0.3:
            reward -= 1.0

    return reward


@torch.no_grad()
def generate_completions(model, enc, prompt_tokens: torch.Tensor, num_completions: int,
                         max_new_tokens: int = 128, temperature: float = 0.8):
    """Generate multiple completions for a prompt."""
    # Repeat prompt for batch generation
    batch_prompts = prompt_tokens.repeat(num_completions, 1)
    prompt_len = batch_prompts.shape[1]

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(batch_prompts, max_new_tokens=max_new_tokens,
                                    temperature=temperature, top_k=50)

    completions = []
    completion_ids = []
    for i in range(num_completions):
        gen_ids = output_ids[i, prompt_len:].tolist()
        text = enc.decode(gen_ids)
        if TURN_END in text:
            text = text[:text.index(TURN_END)]
        completions.append(text.strip())
        completion_ids.append(output_ids[i])

    return completions, completion_ids


def compute_sequence_log_probs(model, input_ids: torch.Tensor, start_pos: int):
    """Compute log probs for tokens from start_pos onward."""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(input_ids)

    # Only care about positions from start_pos onward
    logits = logits[:, start_pos - 1:-1, :]  # shifted for next-token prediction
    targets = input_ids[:, start_pos:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # Mask padding (eot tokens at the end)
    mask = (targets != 0).float()  # crude mask, will refine if needed
    return (token_log_probs * mask).sum(dim=-1), mask.sum(dim=-1)


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
    print(f"  Saved GRPO checkpoint at step {step}")


def prepare_default_prompts():
    """Generate a set of diverse prompts for GRPO training."""
    prompts = [
        "Explain how photosynthesis works.",
        "What are the main causes of climate change?",
        "Write a short story about a robot learning to paint.",
        "What is the difference between machine learning and deep learning?",
        "How does a computer processor work?",
        "Explain the water cycle in simple terms.",
        "What are the benefits of regular exercise?",
        "Describe the process of making bread from scratch.",
        "What causes earthquakes?",
        "Explain how the internet works.",
        "What is the theory of evolution?",
        "How do vaccines work?",
        "Explain the concept of supply and demand.",
        "What is DNA and why is it important?",
        "How does electricity reach our homes?",
        "What are black holes?",
        "Explain the greenhouse effect.",
        "How does memory work in the human brain?",
        "What causes the seasons to change?",
        "Describe how a car engine works.",
        "What is artificial intelligence?",
        "How do antibiotics work?",
        "Explain the basics of quantum mechanics.",
        "What is the scientific method?",
        "How do airplanes fly?",
        "What causes inflation?",
        "Explain how solar panels generate electricity.",
        "What is the Big Bang theory?",
        "How does the human immune system work?",
        "Explain the concept of gravity.",
    ]
    return prompts


def train(args):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # GRPO hyperparameters
    max_lr = 1e-6
    min_lr = 1e-7
    warmup_steps = 20
    max_steps = args.max_steps
    group_size = 8         # G: number of completions per prompt
    max_gen_tokens = 128
    clip_eps = 0.2         # PPO-style clipping
    kl_coeff = 0.05        # KL penalty coefficient
    log_interval = 5
    save_interval = 200

    # Load SFT model
    base_path = args.base_checkpoint
    print(f"Loading SFT model: {base_path}")
    checkpoint = torch.load(base_path, map_location="cuda", weights_only=False)

    if "model_config" in checkpoint:
        model_cfg = ModelConfig(**checkpoint["model_config"])
    else:
        model_cfg = ModelConfig()

    # Policy model
    policy_model = NanoModel(model_cfg).cuda()
    policy_model.load_state_dict(checkpoint["model"])

    # Reference model (frozen)
    ref_model = NanoModel(model_cfg).cuda()
    ref_model.load_state_dict(checkpoint["model"])
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    enc = tiktoken.get_encoding("gpt2")

    # Prompts
    if args.prompts_file and os.path.exists(args.prompts_file):
        prompts = load_prompts(args.prompts_file)
    else:
        prompts = prepare_default_prompts()
    print(f"  {len(prompts)} training prompts")

    # Optimizer
    param_groups = [
        {"params": [p for n, p in policy_model.named_parameters() if p.dim() >= 2], "weight_decay": 0.01},
        {"params": [p for n, p in policy_model.named_parameters() if p.dim() < 2], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=max_lr, betas=(0.9, 0.95), fused=True)

    if args.wandb:
        import wandb
        wandb.init(project="nanomodel", name="grpo", config={
            "max_lr": max_lr, "max_steps": max_steps, "group_size": group_size,
            "clip_eps": clip_eps, "kl_coeff": kl_coeff,
        })

    print(f"\nGRPO config:")
    print(f"  Steps: {max_steps}, Group size: {group_size}")
    print(f"  Clip: {clip_eps}, KL coeff: {kl_coeff}")
    print(f"  LR: {max_lr}, Gen tokens: {max_gen_tokens}")
    print()

    running_reward = 0.0
    running_loss = 0.0
    t0 = time.time()
    ckpt_dir = os.path.join(TrainConfig.checkpoint_dir, "grpo")

    for step in range(max_steps):
        lr = get_lr(step, max_steps, max_lr, min_lr, warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Sample a prompt
        prompt = prompts[step % len(prompts)]
        formatted = format_prompt(prompt)
        prompt_tokens = torch.tensor([enc.encode(formatted, allowed_special="all")],
                                     dtype=torch.long, device="cuda")
        prompt_len = prompt_tokens.shape[1]

        # Generate G completions with current policy (no grad)
        policy_model.eval()
        completions, completion_ids = generate_completions(
            policy_model, enc, prompt_tokens, group_size, max_gen_tokens
        )
        policy_model.train()

        # Score completions
        rewards = torch.tensor([reward_fn(prompt, c) for c in completions], device="cuda")

        # Group-relative advantages: normalize within group
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Pad completion sequences to same length
        max_len = max(ids.shape[0] for ids in completion_ids)
        max_len = min(max_len, model_cfg.max_seq_len)
        padded_ids = torch.full((group_size, max_len), enc.eot_token, dtype=torch.long, device="cuda")
        for i, ids in enumerate(completion_ids):
            seq_len = min(ids.shape[0], max_len)
            padded_ids[i, :seq_len] = ids[:seq_len]

        # Compute old log probs (detached, from current policy before update)
        with torch.no_grad():
            old_log_probs, token_counts = compute_sequence_log_probs(policy_model, padded_ids, prompt_len)

        # Compute new log probs and reference log probs
        new_log_probs, _ = compute_sequence_log_probs(policy_model, padded_ids, prompt_len)
        with torch.no_grad():
            ref_log_probs, _ = compute_sequence_log_probs(ref_model, padded_ids, prompt_len)

        # Normalize by sequence length
        norm_new = new_log_probs / token_counts.clamp(min=1)
        norm_old = old_log_probs / token_counts.clamp(min=1)
        norm_ref = ref_log_probs / token_counts.clamp(min=1)

        # PPO-style clipped surrogate loss
        ratio = torch.exp(norm_new - norm_old)
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        surrogate = torch.min(ratio * advantages, clipped_ratio * advantages)
        policy_loss = -surrogate.mean()

        # KL penalty (policy vs reference)
        kl = (norm_new - norm_ref).mean()
        kl_loss = kl_coeff * kl

        loss = policy_loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()

        running_reward += rewards.mean().item()
        running_loss += loss.item()

        if (step + 1) % log_interval == 0:
            dt = time.time() - t0
            avg_reward = running_reward / log_interval
            avg_loss = running_loss / log_interval
            print(f"step {step+1:5d} | loss {avg_loss:.4f} | reward {avg_reward:.3f} | "
                  f"kl {kl.item():.4f} | lr {lr:.2e} | {dt:.1f}s")
            if args.wandb:
                import wandb
                wandb.log({
                    "grpo/loss": avg_loss, "grpo/reward": avg_reward,
                    "grpo/kl": kl.item(), "grpo/lr": lr,
                }, step=step+1)
            running_reward = 0.0
            running_loss = 0.0
            t0 = time.time()

        if (step + 1) % save_interval == 0:
            raw_model = policy_model._orig_mod if hasattr(policy_model, "_orig_mod") else policy_model
            save_checkpoint(raw_model, optimizer, step + 1, loss.item(), ckpt_dir)
            t0 = time.time()

    # Final save
    raw_model = policy_model._orig_mod if hasattr(policy_model, "_orig_mod") else policy_model
    save_checkpoint(raw_model, optimizer, max_steps, loss.item(), ckpt_dir)

    if args.wandb:
        import wandb
        wandb.finish()

    print("\nGRPO training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_checkpoint", type=str,
                        default="/lambda/nfs/nanomodel1/checkpoints/sft/latest.pt")
    parser.add_argument("--prompts_file", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    train(args)
