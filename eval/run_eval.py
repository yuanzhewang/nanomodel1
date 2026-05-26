"""
Evaluate checkpoints using lm-evaluation-harness.

Runs standard benchmarks across all training phases to measure
how each stage affects model capabilities.

Usage:
    python -m eval.run_eval                                      # eval latest pretrain
    python -m eval.run_eval --checkpoint /path/to/sft/latest.pt  # eval specific checkpoint
    python -m eval.run_eval --all                                # eval all phases
    python -m eval.run_eval --tasks hellaswag,arc_easy           # specific tasks
"""

import argparse
import json
import os

import torch

from configs.config import ModelConfig
from model.transformer import NanoModel


class NanoModelWrapper(torch.nn.Module):
    """Wrapper to make NanoModel compatible with lm-eval-harness."""

    def __init__(self, model, model_cfg):
        super().__init__()
        self.model = model
        self.config = model_cfg

    def forward(self, input_ids, **kwargs):
        logits, _ = self.model(input_ids)
        return logits


def build_lm_eval_model(checkpoint_path: str):
    """Load checkpoint and wrap for lm-eval-harness."""
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)

    if "model_config" in checkpoint:
        model_cfg = ModelConfig(**checkpoint["model_config"])
    else:
        model_cfg = ModelConfig()

    model = NanoModel(model_cfg).cuda()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, model_cfg


def eval_checkpoint(checkpoint_path: str, tasks: list[str], num_fewshot: int = 0, limit: int = None):
    """Run lm-eval-harness on a checkpoint."""
    import lm_eval
    import tiktoken

    print(f"\nEvaluating: {checkpoint_path}")
    print(f"  Tasks: {', '.join(tasks)}")

    model, model_cfg = build_lm_eval_model(checkpoint_path)
    enc = tiktoken.get_encoding("gpt2")

    # Use the simple_evaluate with a custom model
    # We need to wrap our model to be compatible with lm-eval
    from eval.lm_eval_adapter import NanoModelLM

    lm = NanoModelLM(model, enc, model_cfg, batch_size=8)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=8,
    )

    return results


def print_results(results: dict, label: str = ""):
    """Pretty-print evaluation results."""
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

    if "results" in results:
        for task, metrics in results["results"].items():
            print(f"\n  {task}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"    {metric}: {value:.4f}")
                elif metric != "alias":
                    print(f"    {metric}: {value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--all", action="store_true", help="Evaluate all phase checkpoints")
    parser.add_argument("--tasks", type=str, default="hellaswag,arc_easy,piqa,winogrande")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=200, help="Max examples per task (None for full)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]
    ckpt_dir = "/lambda/nfs/nanomodel1/checkpoints"

    if args.all:
        # Evaluate all phases
        phases = [
            ("Pretrained", os.path.join(ckpt_dir, "pretrain", "latest.pt")),
            ("SFT", os.path.join(ckpt_dir, "sft", "latest.pt")),
            ("DPO", os.path.join(ckpt_dir, "dpo", "latest.pt")),
            ("GRPO", os.path.join(ckpt_dir, "grpo", "latest.pt")),
        ]

        all_results = {}
        for label, path in phases:
            if os.path.exists(path):
                results = eval_checkpoint(path, tasks, args.num_fewshot, args.limit)
                print_results(results, label)
                all_results[label] = results.get("results", {})
            else:
                print(f"\n  Skipping {label}: {path} not found")

        # Summary table
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        print(f"{'Task':<20}", end="")
        for label, _ in phases:
            if label in all_results:
                print(f"{label:>12}", end="")
        print()
        print("-" * 68)

        for task in tasks:
            print(f"{task:<20}", end="")
            for label, _ in phases:
                if label in all_results and task in all_results[label]:
                    # Try to find accuracy metric
                    metrics = all_results[label][task]
                    acc = metrics.get("acc,none", metrics.get("acc_norm,none", None))
                    if acc is not None:
                        print(f"{acc:>11.4f}", end=" ")
                    else:
                        print(f"{'N/A':>12}", end="")
                else:
                    print(f"{'---':>12}", end="")
            print()

        if args.output:
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to {args.output}")

    else:
        # Single checkpoint
        path = args.checkpoint or os.path.join(ckpt_dir, "pretrain", "latest.pt")
        results = eval_checkpoint(path, tasks, args.num_fewshot, args.limit)
        print_results(results)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results.get("results", {}), f, indent=2)


if __name__ == "__main__":
    main()
