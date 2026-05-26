"""
Generate text samples from a trained checkpoint.

Usage:
    python -m scripts.generate --prompt "The theory of relativity"
    python -m scripts.generate --interactive
    python -m scripts.generate --checkpoint /path/to/step_010000.pt --prompt "Once upon"
"""

import argparse

import tiktoken
import torch

from configs.config import ModelConfig
from model.transformer import NanoModel


def load_model(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)

    if "model_config" in checkpoint:
        cfg = ModelConfig(**checkpoint["model_config"])
    else:
        cfg = ModelConfig()

    model = NanoModel(cfg).cuda()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, cfg


def generate(model, enc, prompt: str, max_tokens: int = 200, temperature: float = 0.8, top_k: int = 50):
    tokens = enc.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    return enc.decode(out[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/lambda/nfs/nanomodel1/checkpoints/pretrain/latest.pt")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg = load_model(args.checkpoint)
    enc = tiktoken.get_encoding("gpt2")
    print(f"Model loaded ({cfg.n_layers}L, {cfg.n_embd}D, {cfg.n_heads}H)\n")

    if args.interactive:
        print("Interactive mode (type 'quit' to exit)")
        print("-" * 50)
        while True:
            prompt = input("\nPrompt> ")
            if prompt.lower() in ("quit", "exit", "q"):
                break
            text = generate(model, enc, prompt, args.max_tokens, args.temperature, args.top_k)
            print(f"\n{text}")
            print("-" * 50)
    elif args.prompt:
        for i in range(args.num_samples):
            if args.num_samples > 1:
                print(f"--- Sample {i+1} ---")
            text = generate(model, enc, args.prompt, args.max_tokens, args.temperature, args.top_k)
            print(text)
            if args.num_samples > 1:
                print()
    else:
        prompts = [
            "The solar system consists of",
            "In a groundbreaking study, researchers found that",
            "The history of mathematics begins with",
            "Once upon a time in a small village,",
            "To solve this problem, we need to",
        ]
        print("Generating samples from default prompts:\n")
        for prompt in prompts:
            print(f"PROMPT: {prompt}")
            text = generate(model, enc, prompt, args.max_tokens, args.temperature, args.top_k)
            print(f"OUTPUT: {text}\n")
            print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
