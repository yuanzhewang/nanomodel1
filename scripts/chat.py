"""
Interactive chat with an SFT-trained model.

Usage:
    python -m scripts.chat
    python -m scripts.chat --checkpoint /path/to/sft/latest.pt
"""

import argparse

import tiktoken
import torch

from configs.config import ModelConfig
from model.transformer import NanoModel


USER_START = "<|user|>"
ASSISTANT_START = "<|assistant|>"
TURN_END = "<|end|>"


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


def chat_generate(model, enc, conversation: str, max_tokens: int = 300,
                  temperature: float = 0.7, top_k: int = 50):
    prompt = conversation + f"{ASSISTANT_START}\n"
    tokens = enc.encode(prompt, allowed_special="all")
    idx = torch.tensor([tokens], dtype=torch.long, device="cuda")

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)

    full_text = enc.decode(out[0].tolist())
    # Extract just the assistant response
    response = full_text[len(prompt):]
    if TURN_END in response:
        response = response[:response.index(TURN_END)]
    return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="/lambda/nfs/nanomodel1/checkpoints/sft/latest.pt")
    parser.add_argument("--max_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print(f"Loading model: {args.checkpoint}")
    model, cfg = load_model(args.checkpoint)
    enc = tiktoken.get_encoding("gpt2")
    print(f"Model loaded. Type 'quit' to exit, 'reset' to clear history.\n")

    conversation = ""

    while True:
        user_input = input("You> ")
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "reset":
            conversation = ""
            print("(conversation reset)\n")
            continue

        conversation += f"{USER_START}\n{user_input}{TURN_END}\n"

        response = chat_generate(model, enc, conversation, args.max_tokens, args.temperature)
        conversation += f"{ASSISTANT_START}\n{response}{TURN_END}\n"

        print(f"Bot> {response}\n")


if __name__ == "__main__":
    main()
