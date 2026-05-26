from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    n_kv_heads: int = 4  # grouped query attention
    intermediate_size: int = 2048  # SwiGLU hidden dim
    max_seq_len: int = 1024
    dropout: float = 0.0
    rope_theta: float = 10000.0


@dataclass
class TrainConfig:
    # Paths
    data_dir: str = "/lambda/nfs/nanomodel1/data"
    checkpoint_dir: str = "/lambda/nfs/nanomodel1/checkpoints"

    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 1000
    max_steps: int = 50000
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 5000
    eval_steps: int = 20

    # System
    dtype: str = "bfloat16"
    compile: bool = True
    seed: int = 42
