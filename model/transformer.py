import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    angles = torch.outer(t, freqs)  # (max_seq_len, dim//2)
    cos = angles.cos()
    sin = angles.sin()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # x: (B, n_heads, T, head_dim)
    T = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim//2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * torch.cat([cos, cos], dim=-1) + rotated * torch.cat([sin, sin], dim=-1))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.n_embd // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(config.n_embd, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.n_embd, bias=False)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # expand KV heads to match Q heads (GQA)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.ffn = SwiGLU(config.n_embd, config.intermediate_size)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class NanoModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.tok_emb.weight

        # precompute RoPE frequencies
        head_dim = config.n_embd // config.n_heads
        cos, sin = precompute_rope_freqs(head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        x = self.tok_emb(idx)

        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    def count_parameters(self):
        # don't double-count tied weights
        total = sum(p.numel() for p in self.parameters())
        return total
