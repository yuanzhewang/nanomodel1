"""
Adapter to make NanoModel work with lm-evaluation-harness.
"""

import torch
import torch.nn.functional as F
import tiktoken
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


class NanoModelLM(LM):
    def __init__(self, model, enc: tiktoken.Encoding, model_cfg, batch_size: int = 8):
        super().__init__()
        self._model = model
        self._enc = enc
        self._model_cfg = model_cfg
        self._batch_size = batch_size
        self._device = next(model.parameters()).device

    @property
    def eot_token_id(self):
        return self._enc.eot_token

    @property
    def max_length(self):
        return self._model_cfg.max_seq_len

    @property
    def max_gen_toks(self):
        return 128

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return self._enc.encode(string, allowed_special="all")

    def tok_decode(self, tokens: list[int], **kwargs) -> str:
        return self._enc.decode(tokens)

    def _model_call(self, inps: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = self._model(inps)
        return logits

    def _model_generate(self, context, max_length, stop, **kwargs):
        raise NotImplementedError("Generation not used for these benchmarks")

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        results = []
        for chunk_start in range(0, len(requests), self._batch_size):
            chunk = requests[chunk_start:chunk_start + self._batch_size]
            batch_results = self._loglikelihood_batch(chunk)
            results.extend(batch_results)
        return results

    def _loglikelihood_batch(self, requests) -> list[tuple[float, bool]]:
        results = []
        for req in requests:
            context, continuation = req.args

            ctx_tokens = self.tok_encode(context)
            cont_tokens = self.tok_encode(continuation)
            all_tokens = (ctx_tokens + cont_tokens)[-self.max_length:]

            input_ids = torch.tensor([all_tokens[:-1]], dtype=torch.long, device=self._device)
            targets = torch.tensor([all_tokens[1:]], dtype=torch.long, device=self._device)

            logits = self._model_call(input_ids)
            log_probs = F.log_softmax(logits, dim=-1)

            # Only score the continuation tokens
            cont_start = len(all_tokens) - len(cont_tokens) - 1
            cont_log_probs = log_probs[0, cont_start:]
            cont_targets = targets[0, cont_start:]

            token_log_probs = cont_log_probs.gather(-1, cont_targets.unsqueeze(-1)).squeeze(-1)
            total_ll = token_log_probs.sum().item()

            # Check if continuation is the greedy choice
            greedy = cont_log_probs.argmax(dim=-1)
            is_greedy = (greedy == cont_targets).all().item()

            results.append((total_ll, is_greedy))

        return results

    def loglikelihood_rolling(self, requests) -> list[tuple[float, bool]]:
        results = []
        for req in requests:
            text = req.args[0]
            tokens = self.tok_encode(text)
            tokens = tokens[-self.max_length:]

            input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=self._device)
            targets = torch.tensor([tokens[1:]], dtype=torch.long, device=self._device)

            logits = self._model_call(input_ids)
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            results.append((token_log_probs.sum().item(), True))

        return results

    def generate_until(self, requests) -> list[str]:
        results = []
        for req in requests:
            context = req.args[0]
            tokens = self.tok_encode(context)[-self.max_length:]
            idx = torch.tensor([tokens], dtype=torch.long, device=self._device)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = self._model.generate(idx, max_new_tokens=self.max_gen_toks, temperature=0.0, top_k=1)

            gen_tokens = out[0, len(tokens):].tolist()
            results.append(self.tok_decode(gen_tokens))

        return results
