"""GGUF backend built on llama-cpp-python."""

from __future__ import annotations

import threading
from pathlib import Path

QUANTIZATION_MARKERS = (
    "q2_k",
    "q3_k_m",
    "q3_k_s",
    "q4_0",
    "q4_k_m",
    "q4_k_s",
    "q5_0",
    "q5_k_m",
    "q5_k_s",
    "q6_k",
    "q8_0",
    "fp16",
    "f16",
)


def _as_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _default_chat_apply(template, messages, add_assistant: bool) -> str:
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    formatter = Jinja2ChatFormatter(
        template,
        eos_token="",
        bos_token="",
        add_generation_prompt=add_assistant,
    )
    return _as_text(formatter(messages=messages).prompt)


class LlamaCppBackend:
    backend_name = "llama_cpp"

    def __init__(
        self,
        model: str,
        max_input_tokens: int = 4096,
        n_gpu_layers: int = 0,
        n_threads: int | None = None,
        n_batch: int = 512,
        chat_template: str | None = None,
        llama_factory=None,
        chat_apply=None,
        model_id: str | None = None,
    ):
        if llama_factory is None:
            from llama_cpp import Llama as llama_factory
        self.model_path = str(model)
        self.model_id = model_id or Path(self.model_path).name
        self._chat_template = chat_template
        self._chat_apply = chat_apply or _default_chat_apply
        self._n_gpu_layers = n_gpu_layers
        self._lock = threading.Lock()
        self._llama = llama_factory(
            model_path=self.model_path,
            n_ctx=max_input_tokens,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_batch=n_batch,
            logits_all=False,
            verbose=False,
        )
        self.max_context_tokens = int(self._llama.n_ctx())
        self._vocab_size = int(self._llama.n_vocab())
        self.is_cuda = self._detect_cuda()

    def _detect_cuda(self) -> bool:
        if not self._n_gpu_layers:
            return False
        try:
            import llama_cpp

            return bool(llama_cpp.llama_supports_gpu_offload())
        except (AttributeError, ImportError, RuntimeError):
            return True

    def _tokenize(self, text: str, add_bos: bool, special: bool) -> list[int]:
        return list(
            self._llama.tokenize(
                text.encode("utf-8"),
                add_bos=add_bos,
                special=special,
            )
        )

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tokenize(text, add_bos=add_special_tokens, special=False)

    def special_token_ids(self) -> set[int]:
        ids = set()
        for name in ("token_bos", "token_eos"):
            try:
                ids.add(int(getattr(self._llama, name)()))
            except (AttributeError, TypeError, ValueError):
                pass
        return ids

    def format_prompt(self, prompt: str) -> str:
        template = self._chat_template
        if template is None:
            metadata = getattr(self._llama, "metadata", None) or {}
            template = metadata.get("tokenizer.chat_template")
        if not template:
            return prompt
        try:
            rendered = self._chat_apply(template, [{"role": "user", "content": prompt}], True)
        except Exception as error:
            raise ValueError("failed to apply the GGUF chat template") from error
        return rendered

    def _prompt_tokens(self, prompt: str) -> list[int]:
        tokens = self._tokenize(self.format_prompt(prompt), add_bos=False, special=True)
        if not tokens:
            raise ValueError("prompt must tokenize to at least one token")
        if len(tokens) > self.max_context_tokens:
            raise ValueError(f"prompt exceeds {self.max_context_tokens} tokens; reduce the input")
        return tokens

    def _logits_array(self):
        import numpy as np

        values = getattr(self._llama, "logits", None)
        if values is None:
            import llama_cpp

            ctx = getattr(self._llama, "ctx", None)
            if ctx is None:
                ctx = self._llama._ctx.ctx
            values = llama_cpp.llama_get_logits_ith(ctx, -1)
            values = np.ctypeslib.as_array(values, shape=(self._vocab_size,))
        else:
            values = np.asarray(values, dtype=np.float32)
        if values.ndim > 1:
            values = values[-1]
        return values

    def _logits(self) -> list[float]:
        return [float(value) for value in self._logits_array()]

    def next_token_logits(self, prompt: str) -> list[float]:
        tokens = self._prompt_tokens(prompt)
        with self._lock:
            self._llama.reset()
            self._llama.eval(tokens)
            return self._logits()

    def sequence_logprob(self, prompt: str, choice: str) -> float:
        prefix = self._prompt_tokens(prompt)
        targets = self.encode(choice, add_special_tokens=False)
        if not targets:
            raise ValueError("choice must tokenize to at least one token")
        if len(prefix) + len(targets) > self.max_context_tokens:
            raise ValueError(f"prompt and choice exceed {self.max_context_tokens} tokens")
        with self._lock:
            self._llama.reset()
            self._llama.eval(prefix)
            import numpy as np

            scores = []
            for target in targets:
                row = self._logits_array()
                logsumexp = row.max() + np.log(np.exp(row - row.max()).sum())
                scores.append(float(row[target]) - float(logsumexp))
                self._llama.eval([target])
            return sum(scores) / len(scores)

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, int, bool]:
        formatted = self.format_prompt(prompt)
        with self._lock:
            self._llama.reset()
            result = self._llama.create_completion(
                formatted,
                max_tokens=max_new_tokens,
                temperature=0.0,
                top_k=1,
                echo=False,
            )
            self._llama.reset()
        choice = result["choices"][0]
        usage = result.get("usage") or {}
        tokens = int(usage.get("completion_tokens", 0))
        hit_limit = choice.get("finish_reason") == "length"
        return str(choice.get("text", "")), tokens, hit_limit

    def close(self) -> None:
        close = getattr(self._llama, "close", None)
        if close is not None:
            close()

    def synchronize(self) -> None:
        pass

    def memory_allocated(self) -> float | None:
        return None

    def reset_peak_memory(self) -> None:
        pass

    def max_memory_allocated(self) -> float | None:
        return None

    def max_memory_reserved(self) -> float | None:
        return None

    def device_properties(self) -> dict:
        try:
            import torch

            if torch.cuda.is_available():
                properties = torch.cuda.get_device_properties(0)
                return {
                    "name": properties.name,
                    "total_memory": properties.total_memory,
                    "compute_capability": [properties.major, properties.minor],
                }
        except (ImportError, RuntimeError):
            pass
        return {"name": "llama.cpp", "total_memory": None, "compute_capability": None}

    def quantization(self) -> str | None:
        name = Path(self.model_path).name.lower()
        for marker in QUANTIZATION_MARKERS:
            if marker in name:
                return marker.upper()
        metadata = getattr(self._llama, "metadata", None) or {}
        value = metadata.get("general.file_type")
        return str(value) if value is not None else None

    def runtime_metadata(self) -> dict:
        try:
            import llama_cpp

            backend_version = getattr(llama_cpp, "__version__", None)
        except ImportError:
            backend_version = None
        return {
            "backend": self.backend_name,
            "backend_version": backend_version,
            "quantization": self.quantization(),
            "dtype": None,
            "model_revision": None,
            "model_path": self.model_path,
            "n_gpu_layers": self._n_gpu_layers,
            "n_ctx": self.max_context_tokens,
        }
