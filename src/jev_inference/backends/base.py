"""Runtime abstraction for logit-based decision scoring."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    model_id: str
    backend_name: str
    max_context_tokens: int
    is_cuda: bool

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def special_token_ids(self) -> set[int]: ...

    def format_prompt(self, prompt: str) -> str: ...

    def next_token_logits(self, prompt: str) -> list[float]: ...

    def sequence_logprob(self, prompt: str, choice: str) -> float: ...

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, int, bool]: ...

    def close(self) -> None: ...

    def synchronize(self) -> None: ...

    def memory_allocated(self) -> float | None: ...

    def reset_peak_memory(self) -> None: ...

    def max_memory_allocated(self) -> float | None: ...

    def max_memory_reserved(self) -> float | None: ...

    def device_properties(self) -> dict: ...

    def runtime_metadata(self) -> dict: ...


def create_backend(
    name: str,
    model: str,
    device: str = "auto",
    dtype: str = "auto",
    max_input_tokens: int = 4096,
    **options,
) -> Backend:
    if name == "transformers":
        from .transformers_backend import TransformersBackend

        return TransformersBackend(model, device, max_input_tokens, dtype)
    if name == "llama_cpp":
        from .llamacpp_backend import LlamaCppBackend

        return LlamaCppBackend(model, max_input_tokens=max_input_tokens, **options)
    raise ValueError(f"unsupported backend: {name!r}")
