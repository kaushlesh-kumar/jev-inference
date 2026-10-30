# Spec: Pluggable inference backends (Transformers and llama.cpp/GGUF)

Status: accepted for implementation
Owner: jev-inference
Scope: `src/jev_inference/`

## 1. Problem

The decision engine currently owns a Transformers tokenizer and `AutoModelForCausalLM`
directly. Its scoring modes depend on three primitives:

1. tokenize text without adding special tokens,
2. obtain next-token logits for a prompt,
3. obtain conditional log-probabilities for every token of a continuation.

Those primitives are not Transformers-specific. Extracting them lets the same API,
prompts, normalization and benchmark definitions run on GGUF models through
llama.cpp, including quantized GGUF files.

## 2. Goals

- Add a backend seam without changing the public `/decide` contract.
- Keep the existing Transformers behavior and benchmark results reproducible.
- Add a llama.cpp backend for GGUF models using `llama-cpp-python`.
- Preserve exact mode semantics: `labels`, `independent`, `sequence_likelihood`.
- Record the backend and quantization in benchmark metadata.
- Add a comparison utility for two JevBench result files.
- Follow TDD: tests define the backend contract before implementation.

## 3. Non-goals

- llama.cpp HTTP server integration.
- HF-native quantized formats (bitsandbytes, GPTQ, AWQ).
- Multimodal or encoder-only models.
- Changing prompt text or the three scoring definitions.
- Making latency or probability values comparable across backends; only accuracy,
  calibration and order-sensitivity are compared, with differences reported.

## 4. Backend contract

A backend must expose:

```python
model_id: str
backend_name: str                 # "transformers" | "llama_cpp"
max_context_tokens: int
is_cuda: bool

def encode(text: str, add_special_tokens: bool = False) -> list[int]
def special_token_ids() -> set[int]
def format_prompt(prompt: str) -> str
def next_token_logits(prompt: str) -> list[float]
def sequence_logprob(prompt: str, choice: str) -> float
def generate(prompt: str, max_new_tokens: int) -> tuple[str, int, bool]
def close() -> None

# Measurement hooks. Return None when a metric is not available.
def synchronize() -> None
def memory_allocated() -> float | None
def reset_peak_memory() -> None
def max_memory_allocated() -> float | None
def max_memory_reserved() -> float | None
def device_properties() -> dict
def runtime_metadata() -> dict
```

`token_ids(labels)` validation remains in the engine: each label must encode to
exactly one distinct non-special token.

## 5. Engine behavior

`DecisionEngine` becomes an orchestrator:

- `backend="transformers"` keeps the current Hugging Face path.
- `backend="llama_cpp"` uses a GGUF file path.
- A caller may pass a backend instance directly (used by tests).
- `max_input_tokens` is `min(requested, backend.max_context_tokens)`.
- `decide`, `normalize`, tie-breaking, response fields and the request lock are unchanged.
- Errors keep the existing mapping: `ValueError -> 422`, other failures -> 503.

## 6. llama.cpp semantics

- Load with `n_ctx`, `n_gpu_layers`, `n_threads`, `n_batch`, `logits_all=False`.
- Tokenize with `add_bos=False`; prompt formatting uses the GGUF chat template with
  an assistant generation prefix, matching the Transformers chat-template path.
- Clear the KV cache before every independent prompt evaluation.
- `next_token_logits`: evaluate prompt, return the last-position logits.
- `sequence_logprob`: evaluate the prompt, then feed continuation tokens one at a time,
  scoring each target from the logits before it is consumed. This includes the first
  continuation token and excludes EOS, matching teacher forcing, without storing
  all-position logits.
- `generate`: greedy completion with the same generated-token limits used by the
  existing generation baselines. Return `(text, generated_tokens, hit_token_limit)`.
- Context overflow raises `ValueError`; it is never truncated.

## 7. Configuration

CLI additions:

- `--backend transformers|llama_cpp`
- `--n-gpu-layers` (llama.cpp; `-1` means all layers)
- `--n-threads`, `--n-batch`, `--chat-template` (optional override)

Existing `--model`, `--dtype`, `--max-input-tokens`, `--device` remain. For
`llama_cpp`, `--model` is a local GGUF path and `--dtype` is ignored.

Modal receives the same values through `JEV_*` environment variables. The Modal
image installs `llama-cpp-python` when the llama.cpp backend is selected.

## 8. Benchmarks and comparison

- `generate_choice` uses `engine.generate` instead of `model.generate`.
- CUDA memory metrics are optional. Transformers reports PyTorch allocator figures;
  llama.cpp reports `null` and the report renders N/A.
- Metadata gains `backend`, `backend_version`, `quantization`, `runtime`, and
  `memory_measurement`.
- `jevbench_compare` compares two result files on accuracy, invalid outputs,
  latency, Brier, ECE and order-change rate. It records both model/backend identities
  and refuses mismatched datasets.

## 9. Acceptance criteria

1. Existing tests pass after adapting tests that reached into Transformers internals.
2. New backend-contract tests pass with fake backends.
3. `ruff check .` and `ruff format --check .` pass.
4. A Qwen2.5-1.5B-Instruct Q5_K_M GGUF can be served by `DecisionEngine` via
   `llama_cpp` and passes a smoke `/decide` request.
5. JevBench runs with `--backend llama_cpp` and writes a result JSON.
6. A comparison document is generated against
   `docs/jevbench-results-qwen2.5-1.5b.json`.
