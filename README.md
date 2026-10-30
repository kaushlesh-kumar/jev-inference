# Jev Inference

Use a Hugging Face causal LM to choose from supplied options without training or
an autoregressive decoding loop. Serve it locally or deploy an HTTP API on Modal.
`--model` accepts any compatible text-only `AutoModelForCausalLM` model.

## Deploy on Modal

Requires Python 3.10+ (3.11 recommended) and a Modal account.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[modal]'
python -m modal setup
python run.py --model Qwen/Qwen2.5-0.5B-Instruct --modal --dtype float16
```

The command prints the HTTPS URL; open `/docs` to try the API. The default GPU is
T4. One container serves requests serially and scales down after 300 idle seconds;
model weights persist in a Modal Volume. The first request loads the model.
Use `--gpu`, `--app-name` and `--max-input-tokens` to configure the deployment.
Deploying with the same app name updates it.

The API is public by default. For authentication, put `JEV_API_KEY` in a Modal
Secret and pass `--secret SECRET_NAME`. Add `HF_TOKEN` for private or gated models.

## API

```bash
curl "$JEV_URL/decide" \
  -H 'Content-Type: application/json' \
  -d '{"context":"Enemy nearby. HP 10/100. Healing potion available.","choices":["attack","heal","retreat"],"mode":"labels","temperature":1.0}'
```

With authentication enabled, add `-H "Authorization: Bearer $JEV_API_KEY"`.
The response contains `choice`, `index`, each option's `score` and `probability`,
`model`, `mode`, `latency_ms` and `forward_passes`. `/health` reports readiness.

| Mode | Score | Forward passes |
| --- | --- | --- |
| `labels` | Next-token A/B/C logits; each label represents a complete option | 1 |
| `independent` | Yes-minus-No logit margin for each option separately | One per option |
| `sequence_likelihood` | Mean conditional log-probability over every token of an option | One per option |

Scores are normalized with `softmax(score / temperature)`. These are relative
choice weights, **not calibrated correctness probabilities**. Ties select the
first option. Independent scoring is an inference heuristic, not trained NLI.

Sequence likelihood uses teacher forcing, including the first continuation token
and excluding EOS. Prompt and option tokens are encoded separately; the shared
prompt lists all options. Labels and sequence likelihood can depend on option order.

Requests accept 2–26 unique, nonempty options. Label and Yes/No scoring require
single-token labels. Sequence likelihood supports multi-token options without
that requirement. Inputs exceeding the context limit fail instead of being truncated.
Remote model code is disabled. Models must fit available memory. Transformers models
run in their native dtype; quantized GGUF files run through the optional llama.cpp
backend. Encoder-only and multimodal models are outside the current implementation.

## Local execution

```bash
pip install -e '.[inference]'
python run.py --model Qwen/Qwen2.5-1.5B-Instruct --device auto

# Quantized GGUF through llama.cpp (all layers on the GPU)
pip install -e '.[inference,llamacpp]'
python run.py --backend llama_cpp --model .cache/gguf/qwen2.5-1.5b-instruct-q5_k_m.gguf \
  --n-gpu-layers -1 --n-threads 8 --max-input-tokens 8192
```

The API listens on `http://127.0.0.1:8000`. Options include `--device cpu|cuda|mps`,
`--host`, `--port` and `--max-input-tokens` (default 4096). Set `JEV_API_KEY` to enable
authentication. Serving and benchmark runners accept
`--dtype auto|float16|bfloat16|float32`; `auto` uses the model's configured dtype.
T4 supports native FP16 Tensor Cores, so the Modal example explicitly selects FP16.
For GGUF runs, pass `--backend llama_cpp`, `--n-gpu-layers` (`-1` offloads all layers),
and optionally `--n-threads`, `--n-batch` and `--chat-template`. `--dtype` and
`--device` are ignored by llama.cpp. CUDA allocator memory is reported for Transformers;
llama.cpp memory appears as `N/A`.

## Benchmarks

The public JevBench runner evaluates 231 pinned tasks: 48 easy, 72 original and
111 hard. It retains state, instructions and option rubrics; gold answers and
provenance never enter prompts. Each task is tested in its original order and
one cyclic rotation. Five methods give 2310 measured requests per model.
This is a public-subset experiment, not an official JevBench leaderboard score.

```bash
python run_jevbench.py --modal --model Qwen/Qwen2.5-0.5B-Instruct --dtype float16 --output docs/jevbench-results-qwen2.5-0.5b.json
python run_jevbench.py --modal --model Qwen/Qwen2.5-1.5B-Instruct --dtype float16 --output docs/jevbench-results-qwen2.5-1.5b.json
python -m jev_inference.jevbench_report docs/jevbench-results-qwen2.5-0.5b.json docs/jevbench-results-qwen2.5-1.5b.json --readme
python -m jev_inference.jevbench_compare docs/jevbench-results-qwen2.5-1.5b.json \
  docs/jevbench-results-qwen2.5-1.5b-q5_k_m.json --output docs/jevbench-comparison-qwen2.5-1.5b.md
```

To benchmark the GGUF model, pass `--backend llama_cpp --model PATH --n-gpu-layers -1`
to `run_jevbench.py`. Compare accuracy, calibration and order sensitivity; latency
deltas reflect the runtime and precision choice as well as the measuring hardware.

For a local CUDA GPU, install `.[inference]` and omit `--modal`. Configure
`--repeats`, `--permutations`, `--max-input-tokens` or `--limit` for diagnostic runs.
Benchmark jobs run separately from the deployed API.

<!-- jevbench-results:start -->
## JevBench public-subset diagnostics

Public tasks from [JevBench](https://github.com/fstandhartinger/jevbench), pinned to `bb05a335bc809e61b20c0f745d25499a82b326fc`. This is a public-subset experiment, not an official leaderboard result. Gold answers and provenance never enter prompts.

| Model | Method | Scorer correct | Median ms | p95 ms | Peak MiB | Brier | ECE | Order changes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-0.5B-Instruct | `generate_label` | 93/231 | 50.35 | 257.97 | 1132.3 | N/A | N/A | 61.0% |
| Qwen/Qwen2.5-0.5B-Instruct | `generate_json` | 89/231 | 204.89 | 1366.12 | 1134.9 | N/A | N/A | 35.5% |
| Qwen/Qwen2.5-0.5B-Instruct | `labels` | 97/231 | 26.78 | 160.74 | 2087.2 | 0.678 | 0.249 | 61.9% |
| Qwen/Qwen2.5-0.5B-Instruct | `independent` | 121/231 | 108.43 | 710.51 | 2045.3 | 0.595 | 0.099 | 0.9% |
| Qwen/Qwen2.5-0.5B-Instruct | `sequence_likelihood` | 123/231 | 115.31 | 741.33 | 2131.5 | 0.609 | 0.129 | 33.8% |
| Qwen/Qwen2.5-1.5B-Instruct | `generate_label` | 129/231 | 69.95 | 440.26 | 3307.8 | N/A | N/A | 35.5% |
| Qwen/Qwen2.5-1.5B-Instruct | `generate_json` | 135/231 | 217.18 | 653.52 | 3313.0 | N/A | N/A | 22.9% |
| Qwen/Qwen2.5-1.5B-Instruct | `labels` | 135/231 | 36.02 | 440.73 | 4094.0 | 0.654 | 0.285 | 29.9% |
| Qwen/Qwen2.5-1.5B-Instruct | `independent` | 139/231 | 140.13 | 1864.81 | 4051.5 | 0.561 | 0.200 | 0.4% |
| Qwen/Qwen2.5-1.5B-Instruct | `sequence_likelihood` | 133/231 | 176.27 | 1915.90 | 4133.5 | 0.556 | 0.108 | 21.6% |

Accuracy and timing use canonical order; order changes compare one cyclic rotation. Generation and logits share the label prompt; JSON uses a separate prompt. Brier/ECE use valid distributions; generation has no calibration score. The upstream scorer resolves ties lexicographically; the API selects the first option. See the full report for returned-choice accuracy and methodology.

Test rig: Modal, Tesla T4, torch.float16, batch/concurrency 1, 8192-token context limit, 2 warmups per method and 1 measured attempt(s) per presentation. 231 unique tasks, 462 presentations and 2310 requests per model. CUDA-synchronized timing excludes loading, HTTP and queueing. The model dtype is explicit in the reproduction command.

Full rig configuration and tier breakdown: [docs/jevbench-report.md](docs/jevbench-report.md).
Raw per-task measurements: [jevbench-results-qwen2.5-0.5b.json](docs/jevbench-results-qwen2.5-0.5b.json), [jevbench-results-qwen2.5-1.5b.json](docs/jevbench-results-qwen2.5-1.5b.json).
<!-- jevbench-results:end -->

GGUF comparison: [Qwen2.5-1.5B-Instruct Q5_K_M with llama.cpp](docs/jevbench-comparison-qwen2.5-1.5b.md).

The earlier [30-task comparison](docs/benchmark-comparison.md) uses BF16 and three
repeats per method. Its different dataset and protocol should be evaluated
separately. Reproduce it with `python run_benchmark.py --modal --model MODEL_ID`.

## Development

```bash
pip install -e '.[dev,modal]'
python -m pytest
ruff check .
ruff format --check .
python -m compileall -q src run.py run_benchmark.py run_jevbench.py
```

Code lives in `src/jev_inference/`; tests use fake models and download no weights.
Third-party scorer and dataset attribution: [THIRD-PARTY.md](THIRD-PARTY.md).
