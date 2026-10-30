# JevBench backend comparison

| Field | Reference | Candidate |
| --- | --- | --- |
| Model | Qwen/Qwen2.5-1.5B-Instruct | qwen2.5-1.5b-instruct-q5_k_m.gguf |
| Backend | transformers | llama_cpp |
| Quantization | none | Q5_K_M |
| Dataset SHA-256 | dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51 | dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51 |
| GPU | NVIDIA GeForce RTX 3060 | NVIDIA GeForce RTX 3060 |
| CPU | Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz | Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz |

> Both runs used the same GPU and host (NVIDIA GeForce RTX 3060, Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz). Latency deltas therefore reflect the runtime and precision choices (reference: torch.float16; candidate: Q5_K_M), not different hardware.

## `generate_label`

| Metric | Reference | Candidate | Delta |
| --- | --- | --- | --- |
| Accuracy | 55.84 | 54.98 | -0.87 pp |
| Brier score | N/A | N/A | N/A |
| Top-label ECE | N/A | N/A | N/A |
| Order change rate | 35.06 | 35.06 | +0.00 pp |
| Latency median (ms) | 208.51 | 157.98 | -50.53 ms |
| Latency p95 (ms) | 574.97 | 715.68 | +140.71 ms |
| Scorable | 231 | 231 |
| Correct | 129 | 127 |
| Invalid outputs | 10 | 11 |
| Operational failures | 0 | 0 |

## `generate_json`

| Metric | Reference | Candidate | Delta |
| --- | --- | --- | --- |
| Accuracy | 58.44 | 55.41 | -3.03 pp |
| Brier score | N/A | N/A | N/A |
| Top-label ECE | N/A | N/A | N/A |
| Order change rate | 23.38 | 25.54 | +2.16 pp |
| Latency median (ms) | 735.22 | 228.28 | -506.94 ms |
| Latency p95 (ms) | 1432.90 | 805.09 | -627.81 ms |
| Scorable | 231 | 231 |
| Correct | 135 | 128 |
| Invalid outputs | 0 | 0 |
| Operational failures | 0 | 0 |

## `labels`

| Metric | Reference | Candidate | Delta |
| --- | --- | --- | --- |
| Accuracy | 58.44 | 57.58 | -0.87 pp |
| Brier score | 0.6536 | 0.6904 | +0.0368 |
| Top-label ECE | 0.2845 | 0.2975 | +0.0130 |
| Order change rate | 29.44 | 31.17 | +1.73 pp |
| Latency median (ms) | 106.72 | 124.36 | +17.64 ms |
| Latency p95 (ms) | 444.89 | 636.84 | +191.95 ms |
| Scorable | 231 | 231 |
| Correct | 135 | 133 |
| Invalid outputs | 0 | 0 |
| Operational failures | 0 | 0 |

## `independent`

| Metric | Reference | Candidate | Delta |
| --- | --- | --- | --- |
| Accuracy | 60.61 | 59.31 | -1.30 pp |
| Brier score | 0.5613 | 0.5342 | -0.0272 |
| Top-label ECE | 0.2006 | 0.1940 | -0.0066 |
| Order change rate | 0.43 | 0.00 | -0.43 pp |
| Latency median (ms) | 415.59 | 474.11 | +58.53 ms |
| Latency p95 (ms) | 1883.28 | 2761.04 | +877.76 ms |
| Scorable | 231 | 231 |
| Correct | 140 | 137 |
| Invalid outputs | 0 | 0 |
| Operational failures | 0 | 0 |

## `sequence_likelihood`

| Metric | Reference | Candidate | Delta |
| --- | --- | --- | --- |
| Accuracy | 57.58 | 56.28 | -1.30 pp |
| Brier score | 0.5556 | 0.5518 | -0.0037 |
| Top-label ECE | 0.1057 | 0.0984 | -0.0074 |
| Order change rate | 21.65 | 18.18 | -3.46 pp |
| Latency median (ms) | 425.77 | 902.30 | +476.53 ms |
| Latency p95 (ms) | 1956.76 | 3347.22 | +1390.46 ms |
| Scorable | 231 | 231 |
| Correct | 133 | 130 |
| Invalid outputs | 0 | 0 |
| Operational failures | 0 | 0 |
