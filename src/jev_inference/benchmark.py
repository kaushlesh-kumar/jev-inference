"""Warmed, single-GPU classification benchmark; no HTTP latency included."""

import gc
import hashlib
import json
import math
import platform
import re
import statistics
import string
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from .prompts import label_prompt
from .schemas import DecisionRequest

METHODS = ("generate_label", "generate_json", "labels", "independent")


def validate_cases(cases: list[dict]) -> list[dict]:
    if not cases:
        raise ValueError("benchmark dataset must not be empty")
    ids = set()
    for case in cases:
        DecisionRequest(context=case["context"], choices=case["choices"])
        if not isinstance(case["id"], str) or not case["id"] or case["id"] in ids:
            raise ValueError("case IDs must be nonempty and unique")
        if not isinstance(case["category"], str) or not case["category"]:
            raise ValueError("category must be nonempty")
        if case["answer"] not in case["choices"]:
            raise ValueError("expected answer must be an allowed choice")
        ids.add(case["id"])
    return cases


def dataset_hash(cases: list[dict]) -> str:
    canonical = json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(canonical).hexdigest()


def parse_generation(text: str, choices: list[str], method: str) -> str | None:
    if method == "generate_label":
        match = re.fullmatch(r"([A-Z])[.)]?", text.strip())
        if match and (index := string.ascii_uppercase.index(match[1])) < len(choices):
            return choices[index]
        return None
    if method != "generate_json":
        raise ValueError(f"unknown generation method: {method}")
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(result, dict) and result.get("choice") in choices:
        return result["choice"]
    return None


def json_prompt(context: str, choices: list[str]) -> str:
    return (
        "Choose the best option for the context below. Reply only with a JSON object "
        'of the form {"choice": "exact option text"}, without explanation or markdown.\n\n'
        f"Context:\n{context}\n\nAllowed choices:\n"
        f"{json.dumps(choices, ensure_ascii=False)}\n\nJSON:"
    )


def generate_choice(engine, case: dict, method: str) -> dict:
    prompt = (
        label_prompt(case["context"], case["choices"])
        if method == "generate_label"
        else json_prompt(case["context"], case["choices"])
    )
    limit = 16 if method == "generate_label" else 64
    input_tokens = len(
        engine.backend.encode(engine.backend.format_prompt(prompt), add_special_tokens=False)
    )
    text, generated_tokens, hit_token_limit = engine.generate(prompt, limit)
    return {
        "choice": parse_generation(text, case["choices"], method),
        "raw_output": text,
        "generated_tokens": generated_tokens,
        "hit_token_limit": hit_token_limit,
        "input_tokens": input_tokens,
    }


def run_choice(engine, case: dict, method: str) -> dict:
    if method.startswith("generate_"):
        return generate_choice(engine, case, method)
    result = engine.decide(
        DecisionRequest(context=case["context"], choices=case["choices"], mode=method)
    )
    return {
        "choice": result.choice,
        "probabilities": [item.probability for item in result.choices],
        "generated_tokens": 0,
        "hit_token_limit": False,
        "forward_passes": result.forward_passes,
    }


def measure(engine, case: dict, method: str) -> dict:
    return measure_operation(engine, lambda: run_choice(engine, case, method))


def _mib(value: float | None) -> float | None:
    return None if value is None else value / 2**20


def _torch(engine):
    return getattr(getattr(engine, "backend", None), "torch", None) or getattr(
        engine, "torch", None
    )


def _oom_errors(engine) -> tuple[type[BaseException], ...]:
    errors: tuple[type[BaseException], ...] = (ValueError, MemoryError)
    torch = _torch(engine)
    if torch is not None:
        errors += (torch.OutOfMemoryError,)
    return errors


def _synchronize(engine) -> None:
    method = getattr(engine, "synchronize", None)
    if method is not None:
        method()
        return
    engine.torch.cuda.synchronize(engine.model.device)


def _memory(engine, name: str) -> float | None:
    method = getattr(engine, name, None)
    if method is not None:
        return method()
    return getattr(engine.torch.cuda, name)(engine.model.device)


def _reset_peak(engine) -> None:
    method = getattr(engine, "reset_peak_memory", None)
    if method is not None:
        method()
        return
    engine.torch.cuda.reset_peak_memory_stats(engine.model.device)


def _is_cuda(engine) -> bool:
    backend = getattr(engine, "backend", None)
    if backend is not None:
        return bool(backend.is_cuda)
    device = engine.model.device
    return device.type == "cuda" and {
        parameter.device for parameter in engine.model.parameters()
    } == {device}


def _device_properties(engine) -> dict:
    method = getattr(engine, "device_properties", None)
    if method is not None:
        return method()
    properties = engine.torch.cuda.get_device_properties(engine.model.device)
    return {
        "name": properties.name,
        "total_memory": properties.total_memory,
        "compute_capability": [properties.major, properties.minor],
    }


def measure_operation(engine, operation, capture_errors: bool = False) -> dict:
    gc.collect()
    _synchronize(engine)
    baseline = _memory(engine, "memory_allocated")
    _reset_peak(engine)
    started = perf_counter()
    try:
        result = operation()
    except _oom_errors(engine) as error:
        if not capture_errors:
            raise
        result = {"choice": None, "error": f"{type(error).__name__}: {error}"}
    _synchronize(engine)
    elapsed_ms = (perf_counter() - started) * 1000
    peak = _memory(engine, "max_memory_allocated")
    extra = None if peak is None or baseline is None else peak - baseline
    return {
        **result,
        "latency_ms": elapsed_ms,
        "baseline_allocated_mib": _mib(baseline),
        "peak_allocated_mib": _mib(peak),
        "extra_peak_mib": _mib(extra),
        "peak_reserved_mib": _mib(_memory(engine, "max_memory_reserved")),
    }


def max_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        times = sorted(row["latency_ms_median"] for row in selected)
        count = len(selected)
        summary[method] = {
            "correct": sum(row["correct"] for row in selected),
            "total": count,
            "accuracy": sum(row["correct"] for row in selected) / count,
            "invalid_outputs": sum(row["choice"] is None for row in selected),
            "unstable_cases": sum(not row["stable_choice"] for row in selected),
            "token_limit_cases": sum(
                any(run["hit_token_limit"] for run in row["runs"]) for row in selected
            ),
            "latency_ms_median": statistics.median(times),
            "latency_ms_p95": times[math.ceil(0.95 * count) - 1],
            "max_peak_allocated_mib": max_optional([row["peak_allocated_mib"] for row in selected]),
            "max_extra_peak_mib": max_optional([row["extra_peak_mib"] for row in selected]),
        }
    return summary


def run_benchmark(engine, cases: list[dict], repeats: int = 3) -> dict:
    cases = validate_cases(cases)
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not _is_cuda(engine):
        raise ValueError("benchmark requires a model entirely on one CUDA GPU")
    for method in METHODS:
        for _ in range(2):
            run_choice(engine, cases[0], method)
    _synchronize(engine)
    rows = []
    for case_index, case in enumerate(cases):
        runs = {method: [] for method in METHODS}
        for repeat in range(repeats):
            offset = (case_index + repeat) % len(METHODS)
            order = METHODS[offset:] + METHODS[:offset]
            for method in order:
                runs[method].append(measure(engine, case, method))
        for method in METHODS:
            measurements = runs[method]
            choice = measurements[0]["choice"]
            rows.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "method": method,
                    "expected": case["answer"],
                    "choice": choice,
                    "correct": choice == case["answer"],
                    "stable_choice": all(run["choice"] == choice for run in measurements),
                    "latency_ms_median": statistics.median(
                        run["latency_ms"] for run in measurements
                    ),
                    "peak_allocated_mib": max_optional(
                        [run["peak_allocated_mib"] for run in measurements]
                    ),
                    "extra_peak_mib": max_optional([run["extra_peak_mib"] for run in measurements]),
                    "runs": measurements,
                }
            )
        print(f"[{case_index + 1}/{len(cases)}] {case['id']}", flush=True)
    properties = _device_properties(engine)
    runtime_method = getattr(engine, "runtime_metadata", None)
    runtime = runtime_method() if runtime_method is not None else {}
    total_memory = properties.get("total_memory")
    return {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": engine.model_id,
            "model_revision": runtime.get("model_revision"),
            "dataset_sha256": dataset_hash(cases),
            "cases": len(cases),
            "repeats": repeats,
            "warmup_runs_per_method": 2,
            "gpu": properties.get("name"),
            "gpu_total_mib": None if total_memory is None else total_memory / 2**20,
            "dtype": runtime.get("dtype"),
            "python": platform.python_version(),
            "versions": {name: version(name) for name in ("torch", "transformers", "accelerate")},
            "cuda": runtime.get("cuda"),
            "max_input_tokens": engine.max_input_tokens,
            "backend": runtime.get("backend"),
            "backend_version": runtime.get("backend_version"),
            "quantization": runtime.get("quantization"),
            "runtime": runtime,
            "memory_measurement": (
                "torch_cuda_allocator"
                if runtime.get("backend") == "transformers"
                else "unavailable"
            ),
            "timing": "tokenization + model + decode/normalization; GPU synchronized; no HTTP",
            "memory": "PyTorch CUDA allocator only; weights included; MiB = 2^20 bytes",
            "generation_limits": {"generate_label": 16, "generate_json": 64},
            "method_order": "rotated by case and repeat",
            "greedy": True,
            "accuracy_rule": "first repetition per case; invalid output counts as incorrect",
        },
        "dataset": cases,
        "summary": summarize(rows),
        "results": rows,
    }


def save_results(result: dict, output: str):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"Saved {path}", flush=True)
