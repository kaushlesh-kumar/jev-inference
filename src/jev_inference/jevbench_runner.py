"""Public-subset diagnostics using JevBench's pinned core scorer and metrics."""

import hashlib
import os
import platform
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from ._vendor.jevbench.metrics import brier_score, ece_top_label, ordinal_mae, percentile
from ._vendor.jevbench.scoring import score_label, score_task
from ._vendor.jevbench.tasks import Task, dataset_hash
from .benchmark import METHODS as ORIGINAL_METHODS
from .benchmark import generate_choice, max_optional, run_choice
from .jevbench_data import case_for, orders_for

METHODS = (*ORIGINAL_METHODS, "sequence_likelihood")


def decide_task(engine, case: dict, method: str) -> dict:
    if method == "generate_json":
        json_case = {
            **case,
            "context": case["context"] + "\n\nRubric:\n" + "\n".join(case["choices"]),
            "choices": case["canonical_labels"],
        }
        return generate_choice(engine, json_case, method)
    result = run_choice(engine, case, method)
    if result["choice"] is not None:
        result["choice"] = case["canonical_labels"][case["choices"].index(result["choice"])]
    if "probabilities" in result:
        result["probs"] = dict(zip(case["canonical_labels"], result.pop("probabilities")))
    return result


def measure_task(engine, case: dict, method: str) -> dict:
    from .benchmark import measure_operation

    return measure_operation(engine, lambda: decide_task(engine, case, method), capture_errors=True)


def scorable(task: Task) -> bool:
    return task.expected is not None and not task.provenance.get("exclude_reason")


def outcome_for(result: dict, task: Task) -> dict:
    if result.get("error"):
        return {"valid": False, "correct": False, "predicted": None, "probs": None}
    if "probs" in result:
        return score_task(result["probs"], task)
    return score_label(result.get("choice"), task)


def aggregate(rows: list[dict], tasks: dict[str, Task]) -> dict:
    judged = [row for row in rows if scorable(tasks[row["id"]])]
    probability_rows = [row for row in judged if row["outcome"].get("probs")]
    pairs = [
        (max(row["outcome"]["probs"].values()), row["outcome"]["correct"])
        for row in probability_rows
    ]
    ordinal_pairs = []
    noul = []
    fidelity = []
    for row in judged:
        task = tasks[row["id"]]
        outcome = row["outcome"]
        probs = outcome.get("probs")
        if task.question["type"] == "score" and outcome["valid"]:
            predicted = outcome.get("ordinal_ev", float(outcome["predicted"]))
            ordinal_pairs.append((task.expected, predicted))
        if task.question["type"] == "noul":
            predicted = outcome["predicted"]
            if probs:
                yes = probs["yes"]
                predicted = "yes" if yes >= 0.8 else "no" if yes <= 0.2 else None
            noul.append((predicted, predicted == task.expected))
        gold_probs = task.provenance.get("gold_probs")
        if probs and gold_probs:
            fidelity.append(sum(abs(probs[label] - gold_probs[label]) for label in task.labels) / 2)
    times = [row["latency_ms_median"] for row in rows]
    return {
        "attempted": len(rows),
        "scorable": len(judged),
        "correct": sum(row["outcome"]["correct"] for row in judged),
        "accuracy": sum(row["outcome"]["correct"] for row in judged) / len(judged)
        if judged
        else None,
        "invalid_outputs": sum(not row["outcome"]["valid"] for row in rows),
        "operational_failures": sum(bool(row.get("error")) for row in rows),
        "token_limit_cases": sum(
            any(run.get("hit_token_limit", False) for run in row["runs"]) for row in rows
        ),
        "latency_ms_median": percentile(times, 0.5),
        "latency_ms_p95": percentile(times, 0.95),
        "max_peak_allocated_mib": max_optional([row["peak_allocated_mib"] for row in rows]),
        "max_extra_peak_mib": max_optional([row["extra_peak_mib"] for row in rows]),
        "calibration_n": len(probability_rows),
        "brier_mean": (
            sum(
                brier_score(
                    row["outcome"]["probs"], str(tasks[row["id"]].expected), tasks[row["id"]].labels
                )
                for row in probability_rows
            )
            / len(probability_rows)
        )
        if probability_rows
        else None,
        "ece": ece_top_label(pairs) if pairs else None,
        "ordinal_ev_mae": ordinal_mae(ordinal_pairs),
        "ordinal_mae_n": len(ordinal_pairs),
        "noul_threshold_accuracy": sum(correct for _, correct in noul) / len(noul)
        if noul
        else None,
        "noul_abstentions": sum(predicted is None for predicted, _ in noul),
        "noul_n": len(noul),
        "gold_distribution_tv_mean": sum(fidelity) / len(fidelity) if fidelity else None,
        "gold_distribution_n": len(fidelity),
    }


def order_metrics(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["id"]].append(row)
    comparisons = []
    valid_pairs = []
    distances = []
    for group in grouped.values():
        canonical = next(row for row in group if row["variant"] == 0)
        for rotated in group:
            if rotated["variant"] == 0:
                continue
            changed = canonical["choice"] != rotated["choice"]
            comparisons.append(changed)
            if canonical["outcome"]["valid"] and rotated["outcome"]["valid"]:
                valid_pairs.append(changed)
            original = canonical["outcome"].get("probs")
            permuted = rotated["outcome"].get("probs")
            if original and permuted:
                distances.append(
                    sum(abs(original[label] - permuted[label]) for label in original) / 2
                )
    return {
        "comparisons": len(comparisons),
        "decision_change_rate": sum(comparisons) / len(comparisons) if comparisons else None,
        "valid_pairs": len(valid_pairs),
        "choice_flip_rate_valid_pairs": sum(valid_pairs) / len(valid_pairs)
        if valid_pairs
        else None,
        "distribution_pairs": len(distances),
        "probability_tv_mean": sum(distances) / len(distances) if distances else None,
        "probability_tv_max": max(distances) if distances else None,
    }


def summarize_jevbench(rows: list[dict], records: list[dict]) -> dict:
    tasks = {record["task"]["id"]: Task.from_dict(record["task"]) for record in records}
    tiers = {record["task"]["id"]: record["tier"] for record in records}
    summary = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        canonical = [row for row in selected if row["variant"] == 0]
        entry = aggregate(canonical, tasks)
        entry["order_sensitivity"] = order_metrics(selected)
        entry["per_tier"] = {
            tier: aggregate([row for row in canonical if tiers[row["id"]] == tier], tasks)
            for tier in sorted(set(tiers.values()))
        }
        entry["per_type"] = {
            kind: aggregate(
                [row for row in canonical if tasks[row["id"]].question["type"] == kind], tasks
            )
            for kind in sorted({task.question["type"] for task in tasks.values()})
        }
        summary[method] = entry
    return summary


def host_config() -> dict:
    config = {
        "platform": platform.platform(),
        "visible_cpu_count": os.cpu_count(),
        "resource_limits_pinned": False,
    }
    cpu = Path("/proc/cpuinfo")
    if cpu.exists():
        config["cpu_model"] = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpu.read_text().splitlines()
                if line.startswith("model name")
            ),
            None,
        )
    memory = Path("/proc/meminfo")
    if memory.exists():
        line = next(
            (line for line in memory.read_text().splitlines() if line.startswith("MemTotal:")), None
        )
        config["host_visible_ram_mib"] = int(line.split()[1]) / 1024 if line else None
    quota = Path("/sys/fs/cgroup/cpu.max")
    config["cpu_cgroup_quota"] = quota.read_text().strip() if quota.exists() else None
    try:
        probe = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,pstate,clocks.sm,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        config["gpu_telemetry_after"] = probe.stdout.strip() if probe.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        config["gpu_telemetry_after"] = None
    config["gpu_telemetry_fields"] = ["driver_version", "pstate", "sm_clock_mhz", "power_limit_w"]
    config["note"] = (
        "Host-visible resources are not container allocations; GPU telemetry "
        "is an end-of-run snapshot, not pinned clocks during measurement."
    )
    return config


def run_jevbench(
    engine,
    records: list[dict],
    source: dict,
    repeats: int = 1,
    permutations: int = 1,
    checkpoint=None,
) -> dict:
    if repeats < 1 or permutations < 0:
        raise ValueError("repeats must be positive and permutations nonnegative")
    if not engine.backend.is_cuda:
        raise ValueError("benchmark requires a model entirely on one CUDA GPU")
    tasks = [Task.from_dict(record["task"]) for record in records]
    warm = case_for(tasks[0], tasks[0].labels)
    for method in METHODS:
        for _ in range(2):
            decide_task(engine, warm, method)
    rows = []
    for index, task in enumerate(tasks):
        for variant, order in enumerate(orders_for(task, permutations)):
            case = case_for(task, order)
            runs = {method: [] for method in METHODS}
            for repeat in range(repeats):
                offset = (index + variant + repeat) % len(METHODS)
                for method in METHODS[offset:] + METHODS[:offset]:
                    result = measure_task(engine, case, method)
                    runs[method].append(result)
            for method in METHODS:
                measured = runs[method]
                first = measured[0]
                row = {
                    "id": task.id,
                    "variant": variant,
                    "label_order": order,
                    "method": method,
                    "choice": first.get("choice"),
                    "outcome": outcome_for(first, task),
                    "latency_ms_median": percentile([run["latency_ms"] for run in measured], 0.5),
                    "peak_allocated_mib": max_optional(
                        [run["peak_allocated_mib"] for run in measured]
                    ),
                    "extra_peak_mib": max_optional([run["extra_peak_mib"] for run in measured]),
                    "runs": measured,
                }
                if first.get("error"):
                    row["error"] = first["error"]
                rows.append(row)
        print(f"[{index + 1}/{len(tasks)}] {task.id}", flush=True)
        if checkpoint and (index + 1) % 10 == 0:
            checkpoint(rows)
    props = engine.device_properties()
    runtime = engine.runtime_metadata()
    root = Path(__file__).parent
    fingerprint = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "engine.py",
            "benchmark.py",
            "jevbench_runner.py",
            "jevbench_data.py",
            "prompts.py",
            "backends/base.py",
            "backends/transformers_backend.py",
            "backends/llamacpp_backend.py",
        )
    }
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": engine.model_id,
        "model_revision": runtime.get("model_revision"),
        "dataset_sha256": dataset_hash(tasks),
        "source": source,
        "source_sha256": fingerprint,
        "unique_tasks": len(tasks),
        "presentations": len(rows) // len(METHODS),
        "requests": len(rows) * repeats,
        "repeats": repeats,
        "permutations": permutations,
        "host": host_config(),
        "gpu_compute_capability": props.get("compute_capability"),
        "gpu": props.get("name"),
        "gpu_total_mib": (
            None if props.get("total_memory") is None else props["total_memory"] / 2**20
        ),
        "dtype": runtime.get("dtype"),
        "backend": runtime.get("backend"),
        "backend_version": runtime.get("backend_version"),
        "quantization": runtime.get("quantization"),
        "runtime": runtime,
        "memory_measurement": (
            "torch_cuda_allocator" if runtime.get("backend") == "transformers" else "unavailable"
        ),
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("torch", "transformers", "accelerate")},
        "cuda": runtime.get("cuda"),
        "max_input_tokens": engine.max_input_tokens,
        "warmup_runs_per_method": 2,
        "batch_size": 1,
        "concurrency": 1,
        "generation_limits": {"generate_label": 16, "generate_json": 64},
        "greedy": True,
        "temperature": 1.0,
        "scorer": "vendored JevBench v1.2 core; argmax label accuracy; ordinal EV MAE; "
        "supplementary v1.5-style Noul thresholds; no official composite",
        "timing": "tokenization + model + processing; GPU synchronized; no HTTP or cold start",
        "calibration": "Brier multiclass sum; top-label ECE, 10 equal-width bins; "
        "only valid distributions on canonical presentations; label-only=N/A",
        "permutation": "one-position cyclic rotation by default; same canonical labels and rubrics",
        "failure_policy": "failed attempts count as incorrect; no truncation or dropped tasks",
    }
    return {
        "metadata": metadata,
        "dataset": records,
        "summary": summarize_jevbench(rows, records),
        "results": rows,
    }
