"""Ephemeral benchmark app; does not update the deployed inference API."""

import json
import os
from pathlib import Path

import modal

from .modal_app import backend, backend_options, cache, image, model_id, secret_name

app = modal.App("jev-inference-benchmark")


@app.function(
    image=image,
    gpu=os.environ.get("JEV_GPU", "T4"),
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_name(secret_name)] if secret_name else [],
    timeout=1800,
)
def evaluate(cases: list[dict], repeats: int, model: str, dtype: str = "auto"):
    from jev_inference.benchmark import run_benchmark
    from jev_inference.engine import DecisionEngine

    engine = DecisionEngine(model, device="cuda", dtype=dtype, backend=backend, **backend_options())
    cache.commit()
    return run_benchmark(engine, cases, repeats)


@app.local_entrypoint()
def main(
    dataset: str = "benchmarks/classification30.json",
    output: str = "docs/benchmark-results.json",
    repeats: int = 3,
    dtype: str = "auto",
):
    from .benchmark import save_results, validate_cases

    cases = validate_cases(json.loads(Path(dataset).read_text()))
    save_results(evaluate.remote(cases, repeats, model_id, dtype), output)
