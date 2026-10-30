"""Long-running public benchmark; separate from the persistent HTTP API."""

import hashlib
import json
import os
from pathlib import Path

import modal

from .modal_app import backend, backend_options, cache, image, model_id, secret_name

app = modal.App("jev-inference-public-benchmark")


@app.function(
    image=image,
    gpu=os.environ.get("JEV_GPU", "T4"),
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_name(secret_name)] if secret_name else [],
    timeout=7200,
)
def evaluate(
    records: list[dict],
    source: dict,
    model: str,
    repeats: int,
    permutations: int,
    max_input_tokens: int,
    dtype: str,
):
    from jev_inference.engine import DecisionEngine
    from jev_inference.jevbench_runner import run_jevbench

    engine = DecisionEngine(
        model, "cuda", max_input_tokens, dtype, backend=backend, **backend_options()
    )
    cache.commit()
    identity = hashlib.sha256(
        json.dumps([model, source, repeats, permutations, max_input_tokens, dtype]).encode()
    ).hexdigest()[:16]
    path = Path(f"/cache/benchmarks/jevbench-{identity}-checkpoint.json")
    path.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint(rows):
        path.write_text(json.dumps({"model": model, "completed_rows": len(rows), "results": rows}))
        cache.commit()

    result = run_jevbench(engine, records, source, repeats, permutations, checkpoint)
    checkpoint(result["results"])
    return result


@app.local_entrypoint()
def main(
    payload: str,
    output: str = "docs/jevbench-results.json",
    repeats: int = 1,
    permutations: int = 1,
    max_input_tokens: int = 8192,
    dtype: str = "auto",
):
    from .benchmark import save_results

    data = json.loads(Path(payload).read_text())
    result = evaluate.remote(
        data["records"], data["source"], model_id, repeats, permutations, max_input_tokens, dtype
    )
    if result["metadata"]["model"] != model_id:
        raise ValueError("remote model identity mismatch")
    save_results(result, output)
