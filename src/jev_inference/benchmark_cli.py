import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .benchmark import save_results, validate_cases
from .cli import backend_options, chat_template


def main():
    parser = argparse.ArgumentParser(description="Benchmark generation versus logits on a CUDA GPU")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto"
    )
    parser.add_argument("--dataset", default="benchmarks/classification30.json")
    parser.add_argument("--output", default="docs/benchmark-results.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modal", action="store_true")
    parser.add_argument("--gpu", default="T4")
    parser.add_argument("--secret", help="Modal Secret for private HF models")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=["transformers", "llama_cpp"], default="transformers")
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="GGUF layers to offload")
    parser.add_argument("--n-threads", type=int, help="llama.cpp CPU threads")
    parser.add_argument("--n-batch", type=int, default=512, help="llama.cpp prompt batch size")
    parser.add_argument("--chat-template", help="Override the GGUF chat template")
    parser.add_argument("--chat-template-file", help="Read a chat template override from a file")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    cases = validate_cases(json.loads(Path(args.dataset).read_text()))
    if args.modal:
        environment = os.environ.copy()
        environment.update(
            JEV_MODEL=args.model,
            JEV_DTYPE=args.dtype,
            JEV_GPU=args.gpu,
            JEV_MODAL_SECRET=args.secret or "",
            JEV_BACKEND=args.backend,
            JEV_N_GPU_LAYERS=str(args.n_gpu_layers),
            JEV_N_THREADS="" if args.n_threads is None else str(args.n_threads),
            JEV_N_BATCH=str(args.n_batch),
            JEV_CHAT_TEMPLATE=chat_template(args) or "",
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "modal",
                "run",
                "-m",
                "jev_inference.benchmark_modal",
                "--dtype",
                args.dtype,
                "--dataset",
                args.dataset,
                "--output",
                args.output,
                "--repeats",
                str(args.repeats),
            ],
            env=environment,
            check=False,
        )
        raise SystemExit(result.returncode)
    from .benchmark import run_benchmark
    from .engine import DecisionEngine

    save_results(
        run_benchmark(
            DecisionEngine(
                args.model,
                args.device,
                dtype=args.dtype,
                backend=args.backend,
                **backend_options(args),
            ),
            cases,
            args.repeats,
        ),
        args.output,
    )
