import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .benchmark import save_results
from .cli import backend_options, chat_template
from .jevbench_data import load_public


def main():
    parser = argparse.ArgumentParser(
        description="JevBench public-subset and option-order diagnostics"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto"
    )
    parser.add_argument("--modal", action="store_true")
    parser.add_argument("--gpu", default="T4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=["transformers", "llama_cpp"], default="transformers")
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="GGUF layers to offload")
    parser.add_argument("--n-threads", type=int, help="llama.cpp CPU threads")
    parser.add_argument("--n-batch", type=int, default=512, help="llama.cpp prompt batch size")
    parser.add_argument("--chat-template", help="Override the GGUF chat template")
    parser.add_argument("--chat-template-file", help="Read a chat template override from a file")
    parser.add_argument("--secret")
    parser.add_argument("--source-dir", default=".cache/jevbench-public")
    parser.add_argument("--output", default="docs/jevbench-results.json")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--permutations", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument(
        "--limit", type=int, help="Explicit diagnostic subset; not a full public run"
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.permutations < 0 or args.max_input_tokens < 1:
        parser.error("invalid repeat/permutation/context limits")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    records, source = load_public(args.source_dir)
    if args.limit:
        records = records[: args.limit]
        source = {**source, "diagnostic_limit": args.limit}
    if args.modal:
        environment = os.environ.copy()
        environment.update(
            JEV_MODEL=args.model,
            JEV_DTYPE=args.dtype,
            JEV_GPU=args.gpu,
            JEV_MODAL_SECRET=args.secret or "",
            JEV_MAX_INPUT_TOKENS=str(args.max_input_tokens),
            JEV_BACKEND=args.backend,
            JEV_N_GPU_LAYERS=str(args.n_gpu_layers),
            JEV_N_THREADS="" if args.n_threads is None else str(args.n_threads),
            JEV_N_BATCH=str(args.n_batch),
            JEV_CHAT_TEMPLATE=chat_template(args) or "",
        )
        # Only a public, hash-checked payload is staged; no credential material.
        payload = Path(f".cache/jevbench-payload-{os.getpid()}.json")
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_text(json.dumps({"records": records, "source": source}))
        command = [
            sys.executable,
            "-m",
            "modal",
            "run",
            "-m",
            "jev_inference.jevbench_modal",
            "--payload",
            str(payload),
            "--output",
            args.output,
            "--repeats",
            str(args.repeats),
            "--permutations",
            str(args.permutations),
            "--dtype",
            args.dtype,
            "--max-input-tokens",
            str(args.max_input_tokens),
        ]
        result = subprocess.run(command, env=environment, check=False)
        raise SystemExit(result.returncode)
    from .engine import DecisionEngine
    from .jevbench_runner import run_jevbench

    options = backend_options(args)
    engine = DecisionEngine(
        args.model, args.device, args.max_input_tokens, args.dtype, backend=args.backend, **options
    )
    save_results(
        run_jevbench(engine, records, source, args.repeats, args.permutations), args.output
    )
