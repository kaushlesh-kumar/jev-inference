import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Serve choices from causal LM logits")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto"
    )
    parser.add_argument("--modal", action="store_true", help="Deploy a persistent Modal API")
    parser.add_argument("--gpu", default="T4", help="Modal GPU type")
    parser.add_argument("--app-name", default="jev-inference")
    parser.add_argument("--secret", help="Modal Secret with JEV_API_KEY and/or HF_TOKEN")
    parser.add_argument("--device", default="auto", help="Local device: auto, cpu, cuda, mps")
    parser.add_argument("--backend", choices=["transformers", "llama_cpp"], default="transformers")
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="GGUF layers to offload")
    parser.add_argument("--n-threads", type=int, help="llama.cpp CPU threads")
    parser.add_argument("--n-batch", type=int, default=512, help="llama.cpp prompt batch size")
    parser.add_argument("--chat-template", help="Override the GGUF chat template")
    parser.add_argument("--chat-template-file", help="Read a chat template override from a file")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.max_input_tokens < 1:
        parser.error("--max-input-tokens must be positive")
    if args.modal:
        environment = os.environ.copy()
        environment.update(
            JEV_MODEL=args.model,
            JEV_DTYPE=args.dtype,
            JEV_GPU=args.gpu,
            JEV_APP_NAME=args.app_name,
            JEV_MAX_INPUT_TOKENS=str(args.max_input_tokens),
            JEV_MODAL_SECRET=args.secret or "",
            JEV_BACKEND=args.backend,
            JEV_N_GPU_LAYERS=str(args.n_gpu_layers),
            JEV_N_THREADS="" if args.n_threads is None else str(args.n_threads),
            JEV_N_BATCH=str(args.n_batch),
            JEV_CHAT_TEMPLATE=chat_template(args) or "",
        )
        result = subprocess.run(
            [sys.executable, "-m", "modal", "deploy", "-m", "jev_inference.modal_app"],
            env=environment,
            check=False,
        )
        raise SystemExit(result.returncode)
    import uvicorn

    from .api import create_app
    from .engine import DecisionEngine

    options = backend_options(args)
    engine = DecisionEngine(
        args.model, args.device, args.max_input_tokens, args.dtype, backend=args.backend, **options
    )
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


def chat_template(args):
    if args.chat_template_file:
        return Path(args.chat_template_file).read_text()
    return args.chat_template


def backend_options(args):
    if args.backend != "llama_cpp":
        return {}
    return {
        "n_gpu_layers": args.n_gpu_layers,
        "n_threads": args.n_threads,
        "n_batch": args.n_batch,
        "chat_template": chat_template(args),
    }
