"""Modal application definition, configured by the launcher environment."""

import os

import modal

model_id = os.environ.get("JEV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
model_dtype = os.environ.get("JEV_DTYPE", "auto")
max_tokens = int(os.environ.get("JEV_MAX_INPUT_TOKENS", "4096"))
backend = os.environ.get("JEV_BACKEND", "transformers")
n_gpu_layers = int(os.environ.get("JEV_N_GPU_LAYERS", "0"))
n_threads = os.environ.get("JEV_N_THREADS") or None
n_batch = int(os.environ.get("JEV_N_BATCH", "512"))
chat_template = os.environ.get("JEV_CHAT_TEMPLATE") or None
secret_name = os.environ.get("JEV_MODAL_SECRET")
app = modal.App(os.environ.get("JEV_APP_NAME", "jev-inference"))
packages = [
    "torch>=2.4,<3",
    "transformers>=4.51,<5",
    "accelerate>=1,<2",
    "fastapi>=0.115,<1",
    "pydantic>=2.8,<3",
]
if backend == "llama_cpp":
    packages.append("llama-cpp-python>=0.3,<0.4")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*packages)
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "JEV_MODEL": model_id,
            "JEV_DTYPE": model_dtype,
            "JEV_MAX_INPUT_TOKENS": str(max_tokens),
            "JEV_BACKEND": backend,
            "JEV_N_GPU_LAYERS": str(n_gpu_layers),
            "JEV_N_THREADS": n_threads or "",
            "JEV_N_BATCH": str(n_batch),
            "JEV_CHAT_TEMPLATE": chat_template or "",
        }
    )
    .add_local_python_source("jev_inference")
)


def backend_options():
    if backend != "llama_cpp":
        return {}
    return {
        "n_gpu_layers": n_gpu_layers,
        "n_threads": int(n_threads) if n_threads else None,
        "n_batch": n_batch,
        "chat_template": chat_template,
    }


cache = modal.Volume.from_name("jev-inference-model-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=os.environ.get("JEV_GPU", "T4"),
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_name(secret_name)] if secret_name else [],
    timeout=900,
    scaledown_window=300,
    max_containers=1,
)
@modal.asgi_app()
def api():
    from jev_inference.api import create_app
    from jev_inference.engine import DecisionEngine

    engine = DecisionEngine(
        model_id,
        max_input_tokens=max_tokens,
        dtype=model_dtype,
        backend=backend,
        **backend_options(),
    )
    cache.commit()
    return create_app(engine)
