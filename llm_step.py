import json
from typing import Any
import aiohttp
import modal

#MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507-FP8"
#MODEL_REVISION = "953532f942706930ec4bb870569932ef63038fdf"
#MODEL_NAME = "Nanbeige/Nanbeige4.1-3B"
#MODEL_REVISION = "6f3b2c34ac928f8b27849d92a185b9a4af59be63"

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
MODEL_REVISION = "5a5a776300a41aaa681dd7ff0106608ef2bc90db"

FAST_BOOT = True
N_GPU = 1
VLLM_PORT = 8000

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python='3.12')
        .entrypoint([])
        .uv_pip_install(
            "vllm==0.13.0",
            "huggingface-hub==0.36.0",
        )
        .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster model transfers
)

hf_cache_volume = modal.Volume.from_name('huggingface-cache', create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


app = modal.App('readyformai-llm-step')

@app.function(
    image=vllm_image,
    gpu=f"L40S:{N_GPU}",
    scaledown_window=30 * 60,
    timeout=30 * 60,
    max_containers=1,
    volumes={
        "/root/.cache/huggingface": hf_cache_volume,
        "/root/.cache/vllm": vllm_cache_vol,
    }
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=VLLM_PORT, startup_timeout=10*60)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "llm",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--max-num-batched-tokens",
        "32768",
        "--max-model-len",
        "32768",
    ]
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
    cmd += ["--tensor-parallel-size", str(N_GPU)]
    print(*cmd)

    subprocess.Popen(" ".join(cmd), shell=True)
