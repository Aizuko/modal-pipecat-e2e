import json
import socket
import subprocess
from typing import Any

import aiohttp
import modal

#MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507-FP8"
#MODEL_REVISION = "953532f942706930ec4bb870569932ef63038fdf"
#MODEL_NAME = "Nanbeige/Nanbeige4.1-3B"
#MODEL_REVISION = "6f3b2c34ac928f8b27849d92a185b9a4af59be63"
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
MODEL_REVISION = "5a5a776300a41aaa681dd7ff0106608ef2bc90db"

N_GPU = 1
VLLM_PORT = 8000
MINUTES = 60

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python='3.12')
        .entrypoint([])
        .uv_pip_install(
            "vllm==0.13.0",
            "huggingface-hub==0.36.0",
        )
        .env({
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_SERVER_DEV_MODE": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        })
)

hf_cache_volume = modal.Volume.from_name('huggingface-cache', create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App('readyformai-llm-step')

with vllm_image.imports():
    import requests

    def _sleep(level=1):
        requests.post(f"http://localhost:{VLLM_PORT}/sleep?level={level}").raise_for_status()

    def _wake_up():
        requests.post(f"http://localhost:{VLLM_PORT}/wake_up").raise_for_status()


def wait_ready(proc: subprocess.Popen):
    while True:
        try:
            socket.create_connection(("localhost", VLLM_PORT), timeout=1).close()
            return
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with {proc.returncode}")


def warmup():
    import requests
    for _ in range(3):
        requests.post(
            f"http://localhost:{VLLM_PORT}/v1/chat/completions",
            json={"model": "llm", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 16},
            timeout=300,
        ).raise_for_status()


@app.cls(
    image=vllm_image,
    gpu=f"L40S:{N_GPU}",
    scaledown_window=30 * MINUTES,
    timeout=30 * MINUTES,
    max_containers=1,
    volumes={
        "/root/.cache/huggingface": hf_cache_volume,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=8)
class VllmServer:
    @modal.enter(snap=True)
    def start(self):
        cmd = [
            "vllm", "serve", "--uvicorn-log-level=info",
            MODEL_NAME,
            "--revision", MODEL_REVISION,
            "--served-model-name", MODEL_NAME, "llm",
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--max-num-batched-tokens", "32768",
            "--max-model-len", "32768",
            "--enforce-eager",
            "--tensor-parallel-size", str(N_GPU),
            "--enable-sleep-mode",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes"
        ]
        print(*cmd)
        self.vllm_proc = subprocess.Popen(cmd)
        wait_ready(self.vllm_proc)
        warmup()
        _sleep()

    @modal.enter(snap=False)
    def restore(self):
        _wake_up()
        wait_ready(self.vllm_proc)

    @modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
    def serve(self):
        pass

    @modal.exit()
    def stop(self):
        self.vllm_proc.terminate()


if __name__ == "__main__":
    import time
    cls = modal.Cls.from_name("readyformai-llm-step", "VllmServer")
    for _ in range(3):
        start = time.time()
        resp = requests.post(
            "https://aizuko--readyformai-llm-step-vllmserver-serve.modal.run/v1/chat/completions",
            json={"model": "llm", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 16},
            timeout=300,
        )
        resp.raise_for_status()
        print(f"Response ({time.time() - start:.2f}s): {resp.json()['choices'][0]['message']['content'][:80]}")
        time.sleep(1)
