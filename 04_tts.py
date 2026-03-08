import asyncio
import time
import json
from concurrent.futures import ThreadPoolExecutor

import modal

app = modal.App("kyutai-tts")

UVICORN_PORT = 8000
DEFAULT_SPEAKER = "Ryan"

tts_cache = modal.Volume.from_name("kyutai-tts-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(
        "qwen-tts",
        "torch",
        "torchaudio",
        "numpy",
        "fastapi[standard]",
        "uvicorn[standard]",
    )
    .env({"HF_HOME": "/cache"})
)

with image.imports():
    import numpy as np
    import torch
    from qwen_tts import Qwen3TTSModel
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    import threading
    import uvicorn


@app.cls(
    image=image,
    volumes={"/cache": tts_cache},
    gpu="L40S",
    scaledown_window=30 * 60,
    timeout=10 * 60,
)
@modal.concurrent(max_inputs=10)
class KyutaiTTS:
    @modal.enter()
    def load(self):
        self.tunnel_ctx = None
        self.tunnel = None
        self.websocket_url = None
        self._executor = ThreadPoolExecutor(max_workers=1)

        print("Loading Qwen3-TTS model...")
        self.tts_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        tts_cache.commit()

        print("Warming up TTS...")
        self.tts_model.generate_custom_voice(
            text="Hello, how are you today?",
            language="Auto",
            speaker=DEFAULT_SPEAKER,
        )
        print("TTS warmed up")

    @modal.enter()
    def _start_server(self):
        self.web_app = FastAPI()

        @self.web_app.websocket("/ws")
        async def run_with_websocket(ws: WebSocket):
            prompt_queue = asyncio.Queue()
            audio_queue = asyncio.Queue()

            async def recv_loop():
                while True:
                    msg = await ws.receive_text()
                    try:
                        data = json.loads(msg)
                        if data.get("type") == "prompt":
                            await prompt_queue.put(data)
                    except Exception:
                        continue

            async def inference_loop():
                loop = asyncio.get_event_loop()
                while True:
                    prompt_msg = await prompt_queue.get()
                    text = prompt_msg["text"]
                    speaker = prompt_msg.get("speaker", DEFAULT_SPEAKER)
                    print(f"[TTS] Synthesizing: {text[:80]!r}")
                    t0 = time.time()
                    audio_bytes = await loop.run_in_executor(
                        self._executor, self._synthesize, text, speaker
                    )
                    print(f"[TTS] Done in {time.time()-t0:.2f}s")
                    if audio_bytes:
                        await audio_queue.put(audio_bytes)

            async def send_loop():
                while True:
                    audio = await audio_queue.get()
                    await ws.send_bytes(audio)

            await ws.accept()
            tasks = []
            try:
                tasks = [
                    asyncio.create_task(recv_loop()),
                    asyncio.create_task(inference_loop()),
                    asyncio.create_task(send_loop()),
                ]
                await asyncio.gather(*tasks)
            except WebSocketDisconnect:
                pass
            except Exception as e:
                print(f"WS error: {e}")
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

        def start_server():
            uvicorn.run(self.web_app, host="0.0.0.0", port=UVICORN_PORT)

        self.server_thread = threading.Thread(target=start_server, daemon=True)
        self.server_thread.start()

        self.tunnel_ctx = modal.forward(UVICORN_PORT)
        self.tunnel = self.tunnel_ctx.__enter__()
        self.websocket_url = self.tunnel.url.replace("https://", "wss://") + "/ws"
        print(f"Websocket URL: {self.websocket_url}")

    def _synthesize(self, text, speaker=DEFAULT_SPEAKER):
        wavs, sr = self.tts_model.generate_custom_voice(
            text=text,
            language="Auto",
            speaker=speaker,
            max_new_tokens=1024,
        )
        if wavs:
            pcm = np.clip(wavs[0], -1, 1)
            return (pcm * 32767).astype(np.int16).tobytes()
        return None

    @modal.method()
    async def run_tunnel_client(self, d: modal.Dict):
        try:
            await d.put.aio("url", self.websocket_url)
            while not await d.contains.aio("is_running"):
                await asyncio.sleep(1.0)
            while await d.get.aio("is_running"):
                await asyncio.sleep(1.0)
        except Exception as e:
            print(f"Tunnel error: {e}")

    @modal.asgi_app()
    def web_endpoint(self):
        return self.web_app

    @modal.method()
    def ping(self):
        return "pong"

    @modal.exit()
    def exit(self):
        if self.tunnel_ctx:
            self.tunnel_ctx.__exit__(None, None, None)
            self.tunnel_ctx = None


if __name__ == "__main__":
    tts = modal.Cls.from_name("kyutai-tts", "KyutaiTTS")
    for _ in range(5):
        start = time.time()
        tts().ping.remote()
        print(f"Ping: {time.time() - start:.3f}s")
        time.sleep(10.0)
