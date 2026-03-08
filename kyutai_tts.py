import asyncio
import time
import json

import modal

app = modal.App("kyutai-tts")

UVICORN_PORT = 8000
TTS_SAMPLE_RATE = 24000
DEFAULT_VOICE = "expresso/ex03-ex01_happy_001_channel1_334s.wav"

tts_cache = modal.Volume.from_name("kyutai-tts-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install(
        "moshi>=0.2.11",
        "torch",
        "sphn",
        "fastapi[standard]",
        "uvicorn[standard]",
    )
    .env({"HF_HOME": "/cache"})
)

with image.imports():
    import numpy as np
    import torch
    from moshi.models.loaders import CheckpointInfo
    from moshi.models.tts import DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO, TTSModel
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from starlette.websockets import WebSocketState
    import threading
    import uvicorn


@app.cls(
    image=image,
    volumes={"/cache": tts_cache},
    gpu="L40S",
    scaledown_window=30 * 60,
)
@modal.concurrent(max_inputs=10)
class KyutaiTTS:
    @modal.enter()
    def load(self):
        self.tunnel_ctx = None
        self.tunnel = None
        self.websocket_url = None

        print("Loading Kyutai TTS model...")
        checkpoint_info = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
        self.tts_model = TTSModel.from_checkpoint_info(
            checkpoint_info, n_q=32, temp=0.6, device="cuda"
        )
        self.default_voice_path = self.tts_model.get_voice_path(DEFAULT_VOICE)

        # warm up
        print("Warming up TTS...")
        for _ in range(3):
            list(self._stream_tts("Hello, how are you today?"))
        print("TTS warmed up")

    @modal.enter()
    def _start_server(self):
        self.web_app = FastAPI()

        @self.web_app.websocket("/ws")
        async def run_with_websocket(ws: WebSocket):
            prompt_queue = asyncio.Queue()
            audio_queue = asyncio.Queue()

            async def recv_loop(ws, prompt_queue):
                while True:
                    msg = await ws.receive_text()
                    try:
                        data = json.loads(msg)
                        if data.get("type") == "prompt":
                            await prompt_queue.put(data)
                    except Exception:
                        continue

            async def inference_loop(prompt_queue, audio_queue):
                while True:
                    prompt_msg = await prompt_queue.get()
                    text = prompt_msg["text"]
                    voice = prompt_msg.get("voice", DEFAULT_VOICE)
                    for chunk in self._stream_tts(text, voice=voice):
                        await audio_queue.put(chunk)

            async def send_loop(audio_queue, ws):
                while True:
                    audio = await audio_queue.get()
                    await ws.send_bytes(audio)

            await ws.accept()
            try:
                tasks = [
                    asyncio.create_task(recv_loop(ws, prompt_queue)),
                    asyncio.create_task(inference_loop(prompt_queue, audio_queue)),
                    asyncio.create_task(send_loop(audio_queue, ws)),
                ]
                await asyncio.gather(*tasks)
            except WebSocketDisconnect:
                pass
            except Exception as e:
                print(f"WS error: {e}")
            finally:
                if ws and ws.application_state is WebSocketState.CONNECTED:
                    await ws.close(code=1011)
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

    def _stream_tts(self, text, voice=None):
        if voice is None or voice == DEFAULT_VOICE:
            voice_path = self.default_voice_path
        elif voice.endswith(".safetensors"):
            voice_path = voice
        else:
            voice_path = self.tts_model.get_voice_path(voice)

        entries = self.tts_model.prepare_script([text], padding_between=1)
        condition_attributes = self.tts_model.make_condition_attributes(
            [voice_path], cfg_coef=2.0
        )

        frames_collected = []
        all_pcm = []

        def _on_frame(frame):
            if (frame != -1).all():
                frames_collected.append(frame)

        with self.tts_model.mimi.streaming(1), torch.no_grad():
            self.tts_model.generate(
                [entries], [condition_attributes], on_frame=_on_frame
            )

        for frame in frames_collected:
            pcm = self.tts_model.mimi.decode(frame[:, 1:, :]).cpu().detach().numpy()
            pcm = np.clip(pcm[0, 0], -1, 1)
            all_pcm.append(pcm)

        if all_pcm:
            full_pcm = np.concatenate(all_pcm)
            audio_int16 = (full_pcm * 32767).astype(np.int16)
            # Send in ~0.5s chunks (24kHz * 0.5 = 12000 samples)
            chunk_size = 12000
            for i in range(0, len(audio_int16), chunk_size):
                yield audio_int16[i:i + chunk_size].tobytes()

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
