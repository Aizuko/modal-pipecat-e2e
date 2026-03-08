import asyncio
import os
import sys
import time
import json
import base64

import modal

app = modal.App("parakeet-stt")

model_cache = modal.Volume.from_name("parakeet-model-cache", create_if_missing=True)

SAMPLE_RATE = 16000
UVICORN_PORT = 8000

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/cache",
        "DEBIAN_FRONTEND": "noninteractive",
        "TORCH_HOME": "/cache",
        "CXX": "g++",
        "CC": "gcc",
    })
    .apt_install("ffmpeg", "g++")
    .uv_pip_install(
        "hf_transfer==0.1.9",
        "huggingface_hub[hf-xet]==0.31.2",
        "nemo_toolkit[asr]==2.3.0",
        "cuda-python==12.8.0",
        "fastapi==0.115.12",
        "numpy<2",
        "torchaudio",
        "soundfile",
        "uvicorn[standard]",
    )
    .entrypoint([])
)

with image.imports():
    import io
    import numpy as np
    import logging
    import nemo.collections.asr as nemo_asr
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from starlette.websockets import WebSocketState
    from urllib.request import urlopen
    import torch
    import threading
    import uvicorn


def _bytes_to_torch(data):
    # Handle WAV format (from pipecat's SegmentedSTTService)
    if data[:4] == b'RIFF':
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        if sr != 16000:
            import torchaudio
            t = torch.from_numpy(audio).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, 16000)
            return t.squeeze(0)
        return torch.from_numpy(audio)
    arr = np.frombuffer(data, dtype=np.int16).astype("float32") / 32768.0
    return torch.from_numpy(arr)


class NoStdStreams:
    def __init__(self):
        self.devnull = open(os.devnull, "w")

    def __enter__(self):
        self._stdout, self._stderr = sys.stdout, sys.stderr
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout, sys.stderr = self.devnull, self.devnull

    def __exit__(self, *_):
        sys.stdout, sys.stderr = self._stdout, self._stderr
        self.devnull.close()


@app.cls(
    volumes={"/cache": model_cache},
    gpu="L40S",
    image=image,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    scaledown_window=30 * 60,
)
@modal.concurrent(max_inputs=20)
class Transcriber:
    @modal.enter(snap=True)
    def load(self):
        self.tunnel_ctx = None
        self.tunnel = None
        self.websocket_url = None

        logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)

        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v3"
        )
        self.model.to(torch.bfloat16)
        self.model.eval()

        if self.model.cfg.decoding.strategy != "beam":
            self.model.cfg.decoding.strategy = "greedy_batch"
            self.model.change_decoding_strategy(self.model.cfg.decoding)

        # warm up
        AUDIO_URL = "https://github.com/voxserv/audio_quality_testing_samples/raw/refs/heads/master/mono_44100/156550__acclivity__a-dream-within-a-dream.wav"
        audio_bytes = urlopen(AUDIO_URL).read()
        if audio_bytes.startswith(b"RIFF"):
            audio_bytes = audio_bytes[44:]
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        chunk_size = SAMPLE_RATE
        chunks = [torch.from_numpy(audio_data[i:i+chunk_size]) for i in range(0, len(audio_data), chunk_size)][:5]

        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16), torch.inference_mode(), torch.no_grad():
            for chunk in chunks:
                self.model.transcribe(chunk)
        print("GPU warmed up")

    @modal.enter(snap=False)
    def _start_server(self):
        self.web_app = FastAPI()

        @self.web_app.websocket("/ws")
        async def run_with_websocket(ws: WebSocket):
            audio_queue = asyncio.Queue()
            transcription_queue = asyncio.Queue()

            async def recv_loop(ws, audio_queue):
                while True:
                    msg = await ws.receive_text()
                    try:
                        data = json.loads(msg)
                        if data.get("type") == "audio":
                            audio_bytes = base64.b64decode(data["audio"].encode("utf-8"))
                            await audio_queue.put(audio_bytes)
                    except Exception:
                        continue

            async def inference_loop(audio_queue, transcription_queue):
                while True:
                    audio_data = await audio_queue.get()
                    audio_tensor = _bytes_to_torch(audio_data)
                    transcript = self.transcribe(audio_tensor)
                    await transcription_queue.put(transcript)

            async def send_loop(transcription_queue, ws):
                while True:
                    transcript = await transcription_queue.get()
                    await ws.send_text(transcript)

            await ws.accept()
            try:
                tasks = [
                    asyncio.create_task(recv_loop(ws, audio_queue)),
                    asyncio.create_task(inference_loop(audio_queue, transcription_queue)),
                    asyncio.create_task(send_loop(transcription_queue, ws)),
                ]
                await asyncio.gather(*tasks)
            except WebSocketDisconnect:
                pass
            except Exception as e:
                print(f"WS error: {e}")
            finally:
                if ws and ws.application_state is WebSocketState.CONNECTED:
                    try:
                        await ws.close(code=1011)
                    except RuntimeError:
                        pass
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

    def transcribe(self, audio_data) -> str:
        with NoStdStreams():
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16), torch.inference_mode(), torch.no_grad():
                output = self.model.transcribe([audio_data])
                return output[0].text

    @modal.asgi_app()
    def webapp(self):
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
    stt = modal.Cls.from_name("parakeet-stt", "Transcriber")
    for _ in range(5):
        start = time.time()
        stt().ping.remote()
        print(f"Ping: {time.time() - start:.3f}s")
        time.sleep(10.0)
