import asyncio
import json
import sys
import time
import uuid
import base64
from typing import AsyncGenerator, Optional

import modal

APP_NAME = "voice-agent"
app = modal.App(APP_NAME)

MINUTES = 60
VLLM_BASE_URL = "https://aizuko--readyformai-llm-step-vllmserver-serve.modal.run/v1"

bot_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .uv_pip_install(
        "pipecat-ai[webrtc,openai,silero,noisereduce,soundfile]==0.0.92",
        "websockets",
        "aiofiles",
        "fastapi[standard]",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# ---------------------------------------------------------------------------
# Service bridge classes
# ---------------------------------------------------------------------------

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame, Frame, StartFrame, EndFrame, CancelFrame,
    TranscriptionFrame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame,
    LLMRunFrame, TextFrame,
)
from pipecat.services.websocket_service import WebsocketService
from pipecat.services.tts_service import TTSService
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.transports.smallwebrtc.connection import (
    IceServer, SmallWebRTCConnection,
)
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams


class ModalTunnelManager:
    def __init__(self, app_name: str, cls_name: str):
        self._app_name = app_name
        self._cls_name = cls_name
        self._modal_dict_id = str(uuid.uuid4())
        self._url_dict = modal.Dict.from_name(
            f"{self._modal_dict_id}-url-dict", create_if_missing=True
        )
        self._cls = modal.Cls.from_name(app_name, cls_name)()
        self.function_call = None

    async def start(self):
        await self._url_dict.put.aio("is_running", True)
        self.function_call = await self._cls.run_tunnel_client.spawn.aio(self._url_dict)

    async def get_url(self):
        while not await self._url_dict.contains.aio("url"):
            await asyncio.sleep(0.1)
        return await self._url_dict.get.aio("url")

    async def close(self):
        try:
            await self._url_dict.put.aio("is_running", False)
            await modal.Dict.objects.delete.aio(f"{self._modal_dict_id}-url-dict")
        except Exception as e:
            logger.error(f"Tunnel cleanup error: {e}")
        if self.function_call:
            try:
                self.function_call.cancel()
            except Exception:
                pass
            self.function_call = None


class ModalWebsocketService(WebsocketService):
    def __init__(self, modal_tunnel_manager: ModalTunnelManager, **kwargs):
        super().__init__(reconnect_on_error=True, **kwargs)
        self.modal_tunnel_manager = modal_tunnel_manager
        self._websocket_url = None
        self._websocket = None
        self._receive_task = None

    async def _report_error(self, error: ErrorFrame):
        await self.push_error(error)

    async def _connect(self):
        retries = 240
        while self._websocket_url is None and retries > 0:
            retries -= 1
            self._websocket_url = await self.modal_tunnel_manager.get_url()
            await asyncio.sleep(0.1)
        if self._websocket_url is None:
            raise Exception("Failed to get websocket URL")
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(
                self._receive_task_handler(self._report_error)
            )

    async def _disconnect(self):
        try:
            if self._receive_task:
                await self.cancel_task(self._receive_task, timeout=2.0)
                self._receive_task = None
            await self._disconnect_websocket()
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
        finally:
            if self.modal_tunnel_manager:
                await self.modal_tunnel_manager.close()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            self._websocket = await websocket_connect(self._websocket_url)
        except Exception as e:
            logger.error(f"WS connect error: {e}")
            self._websocket = None

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                await self._websocket.close()
        except Exception as e:
            logger.error(f"WS close error: {e}")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")


class ModalParakeetSTTService(SegmentedSTTService, ModalWebsocketService):
    def __init__(self, **kwargs):
        SegmentedSTTService.__init__(self, **kwargs)
        ModalWebsocketService.__init__(self, **kwargs)

    def can_generate_metrics(self):
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect_only()
        await self._websocket.send(json.dumps({"type": "set_vad", "vad": False}))

    async def _connect_only(self):
        """Connect without starting the background receive task."""
        retries = 240
        while self._websocket_url is None and retries > 0:
            retries -= 1
            self._websocket_url = await self.modal_tunnel_manager.get_url()
            await asyncio.sleep(0.1)
        if self._websocket_url is None:
            raise Exception("Failed to get websocket URL")
        await self._connect_websocket()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def _receive_messages(self):
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not self._websocket:
            yield ErrorFrame("Not connected to Parakeet.", fatal=True)
            return
        await self.start_ttfb_metrics()
        try:
            msg = {
                "type": "audio",
                "audio": base64.b64encode(audio).decode("utf-8"),
            }
            await self._websocket.send(json.dumps(msg))
            response = await self._websocket.recv()
            if isinstance(response, str) and response.strip():
                await self.stop_ttfb_metrics()
                await self.stop_processing_metrics()
                yield TranscriptionFrame(response, "", time_now_iso8601())
        except Exception as e:
            yield ErrorFrame(f"STT error: {e}")

    @traced_stt
    async def _handle_transcription(self, transcript, is_final, language=None):
        pass


class ModalKyutaiTTSService(TTSService, ModalWebsocketService):
    def __init__(self, voice: str = None, **kwargs):
        TTSService.__init__(
            self,
            pause_frame_processing=True,
            push_stop_frames=True,
            push_text_frames=True,
            stop_frame_timeout_s=1.0,
            **kwargs,
        )
        ModalWebsocketService.__init__(self, **kwargs)
        self._voice = voice
        self._running = False

    def can_generate_metrics(self):
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def _receive_messages(self):
        async for message in self._get_websocket():
            try:
                await self.stop_ttfb_metrics()
                await self.push_frame(TTSAudioRawFrame(message, 24000, 1))
            except Exception as e:
                logger.error(f"TTS receive error: {e}")
                await self.push_error(ErrorFrame(f"TTS error: {e}"))

    async def run_tts(self, prompt: str) -> AsyncGenerator[Frame, None]:
        if not self._websocket:
            yield ErrorFrame("Not connected to TTS.", fatal=True)
            return
        try:
            if not self._running:
                await self.start_ttfb_metrics()
                yield TTSStartedFrame()
                self._running = True
            msg = {
                "type": "prompt",
                "text": prompt.strip(),
            }
            if self._voice:
                msg["voice"] = self._voice
            await self._websocket.send(json.dumps(msg))
            await self.push_frame(TextFrame(text=prompt))
        except Exception as e:
            yield ErrorFrame(f"TTS send error: {e}")
            return
        yield None


# ---------------------------------------------------------------------------
# Bot pipeline
# ---------------------------------------------------------------------------

log_dict = modal.Dict.from_name("voice-agent-logs", create_if_missing=True)


async def run_bot(webrtc_connection: SmallWebRTCConnection):
    stt_tunnel = ModalTunnelManager(app_name="parakeet-stt", cls_name="Transcriber")
    tts_tunnel = ModalTunnelManager(app_name="kyutai-tts", cls_name="KyutaiTTS")
    await stt_tunnel.start()
    await tts_tunnel.start()

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.8)),
        ),
    )

    stt = ModalParakeetSTTService(modal_tunnel_manager=stt_tunnel)

    llm = OpenAILLMService(
        api_key="not-needed",
        base_url=VLLM_BASE_URL,
        model="llm",
    )

    tts = ModalKyutaiTTSService(modal_tunnel_manager=tts_tunnel, sample_rate=24000)

    messages = [
        {
            "role": "system",
            "content": (
                """\
You are a Canadian voice assistant. As with all things Canadian, all your replies must work in be in both french and english, no matter if the user is speaking in french or english. Always reply first in English, then repeat the exact same reply in French.
Keep your responses concise and conversational — ideally 1-3 sentences. Reply only in plain text, never use formatting.
"""
            ),
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(
        context,
        user_params=LLMUserAggregatorParams(aggregation_timeout=0.05),
    )

    rtvi = RTVIProcessor()

    pipeline = Pipeline([
        transport.input(),
        rtvi,
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @stt.event_handler("on_stt_update")
    async def on_stt_update(stt, frame):
        logger.info(f"[STT] {frame.text!r}")

    @llm.event_handler("on_llm_context_updated")
    async def on_llm_context_updated(llm, frame):
        msgs = frame.context.get_messages()
        logger.info(f"[LLM] Context has {len(msgs)} messages:")
        for m in msgs:
            role = m.get("role", "?")
            content = m.get("content", "")
            if content:
                logger.info(f"  [{role}] {content[:200]}")

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.cancel()

    runner = PipelineRunner()
    await runner.run(task)


# ---------------------------------------------------------------------------
# Modal containers: bot + frontend
# ---------------------------------------------------------------------------

@app.cls(
    image=bot_image,
    timeout=30 * MINUTES,
    enable_memory_snapshot=True,
    max_inputs=1,
)
class VoiceAgent:
    @modal.enter(snap=True)
    def load(self):
        pass

    @modal.method()
    async def run_bot(self, d: modal.Dict):
        try:
            offer = await d.get.aio("offer")
            ice_servers = await d.get.aio("ice_servers")
            ice_servers = [IceServer(**s) for s in ice_servers]

            conn = SmallWebRTCConnection(ice_servers)
            await conn.initialize(sdp=offer["sdp"], type=offer["type"])

            @conn.event_handler("closed")
            async def handle_closed(conn):
                logger.info("WebRTC closed")

            bot_task = asyncio.create_task(run_bot(conn))
            answer = conn.get_answer()
            await d.put.aio("answer", answer)
            await bot_task
        except Exception as e:
            raise RuntimeError(f"Bot error: {e}")

    @modal.method()
    def ping(self):
        return "pong"


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

from fastapi import FastAPI
from fastapi.responses import HTMLResponse


FRONTEND_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Voice Agent</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }
        button { padding: 12px 24px; font-size: 16px; cursor: pointer; margin: 8px 4px; }
        #status { margin: 16px 0; font-weight: bold; }
        #logs { white-space: pre-wrap; background: #1e1e1e; color: #d4d4d4; padding: 16px; border-radius: 8px; min-height: 200px; max-height: 500px; overflow-y: auto; margin-top: 16px; font-family: monospace; font-size: 13px; }
        .connected { color: green; }        .disconnected { color: red; }
    </style>
</head>
<body>
    <h1>Voice Agent</h1>
    <button id="startBtn" onclick="start()">Start Conversation</button>
    <button id="stopBtn" onclick="stop()" disabled>Stop</button>
    <div id="status" class="disconnected">Disconnected</div>

    <script>
    let pc = null;

    async function start() {
        document.getElementById('startBtn').disabled = true;
        document.getElementById('status').textContent = 'Connecting...';

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            pc = new RTCPeerConnection({
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
            });

            stream.getTracks().forEach(t => pc.addTrack(t, stream));

            pc.ontrack = (event) => {
                const audio = new Audio();
                audio.srcObject = event.streams[0];
                audio.play();
            };

            pc.oniceconnectionstatechange = () => {
                if (pc.iceConnectionState === 'disconnected' || pc.iceConnectionState === 'failed') {
                    stop();
                }
            };

            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);

            await new Promise(resolve => {
                if (pc.iceGatheringState === 'complete') resolve();
                else pc.onicegatheringcomplete = resolve;
                pc.onicecandidate = e => { if (!e.candidate) resolve(); };
            });

            const resp = await fetch('/offer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sdp: pc.localDescription.sdp,
                    type: pc.localDescription.type,
                }),
            });

            const answer = await resp.json();
            await pc.setRemoteDescription(answer);

            document.getElementById('status').textContent = 'Connected';
            document.getElementById('status').className = 'connected';
            document.getElementById('stopBtn').disabled = false;
        } catch (e) {
            console.error(e);
            document.getElementById('status').textContent = 'Error: ' + e.message;
            document.getElementById('startBtn').disabled = false;
        }
    }

    function stop() {
        if (pc) { pc.close(); pc = null; }
        document.getElementById('status').textContent = 'Disconnected';
        document.getElementById('status').className = 'disconnected';
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
    }
    </script>
</body>
</html>"""


@app.function(image=bot_image, min_containers=1)
@modal.asgi_app()
@modal.concurrent(max_inputs=100)
def serve_frontend():
    web_app = FastAPI()

    @web_app.get("/")
    async def root():
        return HTMLResponse(FRONTEND_HTML)

    @web_app.post("/offer")
    async def offer(offer: dict):
        ice_servers = [{"urls": "stun:stun.l.google.com:19302"}]

        d = modal.Dict.from_name(f"offer-{uuid.uuid4()}", create_if_missing=True)
        await d.put.aio("ice_servers", ice_servers)
        await d.put.aio("offer", offer)

        bot_call = await VoiceAgent().run_bot.spawn.aio(d)

        try:
            for _ in range(300):  # 30s timeout
                if await d.contains.aio("answer"):
                    return await d.get.aio("answer")
                await asyncio.sleep(0.1)
            raise TimeoutError("Bot did not produce answer in time")
        except Exception as e:
            logger.error(f"Offer error: {e}")
            bot_call.cancel()
            raise e

    return web_app


if __name__ == "__main__":
    bot = modal.Cls.from_name(APP_NAME, "VoiceAgent")
    for _ in range(5):
        start = time.time()
        bot().ping.remote()
        print(f"Ping: {time.time() - start:.3f}s")
        time.sleep(10.0)
