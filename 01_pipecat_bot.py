import asyncio
import json
import base64
import uuid
from typing import AsyncGenerator

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
# Hardcoded configuration
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
# ReadyFormAI Voice Assistant

You are ReadyFormAI, a patient, friendly voice assistant designed to help users fill out PDF forms.

---

## MOST IMPORTANT RULE

**DO NOT output speech/text when you are making tool calls.**

Speech and tool calls DO NOT MIX. Pick ONE per turn:
- Making tool calls? -> Output ONLY tool calls, no text
- Need to speak? -> Output ONLY speech, no tool calls

### WHY?

When you output both speech AND tools, the speech plays FIRST while tools run in silence. The conversation breaks because:
1. Your speech plays
2. User starts responding
3. Tools are still running in the background
4. Everything gets out of sync

### CORRECT PATTERN

User gives info -> You output ONLY tool calls (no speech) -> Tools run -> You speak in your next turn

**Example:**
User: "My name is Oliver Chen and I'm filing for overtime"
You: [tool calls only - NO SPEECH]
[tools execute]
You (next turn): "Got it Oliver! What dates does the overtime cover?"

### WRONG PATTERN (NEVER DO THIS)

User: "My name is Oliver Chen"
You: "Great, filling that in! What else?" [setFieldValue: Name, Oliver Chen]

This breaks because "Great, filling that in!" plays BEFORE the tool runs!

---

## Your Purpose

Fill out the form using tools. Make tool calls immediately when user gives information.

## Tool Usage

- **setFieldValue** - Enter data into fields
- **getFieldValue** - Check a field's current value
- **focusField** - Highlight a field (auto-scrolls)
- **navigateToSection** - Jump to a form section
- **getFormProgress** - Check completion percentage
- **getFormSummary** - Review all values
- **confirmValue** - Mark a field as confirmed
- **showHelp** - Show help for a field
- **showTooltip** - Display field tooltip popup
- **hangUp** - End call and show completed PDF

Make MULTIPLE tool calls in one turn if user gives multiple pieces of info.

## Intelligent Behavior

### 1. Multi-Field Extraction
When the user provides multiple pieces of information in one sentence, fill ALL relevant fields:
- User: "I'm Frank Miller delivering wheat from 123 Farm Lane"
- Turn 1: [setFieldValue: Producer Name, Frank Miller] [setFieldValue: Grain Type, Wheat] [setFieldValue: Address, 123 Farm Lane]
- Turn 2 (after results): "Got it, Frank. I've entered your name, grain type, and address."

### 2. Automatic Unit Conversion
Check the field's unit and convert if the user gives a different unit.
- If a weight field expects tonnes but user says "45,000 kilograms": enter "45"
- Always tell the user about conversions in your next turn.

### 3. Date Handling
Convert relative dates to actual dates in YYYY-MM-DD format. NEVER enter text like "two weeks ago".

### 4. Checkbox and Boolean Fields
Normalize: "yes"/"check it"/"true" -> "Yes" or "checked"; "no"/"uncheck"/"false" -> "No" or ""

### 5. Smart Field Inference
Use context to determine which fields to fill:
- "ticket number is GR-89" -> probably the Scale Ticket field
- Mention of weight -> determine if gross or vehicle from context

## Response Style (for speech-only turns)

Be brief:
- WRONG: "I have successfully updated the Producer Name field to Frank Miller."
- RIGHT: "Got it, Frank! Ticket number?"

Keep momentum - don't over-confirm every field.

## Starting the Conversation

"Hi! Let's fill out your form. What would you like to start with?"

Or just listen and fill as they speak.

## Key Rules

1. NO speech when making tool calls - most important!
2. Fill multiple fields at once when user gives multiple values
3. Convert units silently, explain after
4. Keep speech brief - confirm and move forward
5. Use tools actively - nothing happens without them
6. Use hangUp with reason "completed" when user says they're done
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "setFieldValue",
            "description": "Set a form field value. Use when user provides information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fieldName": {
                        "type": "string",
                        "description": "The exact field name to update",
                    },
                    "value": {
                        "type": "string",
                        "description": "The value to set (always as a string)",
                    },
                },
                "required": ["fieldName", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "getFieldValue",
            "description": "Get the current value of a form field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fieldName": {
                        "type": "string",
                        "description": "The field name to retrieve",
                    },
                },
                "required": ["fieldName"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "focusField",
            "description": "Highlight a field in the UI and scroll to it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fieldName": {
                        "type": "string",
                        "description": "The field name to highlight",
                    },
                },
                "required": ["fieldName"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "navigateToSection",
            "description": "Scroll to and highlight a specific section of the form.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sectionId": {
                        "type": "string",
                        "description": "The section ID to navigate to",
                    },
                },
                "required": ["sectionId"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "getFormProgress",
            "description": "Get form completion progress (completed/total fields and percentage).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "getFormSummary",
            "description": "Get complete form summary with all field values.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirmValue",
            "description": "Mark a field as confirmed by the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fieldName": {
                        "type": "string",
                        "description": "The field name that was confirmed",
                    },
                },
                "required": ["fieldName"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "showHelp",
            "description": "Show help information for a field or general topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic to explain (field name or 'general')",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "showTooltip",
            "description": "Display the tooltip/description popup for a specific field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fieldName": {
                        "type": "string",
                        "description": "The field name to show tooltip for",
                    },
                },
                "required": ["fieldName"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hangUp",
            "description": "End the call and show the completed PDF. Use when user says they are done, finished, or wants to submit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Reason for ending",
                        "enum": ["completed", "user_requested"],
                    },
                },
                "required": ["reason"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Service bridge classes
# ---------------------------------------------------------------------------

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    EndFrame,
    CancelFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
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
    IceServer,
    SmallWebRTCConnection,
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
        self.function_call = await self._cls.run_tunnel_client.spawn.aio(
            self._url_dict
        )

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
        except Exception as e:
            yield ErrorFrame(f"TTS send error: {e}")
            return
        yield None


# ---------------------------------------------------------------------------
# Bot pipeline
# ---------------------------------------------------------------------------


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
        params=OpenAILLMService.InputParams(temperature=0),
    )

    tts = ModalKyutaiTTSService(modal_tunnel_manager=tts_tunnel, sample_rate=24000)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = OpenAILLMContext(messages, tools=TOOLS)
    context_aggregator = llm.create_context_aggregator(
        context,
        user_params=LLMUserAggregatorParams(aggregation_timeout=0.05),
    )

    rtvi = RTVIProcessor()

    # Register RTVI tool handlers on the LLM so function calls are
    # forwarded to the client natively via the data channel.
    for tool in TOOLS:
        fn_name = tool["function"]["name"]
        llm.register_function(fn_name, rtvi.handle_function_call)

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

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
# Web server
# ---------------------------------------------------------------------------

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


@app.function(image=bot_image, min_containers=1, timeout=30 * MINUTES)
@modal.asgi_app()
@modal.concurrent(max_inputs=100)
def serve_frontend():
    web_app = FastAPI()

    # CORS middleware - allow all origins for frontend connectivity
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global exception handler so 500 errors include CORS headers
    @web_app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )

    @web_app.get("/")
    async def health():
        return {"status": "ok"}

    @web_app.post("/offer")
    async def offer(body: dict):
        """Direct WebRTC connection - no session_id checks."""
        ice_servers = [IceServer(urls="stun:stun.l.google.com:19302")]
        conn = SmallWebRTCConnection(ice_servers)
        await conn.initialize(sdp=body["sdp"], type=body["type"])
        asyncio.create_task(run_bot(conn))
        return conn.get_answer()

    @web_app.patch("/offer")
    async def patch_offer():
        """Absorb late ICE candidates."""
        return {"status": "ok"}

    return web_app
