# ReadyFormAI modal infrastructure

This repository contains example code to self-host local LLMs on modal. This can
easily be adapted to any locally host GPU setup to ensure privacy.

This includes a basic front end interface in `01_pipecat_bot.py` for testing purposes.

# Deploy Instructions

## Architecture

```
Browser ↔ WebRTC ↔ Bot (CPU) ↔ WS tunnel ↔ Parakeet v0.3 STT (L40S GPU)
                              ↔ HTTP      ↔ Qwen3 30b a3b on vLLM  (L40S GPU)
                              ↔ WS tunnel ↔ Qwen3 TTS (L40S GPU)
```

## Prerequisites

```bash
pip install modal
modal setup
```

## Deploy (in order)

```bash
# 1. LLM (skip if already deployed)
modal deploy 03_vllm_server.py

# 2. STT
modal deploy 02_stt.py

# 3. TTS
modal deploy 04_tts.py

# 4. Bot + frontend
modal deploy 01_pipecat_bot.py
```

## Warm up snapshots (optional, reduces cold starts)

Each command pings the service ~5 times to trigger snapshot creation.
First 2-5 cold starts will be slow, then subsequent ones restore from snapshot.

```bash
python 02_stt.py
python 04_tts.py
python 01_pipecat_bot.py
```

## Use it

After deploying `01_pipecat_bot.py`, Modal prints a URL like:

```
https://<workspace>--voice-agent-serve-frontend.modal.run
```

Open that in your browser, click **Start Conversation**, and speak.

## Troubleshooting

- If the bot can't reach the LLM, check that `VLLM_BASE_URL` in `01_pipecat_bot.py` matches your deployed vLLM URL
- If STT/TTS tunnels time out, make sure those services are deployed first
- Check Modal dashboard → your app → Containers tab for snapshot status (⚡ = restored from snapshot)
