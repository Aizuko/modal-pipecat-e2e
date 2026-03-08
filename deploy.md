# Voice Agent — Deploy Instructions

## Architecture

```
Browser ↔ WebRTC ↔ Bot (CPU) ↔ WS tunnel ↔ Parakeet STT (L40S GPU)
                              ↔ HTTP      ↔ vLLM LLM (L40S GPU)
                              ↔ WS tunnel ↔ Kyutai TTS (L40S GPU)
```

## Prerequisites

```bash
pip install modal
modal setup
```

## Deploy (in order)

```bash
# 1. LLM (skip if already deployed)
modal deploy llm_step.py

# 2. STT
modal deploy parakeet_stt.py

# 3. TTS
modal deploy kyutai_tts.py

# 4. Bot + frontend
modal deploy bot.py
```

## Warm up snapshots (optional, reduces cold starts)

Each command pings the service ~5 times to trigger snapshot creation.
First 2-5 cold starts will be slow, then subsequent ones restore from snapshot.

```bash
python parakeet_stt.py
python kyutai_tts.py
python bot.py
```

## Use it

After deploying `bot.py`, Modal prints a URL like:

```
https://<workspace>--voice-agent-serve-frontend.modal.run
```

Open that in your browser, click **Start Conversation**, and speak.

## Troubleshooting

- If the bot can't reach the LLM, check that `VLLM_BASE_URL` in `bot.py` matches your deployed vLLM URL
- If STT/TTS tunnels time out, make sure those services are deployed first
- Check Modal dashboard → your app → Containers tab for snapshot status (⚡ = restored from snapshot)
