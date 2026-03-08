# 1. Llm on modal

Simply launch a llama on modal and let it sit there. Test openai compatibility
with vllm

# 2. tts on modal

Setup kyutai tts on modal. Make sure it works as an endpoint when sending text

# 3. Add parakeet stt in the front

Add a parakeet container in the front. Just audio to text right now. Make sure
it works over api

# 4. Connect things with pipecat

Connect all the pieces together with pipecat websockets, use their local
inference to test the flow.

# 5. Add tool calling through pipecat

Check what tools we need for Readyform AI and put those into PipeCat.

# 6. Integrate into readyformai

Move integration over into ReadyForm AI. So ReadyForm AI is now running entirely
on our pipecat instance.

# 7. Replace gemini

Add another container to replace gemini in readyform ai.
