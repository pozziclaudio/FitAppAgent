"""
FitTrack Voice Agent — LiveKit STT-only agent.

Joins a LiveKit room, subscribes to the user's audio track,
runs Deepgram streaming STT, and sends transcriptions back
to the client via data channel as JSON messages.

No LLM, no TTS — just speech-to-text transcription relay.
"""

import json
import logging

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.stt import SpeechEvent, SpeechEventType
from livekit.plugins import deepgram

logger = logging.getLogger("fittrack-agent")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    """Called when a user joins a room — the agent connects and starts transcribing."""

    logger.info(f"Agent connecting to room: {ctx.room.name}")

    # Wait for a participant (the user) to connect
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    # Create Deepgram STT stream
    stt = deepgram.STT(
        model="nova-3",
        language="en",
        interim_results=True,
        punctuate=False,
    )

    # Stream audio from the participant through STT
    stt_stream = stt.stream()

    # Forward audio track to STT
    async def process_audio():
        async for event in rtc.AudioStream(
            participant=participant,
            track=None,  # auto-select first audio track
        ):
            stt_stream.push_frame(event.frame)

    # Process STT results and send to client
    async def process_stt():
        async for event in stt_stream:
            if isinstance(event, SpeechEvent):
                if event.type == SpeechEventType.INTERIM_TRANSCRIPT:
                    await send_transcript(ctx, event.alternatives[0].text, is_final=False)
                elif event.type == SpeechEventType.FINAL_TRANSCRIPT:
                    text = event.alternatives[0].text.strip()
                    if text:
                        logger.info(f"Final transcript: {text}")
                        await send_transcript(ctx, text, is_final=True)

    import asyncio
    await asyncio.gather(process_audio(), process_stt())


async def send_transcript(ctx: JobContext, text: str, is_final: bool):
    """Send a transcription result to all participants via data channel."""
    if not text:
        return

    payload = json.dumps({
        "transcript": text,
        "isFinal": is_final,
    }).encode("utf-8")

    await ctx.room.local_participant.publish_data(
        payload,
        reliable=is_final,  # Use reliable delivery for final transcripts
        topic="transcription",
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="fittrack-voice",
        ),
    )
