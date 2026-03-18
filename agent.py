"""
FitTrack Voice Agent — LiveKit STT-only agent.

Uses AgentSession with Silero VAD + Deepgram STT.
Sends transcriptions to the client via data channel.
"""

import json
import logging
import asyncio

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import deepgram, silero

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fittrack-agent")


class TranscriptionAgent(Agent):
    """Minimal agent — no LLM, no TTS, just transcription relay."""

    def __init__(self):
        super().__init__(instructions="You are a transcription relay. Do not speak.")


async def entrypoint(ctx: JobContext):
    """Called when a user joins a room."""
    logger.info(f"Agent connecting to room: {ctx.room.name}")

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    # Create VAD + STT
    vad = silero.VAD.load()
    stt = deepgram.STT(
        model="nova-3",
        language="en",
        interim_results=True,
        punctuate=False,
    )

    # AgentSession with VAD to detect speech boundaries
    session = AgentSession(
        stt=stt,
        vad=vad,
    )

    agent = TranscriptionAgent()

    # Listen for user transcription events and forward via data channel
    @session.on("user_speech_committed")
    def on_committed(msg):
        text = msg.content if hasattr(msg, 'content') else str(msg)
        if text and text.strip():
            logger.info(f"FINAL transcript: {text.strip()}")
            asyncio.ensure_future(_send_transcript(ctx, text.strip(), True))

    @session.on("user_started_speaking")
    def on_start():
        logger.info(">>> User started speaking")

    @session.on("user_stopped_speaking")
    def on_stop():
        logger.info("<<< User stopped speaking")

    logger.info("Starting AgentSession with VAD + STT...")
    await session.start(
        agent=agent,
        room=ctx.room,
    )
    logger.info("AgentSession started successfully — listening for speech")


async def _send_transcript(ctx: JobContext, text: str, is_final: bool):
    """Send transcription to client via data channel."""
    payload = json.dumps({
        "transcript": text,
        "isFinal": is_final,
    }).encode("utf-8")

    try:
        await ctx.room.local_participant.publish_data(
            payload,
            reliable=is_final,
            topic="transcription",
        )
        logger.info(f"Sent to client: {text}")
    except Exception as e:
        logger.error(f"Failed to send transcript: {e}")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="fittrack-voice",
        ),
    )
