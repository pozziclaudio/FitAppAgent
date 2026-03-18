"""
FitTrack Voice Agent — LiveKit STT-only agent.

Uses AgentSession with Deepgram STT. Transcriptions are sent
back to the client via data channel as JSON messages.
"""

import json
import logging

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    get_job_context,
)
from livekit.agents.stt import SpeechEvent, SpeechEventType
from livekit.plugins import deepgram

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fittrack-agent")


class TranscriptionAgent(Agent):
    """Minimal agent that just relays STT transcriptions via data channel."""

    def __init__(self):
        super().__init__(instructions="You are a transcription relay. Do not speak.")

    async def on_user_speech_committed(self, message):
        """Called when STT produces a final transcript."""
        text = message.get("text", "") if isinstance(message, dict) else str(message)
        if text:
            logger.info(f"FINAL: {text}")
            await self._send_transcript(text, is_final=True)
        # Return empty string to prevent any LLM/TTS response
        return ""


async def entrypoint(ctx: JobContext):
    """Called when a user joins a room."""
    logger.info(f"Agent connecting to room: {ctx.room.name}")

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    # Create Deepgram STT
    stt = deepgram.STT(
        model="nova-3",
        language="en",
        interim_results=True,
        punctuate=False,
    )

    # Use AgentSession — this keeps the process alive and manages audio properly
    session = AgentSession(
        stt=stt,
        # No LLM or TTS — we only want transcription
    )

    # Listen for all speech events to send interim + final transcripts
    @session.on("user_speech_committed")
    def on_final(msg):
        text = msg.content if hasattr(msg, 'content') else str(msg)
        if text and text.strip():
            logger.info(f"FINAL: {text}")
            import asyncio
            asyncio.ensure_future(_send_transcript(ctx, text.strip(), is_final=True))

    @session.on("user_started_speaking")
    def on_start():
        logger.info("User started speaking")

    @session.on("user_stopped_speaking")
    def on_stop():
        logger.info("User stopped speaking")

    logger.info("Starting AgentSession...")
    await session.start(
        room=ctx.room,
        participant=participant,
    )
    logger.info("AgentSession started successfully")


async def _send_transcript(ctx: JobContext, text: str, is_final: bool):
    """Send a transcription result to all participants via data channel."""
    if not text:
        return

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
        logger.info(f"Sent transcript: {text}")
    except Exception as e:
        logger.error(f"Failed to send transcript: {e}")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="fittrack-voice",
        ),
    )
