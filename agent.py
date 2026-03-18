"""
FitTrack Voice Agent — LiveKit STT-only agent.

Uses AgentSession with Deepgram STT. Transcriptions are forwarded
to the client automatically via LiveKit's built-in transcription system.
"""

import logging

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import deepgram

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fittrack-agent")


class TranscriptionAgent(Agent):
    """Minimal agent that just relays STT transcriptions — no LLM, no TTS."""

    def __init__(self):
        super().__init__(instructions="You are a transcription relay. Do not speak.")


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

    # AgentSession manages the audio pipeline and keeps the process alive
    session = AgentSession(
        stt=stt,
        # No LLM or TTS — transcriptions are forwarded via LiveKit's transcription API
    )

    agent = TranscriptionAgent()

    logger.info("Starting AgentSession...")
    await session.start(
        agent=agent,
        room=ctx.room,
    )
    logger.info("AgentSession started successfully")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="fittrack-voice",
        ),
    )
