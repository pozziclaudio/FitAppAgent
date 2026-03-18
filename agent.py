"""
FitTrack Voice Agent — LiveKit STT-only agent.

Joins a LiveKit room, subscribes to the user's audio track,
runs Deepgram streaming STT, and sends transcriptions back
to the client via data channel as JSON messages.

No LLM, no TTS — just speech-to-text transcription relay.
"""

import json
import logging
import asyncio

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.agents.stt import SpeechEvent, SpeechEventType
from livekit.plugins import deepgram

logger = logging.getLogger("fittrack-agent")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    """Called when a user joins a room — the agent connects and starts transcribing."""

    logger.info(f"Agent connecting to room: {ctx.room.name}")

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    # Wait for the participant to publish an audio track
    audio_track = None
    for pub in participant.track_publications.values():
        if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
            audio_track = pub.track
            break

    if not audio_track:
        # Wait for track to be published
        track_event = asyncio.Event()
        received_track = None

        def on_track_subscribed(track, publication, p):
            nonlocal received_track
            if p.identity == participant.identity and track.kind == rtc.TrackKind.KIND_AUDIO:
                received_track = track
                track_event.set()

        ctx.room.on("track_subscribed", on_track_subscribed)

        logger.info("Waiting for audio track...")
        await asyncio.wait_for(track_event.wait(), timeout=30.0)
        audio_track = received_track

    if not audio_track:
        logger.error("No audio track received from participant")
        return

    logger.info(f"Got audio track: {audio_track.sid}")

    # Create Deepgram STT
    stt = deepgram.STT(
        model="nova-3",
        language="en",
        interim_results=True,
        punctuate=False,
    )

    stt_stream = stt.stream()

    # Forward audio frames to STT
    audio_stream = rtc.AudioStream(audio_track)

    async def forward_audio():
        try:
            async for frame_event in audio_stream:
                stt_stream.push_frame(frame_event.frame)
        except Exception as e:
            logger.error(f"Audio forwarding error: {e}")
        finally:
            await stt_stream.aclose()

    # Process STT events and send to client
    async def process_transcriptions():
        try:
            async for event in stt_stream:
                if not isinstance(event, SpeechEvent):
                    continue
                if event.type == SpeechEventType.INTERIM_TRANSCRIPT:
                    text = event.alternatives[0].text if event.alternatives else ""
                    if text:
                        await send_transcript(ctx, text, is_final=False)
                elif event.type == SpeechEventType.FINAL_TRANSCRIPT:
                    text = (event.alternatives[0].text if event.alternatives else "").strip()
                    if text:
                        logger.info(f"Transcript: {text}")
                        await send_transcript(ctx, text, is_final=True)
        except Exception as e:
            logger.error(f"STT processing error: {e}")

    await asyncio.gather(forward_audio(), process_transcriptions())


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
        reliable=is_final,
        topic="transcription",
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="fittrack-voice",
        ),
    )
