"""
FitTrack Voice Agent — LiveKit STT-only agent.

Joins a LiveKit room, subscribes to the user's audio track,
runs Deepgram streaming STT, and sends transcriptions back
to the client via data channel as JSON messages.
"""

import json
import logging
import asyncio
import traceback

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.agents.stt import SpeechEvent, SpeechEventType
from livekit.plugins import deepgram

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fittrack-agent")


async def entrypoint(ctx: JobContext):
    """Called when a user joins a room — the agent connects and starts transcribing."""

    try:
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
        logger.info("Creating Deepgram STT...")
        stt = deepgram.STT(
            model="nova-3",
            language="en",
            interim_results=True,
            punctuate=False,
        )

        logger.info("Creating STT stream...")
        stt_stream = stt.stream()

        logger.info("Creating audio stream...")
        audio_stream = rtc.AudioStream(audio_track)

        async def forward_audio():
            logger.info("Audio forwarding started")
            frame_count = 0
            try:
                async for frame_event in audio_stream:
                    stt_stream.push_frame(frame_event.frame)
                    frame_count += 1
                    if frame_count == 1:
                        logger.info("First audio frame received and forwarded to STT")
                    if frame_count % 500 == 0:
                        logger.info(f"Forwarded {frame_count} audio frames")
            except Exception as e:
                logger.error(f"Audio forwarding error: {e}")
                logger.error(traceback.format_exc())
            finally:
                logger.info(f"Audio forwarding ended after {frame_count} frames")
                await stt_stream.aclose()

        async def process_transcriptions():
            logger.info("Transcription processing started")
            try:
                async for event in stt_stream:
                    logger.info(f"STT event type: {type(event).__name__}")
                    if not isinstance(event, SpeechEvent):
                        continue
                    logger.info(f"Speech event: {event.type}")
                    if event.type == SpeechEventType.INTERIM_TRANSCRIPT:
                        text = event.alternatives[0].text if event.alternatives else ""
                        if text:
                            logger.info(f"Interim: {text}")
                            await send_transcript(ctx, text, is_final=False)
                    elif event.type == SpeechEventType.FINAL_TRANSCRIPT:
                        text = (event.alternatives[0].text if event.alternatives else "").strip()
                        if text:
                            logger.info(f"FINAL: {text}")
                            await send_transcript(ctx, text, is_final=True)
            except Exception as e:
                logger.error(f"STT processing error: {e}")
                logger.error(traceback.format_exc())

        logger.info("Starting audio forwarding and transcription processing...")
        await asyncio.gather(forward_audio(), process_transcriptions())

    except Exception as e:
        logger.error(f"ENTRYPOINT ERROR: {e}")
        logger.error(traceback.format_exc())


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
