"""Speak the robot's lines over LiveKit: the robot joins a LiveKit room and talks with its own voice.

Adapted from the team's voices.py. Text-to-speech runs on LiveKit Inference, billed to the LiveKit
Cloud project, so no separate TTS key is needed. Only used with `--voice livekit`:

    pip install "livekit-agents>=1.2" livekit livekit-api aiohttp python-dotenv
    pip install sounddevice            # only for --voice-local, to also play on this computer's speaker

It reads these from the environment, or from a .env file in the repo folder (never commit it):

    LIVEKIT_URL=wss://<project>.livekit.cloud
    LIVEKIT_API_KEY=...
    LIVEKIT_API_SECRET=...

To listen, join the room (default `dream-chess`) from another device, for example with LiveKit's
web client and a token from `lk token create`, or add --voice-local to play it here too.
"""
from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import Callable, Optional

DEFAULT_MODEL = "inworld/inworld-tts-2:Edward"  # "provider/model:voice"; your voices.py's Blaze
DEFAULT_ROOM = "dream-chess"
IDENTITY, NAME = "dream-chess-robot", "Robot"


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for directory in (Path(__file__).resolve().parent.parent, Path.cwd()):
        for name, override in ((".env", False), (".env.local", True)):
            if (directory / name).exists():
                load_dotenv(directory / name, override=override)


class _LocalSpeaker:
    """Also plays the audio on this computer's speaker."""

    def __init__(self, sample_rate: int, channels: int):
        import sounddevice as sd

        self.stream = sd.RawOutputStream(samplerate=sample_rate, channels=channels, dtype="int16")
        self.stream.start()

    def write(self, frame) -> None:
        self.stream.write(bytes(frame.data))

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()


class LiveKitSpeaker:
    """Runs asyncio on its own thread, so the game loop, which isn't async, can call say() directly."""

    def __init__(self, room: Optional[str] = None, model: Optional[str] = None, local: bool = False,
                 log: Callable[[str], None] = print):
        load_env()
        self.room_name = room or os.environ.get("CHESS_ROOM", DEFAULT_ROOM)
        self.model = model or DEFAULT_MODEL
        self.local = local
        self.log = log
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.error: Optional[BaseException] = None
        self.thread = threading.Thread(target=self._run, name="robot-voice", daemon=True)

    # ---------- called from the game ----------
    def start(self, timeout: float = 30.0) -> "LiveKitSpeaker":
        missing = [k for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET") if not os.environ.get(k)]
        if missing:
            raise SystemExit(f"--voice livekit needs {', '.join(missing)} in the environment or a .env file")
        self.thread.start()
        if not self.ready.wait(timeout):
            raise SystemExit("couldn't connect to LiveKit in time")
        if self.error:
            raise SystemExit(f"couldn't connect to LiveKit: {type(self.error).__name__}: {self.error}")
        return self

    def say(self, text: str) -> None:
        """Queue a line and return at once. Lines play in order, one at a time."""
        if not text:
            return
        self.log(f"Robot says: {text}")
        future = asyncio.run_coroutine_threadsafe(self._say(text), self.loop)
        future.add_done_callback(self._report)

    def _report(self, future) -> None:
        if future.exception() is not None:
            err = future.exception()
            self.log(f"[voice] couldn't speak: {type(err).__name__}: {err}")

    def close(self) -> None:
        if self.loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self._close(), self.loop).result(timeout=60)
            except Exception as err:
                self.log(f"[voice] closing: {type(err).__name__}: {err}")
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)

    # ---------- the voice's own thread ----------
    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._connect())
        except BaseException as err:
            self.error = err
            self.ready.set()
            return
        self.ready.set()
        self.loop.run_forever()

    async def _connect(self) -> None:
        import aiohttp
        from livekit import api, rtc
        from livekit.agents import inference

        token = (api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
                 .with_identity(IDENTITY).with_name(NAME)
                 .with_grants(api.VideoGrants(room_join=True, room=self.room_name, can_publish=True,
                                              can_subscribe=False))
                 .to_jwt())
        self.http = aiohttp.ClientSession()
        self.lock = asyncio.Lock()  # one line at a time, in order
        self.room = rtc.Room()
        await self.room.connect(os.environ["LIVEKIT_URL"], token)
        self.tts = inference.TTS(self.model, http_session=self.http)
        self.source = rtc.AudioSource(self.tts.sample_rate, self.tts.num_channels)
        track = rtc.LocalAudioTrack.create_audio_track("robot-voice", self.source)
        await self.room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        self.speaker = _LocalSpeaker(self.tts.sample_rate, self.tts.num_channels) if self.local else None
        self.log(f"[voice] the robot joined LiveKit room {self.room_name}"
                 f"{', also playing on this computer' if self.local else ''}")

    async def _say(self, text: str) -> None:
        async with self.lock:
            async with self.tts.synthesize(text) as stream:
                async for audio in stream:
                    await self.source.capture_frame(audio.frame)
                    if self.speaker is not None:
                        await asyncio.to_thread(self.speaker.write, audio.frame)
            await self.source.wait_for_playout()

    async def _close(self) -> None:
        async with self.lock:  # let the last line finish
            pass
        if self.speaker is not None:
            self.speaker.close()
        await self.tts.aclose()
        await self.room.disconnect()
        await self.http.close()


if __name__ == "__main__":
    # A quick check from the repo folder: python -m dreamchess.livekit_voice
    speaker = LiveKitSpeaker(local=True).start()
    speaker.say("Hello! I'm the Dream Chess robot. My knight moves from g8 to f6.")
    speaker.say("Good check!")
    speaker.close()
