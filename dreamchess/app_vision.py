"""The robot's eyes, for the app: a camera view, and how it becomes the 2D board.

In a simulation, a simulated top camera renders the physical board, and the real vision code (vision.py) calibrates
on it and reads every frame, so the page shows the same camera-to-board conversion the station will run.

On the real robot, the station saves the frames it hands Dream-1 (~/.dreamlabs/last-run_inputs, about once a second
during each run), so the page shows exactly what Dream-1 sees. Reading the board from the live camera waits on the
station sharing its camera; until then the board comes from the person's answers.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import chess
import cv2

from .core import Occupancy
from .synth import PIECE_SIZES, SynthCamera
from .vision import CalibrationError, calibrate, classify, draw_debug

ROLES = ("top", "left", "right")


def _jpeg(image) -> bytes:
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return data.tobytes() if ok else b""


class SimEyes:
    """A simulated top camera plus the real vision code reading it."""

    def __init__(self, seed: int = 1):
        self.camera = SynthCamera(tilt_deg=25, seed=seed, color_noise=3.0, side="black")  # the robot's side
        self.lock = threading.Lock()
        self.calibration = None
        self.error = ""
        self._cache_key = None
        self._cache: dict = {}
        threading.Thread(target=self._calibrate, daemon=True).start()

    def _calibrate(self) -> None:
        """Calibrate the way the station would: the empty board, then the starting position."""
        try:
            with self.lock:
                self.camera.pieces = {}
                empty = [self.camera.read() for _ in range(3)]
                self.camera.pieces = self._pieces(chess.Board(), {})
                start = [self.camera.read() for _ in range(3)]
                calibration, _ = calibrate(empty, start, self.camera.corners(1.5))
                self.calibration = calibration
        except CalibrationError as err:
            self.error = f"camera calibration failed: {err}"

    @staticmethod
    def _pieces(board: chess.Board, physical: Occupancy) -> dict:
        """What stands where, with each piece's real size where the game knows it."""
        if not physical:
            physical = {chess.square_name(s): ("w" if p.color else "b") for s, p in board.piece_map().items()}
        pieces = {}
        for square, color in physical.items():
            piece = board.piece_at(chess.parse_square(square))
            kind = piece.piece_type if piece and ("w" if piece.color else "b") == color else chess.PAWN
            pieces[square] = (color, *PIECE_SIZES[kind])
        return pieces

    def view(self, fen: str | None, physical: Occupancy) -> dict:
        """The camera's frame, the frame with the detected grid and pieces drawn on it, and the detected board."""
        key = (fen, tuple(sorted(physical.items())))
        with self.lock:
            if key == self._cache_key:
                return self._cache
            board = chess.Board(fen) if fen else chess.Board()
            self.camera.pieces = self._pieces(board, physical)
            frame = self.camera.read()
            result = {"raw": _jpeg(frame.color), "ready": self.calibration is not None, "error": self.error}
            if self.calibration is not None:
                detected = classify(frame, self.calibration)
                result.update({
                    "overlay": _jpeg(draw_debug(frame, self.calibration, detected)),
                    "detected": detected,
                    "agrees": detected == physical,
                })
            self._cache_key, self._cache = key, result
            return result


class StationFrames:
    """The frames the station last handed Dream-1, one folder per camera, newest file wins."""

    def __init__(self, folder: str):
        self.folder = Path(folder).expanduser()

    def latest(self, role: str) -> tuple[bytes, float] | None:
        if role not in ROLES:
            return None
        files = sorted((self.folder / role).glob("*.jpg"), key=lambda p: p.stat().st_mtime) if (self.folder / role).is_dir() else []
        if not files:
            return None
        newest = files[-1]
        return newest.read_bytes(), newest.stat().st_mtime


def station_telemetry(url: str) -> dict:
    """The parts of the station's status that show Dream-1 at work: its state, model, and API round trips."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/status", timeout=2) as reply:
            status = json.loads(reply.read())
    except (urllib.error.URLError, TimeoutError, ValueError, ConnectionError, OSError):
        return {"reachable": False}
    timing = status.get("timing") or {}
    return {
        "reachable": True,
        "state": status.get("state"),
        "checkpoint": status.get("checkpoint"),
        "prompt": status.get("prompt"),
        "run": status.get("run"),
        "chunks": status.get("chunks"),
        "run_elapsed_s": status.get("run_elapsed_s"),
        "last_rtt_ms": timing.get("last_rtt_ms"),
        "p50_ms": timing.get("p50_ms"),
        "serve_infer_ms": timing.get("serve_infer_ms"),
        "last_error": status.get("last_error"),
        "checked_at": time.strftime("%H:%M:%S"),
    }
