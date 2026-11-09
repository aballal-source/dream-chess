"""Dream Chess: the app. Watch the robot think through every step of a game, and play it from the page.

    python app.py --sim --auto-opponent      # dummy data: a simulated robot playing a random opponent
    python app.py --sim                      # simulated robot; you play White from the page
    python app.py                            # the real robot, on the NUC (a person at the e-stop)
    python app.py --public --port 7860       # the public demo site: the simulation only, open to visitors

Open http://127.0.0.1:8800 (on the NUC, from your laptop: ssh -L 8800:127.0.0.1:8800 dream-nuc).
The game itself is run_robot.play(): this only gives it a page instead of a terminal, so the same code runs both.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import chess

import run_robot
from dreamchess.app_vision import ROLES, SimEyes, StationFrames, station_telemetry
from dreamchess.core import occupancy
from dreamchess.human_eyes import GameQuit

PAGE = Path(__file__).parent / "dreamchess" / "app_page.html"
STATS_PAGE = Path(__file__).parent / "dreamchess" / "stats_page.html"
PACED = ("agent", "attempt", "attempt_result", "opponent_move", "help")
LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]")
PUBLIC_RESTART_EVERY_S = 15.0  # on the public site, one visitor's restart can't keep resetting everyone's game


def question_kind(prompt: str) -> str:
    """How the page should offer an answer to one of the game's questions."""
    text = prompt.lower()
    if "your move" in text:
        return "move"
    if "y/n" in text:
        return "yesno"
    if "type yes" in text:
        return "confirm"
    # Before "press enter": this question also says "press Enter for no".
    if "knocked out of place" in text:
        return "bumps"
    if "press enter" in text:
        return "continue"
    if "where is it now" in text:
        return "where"
    return "text"


class Session:
    """One game at a time, run on its own thread, with its questions answered from the page.

    Only the game thread touches the game's board. Whenever it reports something, it also publishes a snapshot of
    the board for the page, taken at that moment, so the page never sees the board mid-change (the opponent-move
    check tries every legal move on it in turn).
    """

    def __init__(self, args):
        self.args = args
        self.boot = uuid.uuid4().hex[:8]  # changes when the app restarts, so an open page knows to start over
        self.lock = threading.Condition()
        self.events: list[dict] = []
        self.pending: dict | None = None
        self.answers: dict[int, str] = {}
        self.next_id = 0
        self.board: chess.Board | None = None
        self.physical = None
        self.snapshot: dict = {}
        self.running = False
        self.quitting = False
        self.thread: threading.Thread | None = None
        self.summary: dict | None = None
        self.game_number = 0
        # The robot's eyes: a simulated camera read by the real vision code, or the frames the station hands Dream-1.
        self._eyes: SimEyes | None = None
        self.frames = None if args.sim else StationFrames(getattr(args, "frames", "~/.dreamlabs/last-run_inputs"))
        # The station's top camera, read into the board: the checker agent after each robot step, and the person's
        # move when they say they've made it. A picture is a short "hold still" run; it stays in memory.
        self.webcam = None
        if not args.sim and getattr(args, "top_camera", True):
            from dreamchess.station_camera import StationSnapshot
            from dreamchess.station_mover import StationClient
            from dreamchess.webcam import BoardCamera

            frames = getattr(args, "frames", "~/.dreamlabs/last-run_inputs")
            self.webcam = BoardCamera(lambda: StationSnapshot(StationClient(args.station, timeout_s=10), frames),
                                      getattr(args, "top_calibration", "~/.dream-chess/top-calibration.npz"),
                                      frames_per_read=1, frames_per_capture=1, live=False)
            args.board_camera = self.webcam
        self._webcam_seen: tuple[float, dict] | None = None
        # Starting the model on DreamAPI before the first move needs it (tools/warm_dream.py), on the real robot.
        self.warm_file = Path(tempfile.gettempdir()) / f"dream-chess-warm-{os.getpid()}.json"
        self.warm_process: subprocess.Popen | None = None
        self.warm_checkpoint: str | None = None
        self.hand_feed = None  # the hand tracker, which watches each move as it happens (dreamchess/hands.py)
        # What the agent last asked Dream-1 to do, from the game's own events (both modes).
        self.dream: dict = {"state": "idle", "prompt": None, "arm": None, "attempt": None, "calls": 0,
                            "last_seconds": None, "landed": None}
        self.engine_name: str | None = None  # from the game's start event: "stockfish", or the built-in fallback

    # -- what the game thread calls --------------------------------------------------------------------------------
    def _publish(self) -> None:
        """Copy the board and the robot's view of it for the page. Called only from the game thread."""
        board, physical = self.board, self.physical
        if board is None:
            return
        seen = physical() if physical is not None else {}
        logical = occupancy(board)
        snapshot = {
            "fen": board.fen(),
            "last_move": board.peek().uci() if board.move_stack else None,
            "turn": "white" if board.turn else "brown",
            "check": board.is_check(),
            "physical": seen,
            "mismatch": sorted(sq for sq in set(logical) | set(seen) if logical.get(sq) != seen.get(sq)),
        }
        with self.lock:
            self.snapshot = snapshot

    def _add(self, event: dict) -> None:
        with self.lock:
            event = {**event, "seq": len(self.events), "game": self.game_number}
            self.events.append(event)
            self.lock.notify_all()

    def say(self, text: str) -> None:
        if text.strip():
            self._add({"type": "say", "text": text.strip()})
        self._publish()

    def _track_dream(self, event: dict) -> None:
        kind, text = event.get("type"), event.get("text", "")
        with self.lock:
            if kind == "agent" and text.startswith("HOW"):
                found = re.match(r"HOW\s+\[(\d+)/(\d+)\] (.*)", text)
                prompt = found.group(3) if found else text[3:].strip()
                arm = re.search(r"your (left|right) arm", prompt)
                self.dream.update(state="moving", prompt=prompt, arm=arm.group(1) if arm else None,
                                  attempt=f"{found.group(1)}/{found.group(2)}" if found else None, landed=None)
            elif kind == "attempt":
                self.dream.update(state="idle", last_seconds=event.get("seconds"), calls=self.dream["calls"] + 1)
            elif kind == "attempt_result":
                self.dream["landed"] = bool(event.get("landed"))
            elif kind == "agent" and text.startswith("CHECK"):
                self.dream["landed"] = False

    def on_event(self, event: dict) -> None:
        if event.get("type") == "game_start":
            self.engine_name = event.get("engine")
        self._track_dream(event)
        self._add(event)
        self._publish()
        if self.args.sim and self.args.pace and event.get("type") in PACED:
            time.sleep(self.args.pace)  # slow a simulated game down enough to watch

    def on_ready(self, board, physical, stop_arms) -> None:
        self.board, self.physical = board, physical
        self._publish()

    def ask(self, prompt: str) -> str:
        self._publish()
        with self.lock:
            if self.quitting:
                raise GameQuit()
            self.next_id += 1
            qid = self.next_id
            self.pending = {"id": qid, "prompt": prompt.strip(), "kind": question_kind(prompt)}
            self.lock.notify_all()
            while qid not in self.answers:
                if self.quitting:
                    self.pending = None
                    raise GameQuit()
                self.lock.wait(timeout=0.5)
            self.pending = None
            return self.answers.pop(qid)

    # -- what the page and the main thread call --------------------------------------------------------------------
    def answer(self, qid: int, text: str) -> bool:
        with self.lock:
            if not self.pending or self.pending["id"] != qid:
                return False
            self.answers[qid] = text
            self.lock.notify_all()
            return True

    def model(self) -> str:
        """The model the robot runs. On the real robot, the base model unless another was asked for."""
        if self.args.sim:
            return self.args.checkpoint or "Dream-1 (simulated)"
        from dreamchess.station_mover import BASE_MODEL

        return self.args.checkpoint or BASE_MODEL

    def warm_up(self) -> None:
        """Start the model on DreamAPI now, while the person sets up the board, and keep it running."""
        if self.args.sim:
            return
        checkpoint = self.model()
        if self.warm_process is not None and self.warm_process.poll() is None and self.warm_checkpoint == checkpoint:
            return
        self.stop_warm_up()
        python = Path(getattr(self.args, "dream_python", "~/dream-env/bin/python")).expanduser()
        script = Path(__file__).parent / "tools" / "warm_dream.py"
        if not python.exists():
            self.warm_file.write_text(json.dumps({"state": "off", "checkpoint": checkpoint,
                                                  "message": f"no Dream library Python at {python}"}))
            return
        self.warm_checkpoint = checkpoint
        self.warm_process = subprocess.Popen(
            [str(python), "-W", "ignore", str(script), checkpoint, str(self.warm_file), "--parent", str(os.getpid())],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _start_hand_feed(self) -> None:
        if self.args.sim or not getattr(self.args, "supervise", True) or self.hand_feed is not None:
            return
        from dreamchess.hands import HandFeed

        self.hand_feed = HandFeed(getattr(self.args, "dream_python", "~/dream-env/bin/python"),
                                  getattr(self.args, "board_fit", "~/.dream-chess/board_fit.json"))
        self.args.hand_feed = self.hand_feed

    def hand(self) -> dict | None:
        """Where the hand doing the current step is, from the tracker: square, height and grip."""
        if self.hand_feed is None:
            return None
        if not self.hand_feed.ready:
            return {"watching": False, "problem": self.hand_feed.problem or "the hand tracker stopped"}
        sample = self.hand_feed.latest()
        arm = self.dream.get("arm") or "right"
        if not sample or arm not in sample:
            return {"watching": True}
        return {"watching": True, "arm": arm, **sample[arm]}

    def stop_warm_up(self) -> None:
        if self.warm_process is not None and self.warm_process.poll() is None:
            self.warm_process.terminate()
        self.warm_process = None

    def warm_status(self) -> dict | None:
        try:
            status = json.loads(self.warm_file.read_text())
        except (OSError, ValueError):
            return None
        if status.get("checkpoint") != self.model():
            return None
        if status.get("since"):
            status["waited_s"] = round((status.get("ready_at") or time.time()) - status["since"])
        return status

    def start(self) -> bool:
        with self.lock:
            if self.running:
                return False
            self.running, self.quitting, self.summary = True, False, None
            self.board, self.physical = None, None
            self.snapshot = self._starting_snapshot()
            self.game_number += 1
            self.dream.update(state="idle", prompt=None, arm=None, attempt=None, calls=0, last_seconds=None,
                              landed=None)
        self._start_hand_feed()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.warm_up()
        return True

    def _starting_snapshot(self) -> dict:
        """The position the game starts from, shown while the person sets up the board (before the game publishes)."""
        try:
            start = chess.Board(chess.STARTING_FEN if getattr(self.args, "script", None) else self.args.fen)
        except ValueError:
            return {}  # the game itself reports a bad position
        return {"fen": start.fen(), "last_move": None, "turn": "white" if start.turn else "brown",
                "check": start.is_check(), "physical": occupancy(start), "mismatch": [], "setup": True}

    def restart(self, wait_s: float = 20.0) -> bool:
        """End the game now (stopping the arms first, on the real robot) and start a fresh one."""
        if getattr(self.args, "public", False):
            now = time.monotonic()
            if now - getattr(self, "_restarted_at", -1e9) < PUBLIC_RESTART_EVERY_S:
                return False
            self._restarted_at = now
        if self.running:
            if not self.args.sim:
                self.stop_robot()
            with self.lock:
                self.quitting = True
                self.lock.notify_all()
            if self.thread is not None:
                self.thread.join(timeout=wait_s)
            if self.running:
                return False  # still finishing its last step: try again in a moment
        return self.start()

    def _run(self) -> None:
        try:
            summary = run_robot.play(self.args, ask=self.ask, say=self.say, on_event=self.on_event,
                                     on_ready=self.on_ready)
        except GameQuit:
            summary = {"type": "game_end", "ending": "quit"}
        except SystemExit as err:
            summary = {"type": "game_end", "ending": f"didn't start: {err}"}
            self._add({"type": "error", "text": str(err)})
        except Exception as err:  # keep the page alive, stop the arms, and say what broke
            if not self.args.sim:
                self.stop_robot()
            summary = {"type": "game_end", "ending": f"error: {err}"}
            self._add({"type": "error", "text": f"{type(err).__name__}: {err}", "trace": traceback.format_exc()})
        with self.lock:
            self.running, self.summary, self.pending = False, summary, None
            self.lock.notify_all()

    def stop_robot(self) -> str:
        """Ask the station to stop the arms. It goes straight to the station, so it works at any point in a game."""
        if self.args.sim:
            return "This is a simulation: there are no arms to stop."
        from dreamchess.station_mover import StationClient, StationError

        try:
            StationClient(self.args.station, timeout_s=10).stop()
        except StationError as err:
            self._add({"type": "error", "text": f"Couldn't reach the station to stop the arms: {err}. Use the e-stop."})
            return "Couldn't reach the station. Use the e-stop."
        self._add({"type": "say", "text": "Asked the station to stop the arms."})
        return "Asked the station to stop the arms."

    def shutdown(self, wait_s: float = 10.0) -> None:
        """Stop the arms, end the game so it closes its engine and saves its summary, and wait for it."""
        if not self.args.sim:
            self.stop_robot()
        self.stop_warm_up()
        if self.hand_feed is not None:
            self.hand_feed.close()
        with self.lock:
            self.quitting = True
            self.lock.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=wait_s)

    # -- the robot's eyes and brain, for the page ----------------------------------------------------------------
    def eyes(self) -> SimEyes:
        with self.lock:
            if self._eyes is None:  # started on first use: calibrating takes a few seconds
                self._eyes = SimEyes(seed=self.args.seed)
            return self._eyes

    def camera(self, role: str, grid: bool) -> bytes | None:
        """One camera frame. In a simulation, `grid` draws what the vision code detected on it."""
        if role == "webcam":
            if self.webcam is None:
                return None
            from dreamchess.vision import CalibrationError

            try:
                return self.webcam.snapshot(grid)
            except CalibrationError:
                return None
        if role not in ROLES:
            return None
        if self.args.sim:
            if role != "top":
                return None
            with self.lock:
                snapshot = dict(self.snapshot)
            view = self.eyes().view(snapshot.get("fen"), snapshot.get("physical") or {})
            return view.get("overlay") if grid and view.get("overlay") else view["raw"]
        found = self.frames.latest(role)
        return found[0] if found else None

    def vision(self) -> dict:
        """How the top camera's frame became a board: what it detected on each square, and whether that agrees."""
        with self.lock:
            snapshot = dict(self.snapshot)
        physical = snapshot.get("physical") or {}
        if self.webcam is not None:
            return self._webcam_vision(physical)
        if not self.args.sim:
            ages = {role: found[1] for role in ROLES if (found := self.frames.latest(role))}
            return {"source": "person", "ready": True, "detected": physical, "agrees": True,
                    "frames": {role: round(time.time() - mtime, 1) for role, mtime in ages.items()}}
        view = self.eyes().view(snapshot.get("fen"), physical)
        return {"source": "camera", "ready": view["ready"], "error": view["error"],
                "detected": view.get("detected", {}), "agrees": view.get("agrees")}

    def _webcam_vision(self, physical: dict) -> dict:
        from dreamchess.vision import CalibrationError

        status = self.webcam.status()
        out = {"source": "webcam", "ready": status["calibrated"], "calibration": status, "error": status["error"]}
        if not status["calibrated"]:
            return out
        if not self.webcam.live:  # the station's camera: only what the last picture showed, never a new one
            seen = self.webcam.last_seen
            return {**out, "detected": seen or {}, "agrees": seen == physical if seen is not None else None,
                    "picture": status.get("picture_age_s")}
        cached = self._webcam_seen
        if cached is None or time.time() - cached[0] > 1.5:  # reading takes a few frames, so share it between polls
            try:
                cached = (time.time(), self.webcam.read())
            except CalibrationError as err:
                return {**out, "error": str(err)}
            self._webcam_seen = cached
        return {**out, "detected": cached[1], "agrees": cached[1] == physical}

    def webcam_action(self, action: str, body: dict) -> dict:
        """Calibrate the webcam from the page: the empty board, then the starting position; or start over."""
        from dreamchess.vision import CalibrationError

        if self.webcam is None:
            return {"ok": False, "error": "no camera: the top camera works with the real robot, not a simulation"}
        if self.running and self.pending is None:
            return {"ok": False, "error": "the robot is mid-move: take pictures between moves"}
        try:
            if action == "look":
                status = self.webcam.look()
            elif action == "corners":
                status = self.webcam.set_corners(body.get("corners"))
            elif action == "empty":
                status = self.webcam.capture_empty(body.get("a1") or None)
            elif action == "start":
                status = self.webcam.capture_start()
            elif action == "reset":
                status = self.webcam.reset()
            else:
                return {"ok": False, "error": f"no such step: {action}"}
        except CalibrationError as err:
            return {"ok": False, "error": str(err), **self.webcam.status()}
        self._webcam_seen = None
        return {"ok": True, **status}

    def camera_move(self, qid: int) -> dict:
        """The person says they've made their move: read it from the webcam and play it, or say what looks wrong."""
        from dreamchess.vision import CalibrationError
        from dreamchess.webcam import read_move

        with self.lock:
            pending = dict(self.pending) if self.pending else None
            fen = self.snapshot.get("fen")
        if self.webcam is None or not self.webcam.status()["calibrated"]:
            return {"ok": False, "problem": "The webcam isn't calibrated yet.", "fix": "Calibrate it, or type your move."}
        if not pending or pending["id"] != qid or pending["kind"] != "move" or not fen:
            return {"ok": False, "problem": "It isn't your move right now.", "fix": ""}
        try:
            seen = self.webcam.read()
        except CalibrationError as err:
            return {"ok": False, "problem": f"Couldn't read the webcam: {err}", "fix": "Type your move instead."}
        result = read_move(chess.Board(fen), seen)
        if "move" in result:
            self._add({"type": "say", "text": f"The camera read your move: {result['san']}."})
            return {"ok": self.answer(qid, result["move"]), **result}
        self._add({"type": "camera_problem", "text": f"{result['problem']} {result['fix']}".strip(),
                   "spoken": f"{result['problem']} {result['fix']}".strip()})
        return {"ok": False, **{k: result[k] for k in ("problem", "fix")}}

    def voice_token(self) -> dict | None:
        """A listen-only pass to the robot's LiveKit room, for the page to play its voice. Only with --voice livekit,
        and only to this computer's own page (the same local-only check as every other route)."""
        if getattr(self.args, "voice", "none") != "livekit":
            return None
        import os
        import secrets

        from dreamchess.livekit_voice import DEFAULT_ROOM, load_env
        from livekit import api

        load_env()
        room = getattr(self.args, "voice_room", None) or os.environ.get("CHESS_ROOM", DEFAULT_ROOM)
        token = (api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
                 .with_identity(f"page-{secrets.token_hex(3)}").with_name("Dream Chess page")
                 .with_grants(api.VideoGrants(room_join=True, room=room, can_publish=False, can_subscribe=True))
                 .to_jwt())
        return {"url": os.environ["LIVEKIT_URL"], "token": token, "room": room}

    def brain(self) -> dict:
        """Dream-1 at work: what the agent last asked it, plus the station's own status on the real robot."""
        with self.lock:
            dream = dict(self.dream)
        out = {"model": self.model(), "simulated": bool(self.args.sim), **dream}
        if not self.args.sim:
            out["station"] = station_telemetry(self.args.station)
            out["warm"] = self.warm_status()
            out["hand"] = self.hand()
        out["services"] = self.services(out)
        return out

    def services(self, brain: dict) -> list[dict]:
        """Everything the robot depends on, and whether it's connected: the landing page's status panel.
        `state` is ok, warn, bad, off or placeholder (a part the system still needs, see docs/NEXT_STEPS.md)."""
        sim = bool(self.args.sim)
        out = []
        warm = brain.get("warm") or {}
        if sim:
            out.append({"name": "Dream-1 · DreamAPI", "detail": "Simulated for rehearsal", "state": "warn"})
        else:
            w = warm.get("state")
            out.append({"name": "Dream-1 · DreamAPI", "detail": {"ready": "Running and answering",
                        "warming": "Starting the model", "error": warm.get("message") or "Couldn't start the model",
                        "off": warm.get("message") or "Not started"}.get(w, "Checking..."),
                        "state": {"ready": "ok", "warming": "warn", "error": "bad"}.get(w, "off")})
            station = brain.get("station") or {}
            out.append({"name": "Robot station", "detail": "Offline" if not station.get("reachable")
                        else f"Reachable · {station.get('state') or 'unknown'}",
                        "state": "ok" if station.get("reachable") else "bad"})
        engine = self.engine_name
        out.append({"name": "Chess engine", "detail": "Stockfish (C++), run from Python" if engine == "stockfish"
                    else "Built-in fallback: Stockfish isn't installed" if engine else "Starting...",
                    "state": "ok" if engine == "stockfish" else "warn" if engine else "off"})
        from dreamchess.agents import make_agents

        stand_in = {"watcher": "the checker's verdict", "prompt writer": "template sentences"}
        for agent in make_agents(getattr(self.args, "agents", "rules"), getattr(self.args, "agent_model", None)).status():
            if agent["placeholder"] and agent["connected"]:
                detail, state = agent["name"], "ok"
            elif agent["placeholder"]:
                detail, state = f"LLM not connected: {stand_in[agent['role']]}", "placeholder"
            else:
                detail, state = f"Stand-in: {stand_in[agent['role']]}", "placeholder"
            out.append({"name": f"{agent['role'].capitalize()} agent", "detail": detail, "state": state})
        if sim:
            out.append({"name": "Top camera checker", "detail": "Simulated camera", "state": "warn"})
        elif self.webcam is None:
            out.append({"name": "Top camera checker", "detail": "Off: a person checks each step", "state": "off"})
        else:
            calibrated = self.webcam.status()["calibrated"]
            out.append({"name": "Top camera checker", "detail": "Calibrated" if calibrated
                        else "Not calibrated: a person checks each step", "state": "ok" if calibrated else "warn"})
        voice = getattr(self.args, "voice", "none")
        out.append({"name": "Voice · LiveKit", "detail": {"livekit": "Speaking in a LiveKit room",
                    "print": "Text only, no audio"}.get(voice, "Off"),
                    "state": {"livekit": "ok", "print": "warn"}.get(voice, "off")})
        return out

    def stats(self) -> dict:
        """This game's numbers and every instruction sent to Dream-1, for the stats page (not the landing page)."""
        with self.lock:
            game = [e for e in self.events if e.get("game") == self.game_number]
            summary = dict(self.summary) if self.summary else None
        agent_lines = [e.get("text", "") for e in game if e.get("type") == "agent"]
        prompts = []
        for e in game:
            text = e.get("text", "") if e.get("type") == "agent" else ""
            found = re.match(r"HOW\s+\[(\d+)/(\d+)\] (.*)", text)
            if found:
                prompts.append({"attempt": f"{found.group(1)}/{found.group(2)}", "prompt": found.group(3),
                                "kind": "retry" if found.group(1) != "1" else "first"})
            elif e.get("type") == "say" and "SUPERVISE" in e.get("text", ""):
                prompts.append({"attempt": "mid-run", "prompt": e["text"].split("SUPERVISE", 1)[1].strip(),
                                "kind": "supervisor"})
        moves = sum(1 for t in agent_lines if t.startswith("WHERE"))
        if summary and summary.get("robot_moves") is not None:
            moves = summary["robot_moves"]
        elif self.running and moves and not any(t.startswith("SEEN") for t in agent_lines[
                max(i for i, t in enumerate(agent_lines) if t.startswith("WHERE")):]):
            moves -= 1  # the robot is still making its latest move
        return {
            "game": self.game_number, "running": self.running, "summary": summary,
            "robot_moves": moves,
            "dream_calls": sum(1 for e in game if e.get("type") == "attempt"),
            "self_corrections": sum(1 for t in agent_lines if t.startswith("CHECK")),
            "human_rescues": sum(1 for t in agent_lines if t.startswith("HELP")),
            "handoffs": sum(1 for t in agent_lines if t.startswith("HANDOFF")),
            "prompts": prompts,
            "log": [{"type": e.get("type"), "text": e.get("text", "")} for e in game if e.get("text")],
        }

    def state(self, since: int) -> dict:
        with self.lock:
            reset = since > len(self.events)  # the page is from before a restart
            if reset:
                since = 0
            events = self.events[since:]
            pending = dict(self.pending) if self.pending else None
            running, summary, snapshot = self.running, self.summary, dict(self.snapshot)
        return {
            "boot": self.boot,
            "reset": reset,
            "events": events,
            "next": since + len(events),
            "pending": pending,
            "running": running,
            "summary": summary,
            "fen": snapshot.get("fen"),
            "last_move": snapshot.get("last_move"),
            "turn": snapshot.get("turn"),
            "check": snapshot.get("check", False),
            "physical": snapshot.get("physical", {}),
            "mismatch": snapshot.get("mismatch", []),
            "setup": snapshot.get("setup", False),
            "mode": "simulation" if self.args.sim else "live robot",
            "checkpoint": self.model(),
            "auto_opponent": bool(getattr(self.args, "auto_opponent", False)),
            "voice": getattr(self.args, "voice", "none"),
            "public": bool(getattr(self.args, "public", False)),
            "webcam": self.webcam.status() if self.webcam is not None else None,
        }


def make_server(session: Session, port: int, public: bool = False) -> ThreadingHTTPServer:
    """The app's web server. Normally it answers this computer only. `public` is the demo site: open to every visitor,
    but only ever for a simulation, and visitors can watch and restart, nothing else."""
    if public and not session.args.sim:
        raise ValueError("the public site only runs the simulation: it must never reach a real robot")
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value, code: int = 200) -> None:
            self._send(code, json.dumps(value).encode(), "application/json")

        def _local(self) -> bool:
            # A page from another site can point its own name at this computer (DNS rebinding) and then talk to the
            # app as if it were local. Its requests still carry its own name in Host, so only accept this computer's.
            host = (self.headers.get("Host") or "").lower()
            name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
            return name in LOCAL_HOSTS

        def do_GET(self) -> None:
            if not public and not self._local():
                return self._send(403, b"forbidden", "text/plain")
            path, _, query = self.path.partition("?")
            if path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/state":
                found = re.search(r"since=(\d+)", query)
                self._json(session.state(int(found.group(1)) if found else 0))
            elif path == "/vision":
                self._json(session.vision())
            elif path == "/brain":
                self._json(session.brain())
            elif path == "/stats":
                self._send(200, STATS_PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/stats.json":
                self._json(session.stats())
            elif path == "/voice/token":
                try:
                    found = session.voice_token()
                except (ImportError, KeyError) as err:
                    return self._send(503, f"the voice isn't set up here: {err}".encode(), "text/plain")
                if found is None:
                    return self._send(404, b"the voice is off: start the app with --voice livekit", "text/plain")
                self._json(found)
            elif (found := re.fullmatch(r"/camera/(\w+)\.jpg", path)):
                image = session.camera(found.group(1), "grid=1" in query)
                if image is None:
                    return self._send(404, b"no frame yet", "text/plain")
                self._send(200, image, "image/jpeg")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            # Only the app's own page: this computer's address, plus a custom header another site's page can't send
            # without a preflight this server never answers. Otherwise any page could answer the e-stop question.
            if not (public or self._local()) or self.headers.get("X-Dream-Chess") != "1":
                return self._send(403, b"forbidden", "text/plain")
            if public and self.path != "/restart":  # visitors watch; the only thing they can do is start over
                return self._send(403, b"the public demo only watches and restarts", "text/plain")
            try:
                length = min(int(self.headers.get("Content-Length") or 0), 4096)
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("body must be a JSON object")
                if self.path == "/answer":
                    ok = session.answer(int(body.get("id", -1)), str(body.get("text", ""))[:200])
                    return self._json({"ok": ok}, 200 if ok else 409)
                if self.path == "/new":
                    return self._json({"ok": session.start()})
                if self.path == "/restart":
                    return self._json({"ok": session.restart()})
                if self.path == "/stop":
                    return self._json({"ok": True, "message": session.stop_robot()})
                if self.path == "/webcam/move":
                    return self._json(session.camera_move(int(body.get("id", -1))))
                if self.path.startswith("/webcam/"):
                    return self._json(session.webcam_action(self.path.rsplit("/", 1)[1], body))
            except (ValueError, TypeError) as err:
                return self._send(400, f"bad request: {err}".encode(), "text/plain")
            self._send(404, b"not found", "text/plain")

        def log_message(self, *args) -> None:
            pass

    return ThreadingHTTPServer(("0.0.0.0" if public else "127.0.0.1", port), Handler)


def add_game_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sim", action="store_true", help="a simulated robot: nothing moves")
    parser.add_argument("--auto-opponent", action="store_true", help="with --sim, a random opponent plays White")
    parser.add_argument("--pace", type=float, default=1.0, help="with --sim, seconds between steps, to watch it")
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    parser.add_argument("--checkpoint", help="the model to run, such as a fine-tuned ID; default: the base model")
    parser.add_argument("--no-supervise", dest="supervise", action="store_false",
                        help="don't watch moves as they happen (or re-prompt Dream-1 mid-run)")
    parser.add_argument("--prompt-style", choices=("squares", "both", "spatial"), default="both",
                        help="name squares as in our demos (b8), describe where they are from the robot's seat, or both")
    parser.add_argument("--board-fit", default="~/.dream-chess/board_fit.json",
                        help="where tools/fit_board.py saved the board's squares in each arm's frame")
    parser.add_argument("--dream-python", default="~/dream-env/bin/python",
                        help="the Dream library's Python, which starts the model on DreamAPI before the first move")
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="time limit per Dream-1 attempt, not counting time the arms wait on DreamAPI")
    parser.add_argument("--stockfish", help="path to Stockfish (or set STOCKFISH_PATH)")
    parser.add_argument("--station", default="http://127.0.0.1:8091")
    parser.add_argument("--sim-success", type=float, default=0.5, help="with --sim, how often the robot succeeds")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=None)
    parser.add_argument("--script", type=int, choices=(1, 2, 3),
                        help="replay collection game 1, 2 or 3: the robot plays the moves Dream-1 was trained on")
    parser.add_argument("--log-dir", default=str(run_robot.GAMES))
    parser.add_argument("--no-top-camera", dest="top_camera", action="store_false",
                        help="don't take top-camera pictures: a person checks every step and types every move")
    parser.add_argument("--top-calibration", default="~/.dream-chess/top-calibration.npz",
                        help="where the top camera's board calibration is saved, so it survives a restart")
    parser.add_argument("--frames", default="~/.dreamlabs/last-run_inputs",
                        help="where the station saves the camera frames it hands Dream-1")
    run_robot.add_voice_args(parser)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_game_args(parser)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8800)))
    parser.add_argument("--public", action="store_true",
                        help="the public demo site: a simulated robot against a random opponent, open to every "
                             "visitor (watch and restart only), starting a new game after each one ends")
    args = parser.parse_args()
    args.label = None
    if args.public:  # never the real robot, never anything a visitor could set off
        # The voice is "print": its lines and moving bars show on the page, with no LiveKit room or keys.
        args.sim, args.auto_opponent, args.voice, args.top_camera, args.supervise = True, True, "print", False, False

    session = Session(args)
    server = make_server(session, args.port, public=args.public)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    session.start()
    where = f"http://0.0.0.0:{args.port} (public demo)" if args.public else f"http://127.0.0.1:{args.port}"
    print(f"Dream Chess: {where}  ({'simulation' if args.sim else 'LIVE ROBOT'}; Ctrl+C to stop)", flush=True)
    ended_at = None
    try:
        while True:
            time.sleep(0.5)
            if args.public and not session.running:  # the site always has a game on: a new one shortly after each
                ended_at = ended_at or time.monotonic()
                if time.monotonic() - ended_at > 10:
                    session.start()
                    ended_at = None
    except KeyboardInterrupt:
        print("\nStopping: the arms first, then the game.")
        session.shutdown()
        server.shutdown()
        print("Stopped.")


if __name__ == "__main__":
    main()
