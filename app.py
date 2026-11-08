"""Dream Chess: the app. Watch the robot think through every step of a game, and play it from the page.

    python app.py --sim --auto-opponent      # dummy data: a simulated robot playing a random opponent
    python app.py --sim                      # simulated robot; you play White from the page
    python app.py                            # the real robot, on the NUC (a person at the e-stop)

Open http://127.0.0.1:8800 (on the NUC, from your laptop: ssh -L 8800:127.0.0.1:8800 dream-nuc).
The game itself is run_robot.play(): this only gives it a page instead of a terminal, so the same code runs both.
"""
from __future__ import annotations

import argparse
import json
import re
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
PACED = ("agent", "attempt", "attempt_result", "opponent_move", "help")
LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]")


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
        # What the agent last asked Dream-1 to do, from the game's own events (both modes).
        self.dream: dict = {"state": "idle", "prompt": None, "arm": None, "attempt": None, "calls": 0,
                            "last_seconds": None, "landed": None}

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

    def start(self) -> bool:
        with self.lock:
            if self.running:
                return False
            self.running, self.quitting, self.summary = True, False, None
            self.board, self.physical, self.snapshot = None, None, {}
            self.game_number += 1
            self.dream.update(state="idle", prompt=None, arm=None, attempt=None, calls=0, last_seconds=None,
                              landed=None)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return True

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
        if not self.args.sim:
            ages = {role: found[1] for role in ROLES if (found := self.frames.latest(role))}
            return {"source": "person", "ready": True, "detected": physical, "agrees": True,
                    "frames": {role: round(time.time() - mtime, 1) for role, mtime in ages.items()}}
        view = self.eyes().view(snapshot.get("fen"), physical)
        return {"source": "camera", "ready": view["ready"], "error": view["error"],
                "detected": view.get("detected", {}), "agrees": view.get("agrees")}

    def brain(self) -> dict:
        """Dream-1 at work: what the agent last asked it, plus the station's own status on the real robot."""
        with self.lock:
            dream = dict(self.dream)
        model = self.args.checkpoint or ("Dream-1 (simulated)" if self.args.sim else "Dream-1 base")
        out = {"model": model, "simulated": bool(self.args.sim), **dream}
        if not self.args.sim:
            out["station"] = station_telemetry(self.args.station)
        return out

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
            "mode": "simulation" if self.args.sim else "live robot",
            "checkpoint": self.args.checkpoint or ("simulated robot" if self.args.sim else "station default"),
            "auto_opponent": bool(getattr(self.args, "auto_opponent", False)),
        }


def make_server(session: Session, port: int) -> ThreadingHTTPServer:
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
            if not self._local():
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
            if not self._local() or self.headers.get("X-Dream-Chess") != "1":
                return self._send(403, b"forbidden", "text/plain")
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
                if self.path == "/stop":
                    return self._json({"ok": True, "message": session.stop_robot()})
            except (ValueError, TypeError) as err:
                return self._send(400, f"bad request: {err}".encode(), "text/plain")
            self._send(404, b"not found", "text/plain")

        def log_message(self, *args) -> None:
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def add_game_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sim", action="store_true", help="a simulated robot: nothing moves")
    parser.add_argument("--auto-opponent", action="store_true", help="with --sim, a random opponent plays White")
    parser.add_argument("--pace", type=float, default=1.0, help="with --sim, seconds between steps, to watch it")
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    parser.add_argument("--checkpoint", help="the model to run, such as a fine-tuned ID; default keeps the station's")
    parser.add_argument("--seconds", type=float, default=30.0, help="time limit per Dream-1 attempt")
    parser.add_argument("--stockfish", help="path to Stockfish (or set STOCKFISH_PATH)")
    parser.add_argument("--station", default="http://127.0.0.1:8091")
    parser.add_argument("--sim-success", type=float, default=0.5, help="with --sim, how often the robot succeeds")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=None)
    parser.add_argument("--log-dir", default=str(run_robot.GAMES))
    parser.add_argument("--frames", default="~/.dreamlabs/last-run_inputs",
                        help="where the station saves the camera frames it hands Dream-1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_game_args(parser)
    parser.add_argument("--port", type=int, default=8800)
    args = parser.parse_args()
    args.label = None

    session = Session(args)
    server = make_server(session, args.port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    session.start()
    print(f"Dream Chess: http://127.0.0.1:{args.port}  ({'simulation' if args.sim else 'LIVE ROBOT'}; Ctrl+C to stop)")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping: the arms first, then the game.")
        session.shutdown()
        server.shutdown()
        print("Stopped.")


if __name__ == "__main__":
    main()
