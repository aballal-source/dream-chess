"""Checks the station client against a fake station that follows the real one's rules.

The fake is built from the station's source (robots 1.0.37): a start is refused unless idle, a run
with a step limit ends on its own and then homes, and home is refused while a run exists.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from dreamchess.core import Step
from dreamchess.station_mover import CONTROL_HZ, StationClient, StationError, StationMover

STEP = Step("w", "knight", "g1", "f3", "right", "pick up the knight on g1", {})


class FakeStation:
    """A station whose run takes `run_s` seconds of wall time, whatever its step count.

    `warm_s` is how long the model takes to start on DreamAPI before its first action. Like the real station, a
    model that answers only after `move_after_s` doesn't move: the run ends with no actions ("start the run again,
    and it moves at once"), and the next run moves at once. `never_answers` is a model that never starts."""

    def __init__(self, run_s=0.15, home_s=0.1, fail_run=False, never_ends=False, warm_s=0.0, move_after_s=0.3,
                 never_answers=False):
        self.run_s, self.home_s = run_s, home_s
        self.fail_run, self.never_ends = fail_run, never_ends
        self.warm_s, self.move_after_s, self.never_answers = warm_s, move_after_s, never_answers
        self.chunks = 0
        self.state = "idle"
        self.last_error = None
        self.arms = {"left": True, "right": True}
        self.run = 0
        self.requests = []
        self.lock = threading.Lock()
        self._cancel = threading.Event()

    def status(self):
        with self.lock:
            return {"state": self.state, "run": self.run, "last_error": self.last_error, "arms": dict(self.arms),
                    "chunks": self.chunks}

    def _later(self, seconds, action):
        def go():
            if not self._cancel.wait(seconds):
                action()
        threading.Thread(target=go, daemon=True).start()

    def _set(self, state, error=None):
        with self.lock:
            self.state, self.last_error = state, error

    def start(self, body):
        with self.lock:
            if self.state not in ("idle", "error"):
                return 409, {"error": f"start refused: station is {self.state}"}
            if not isinstance(body.get("steps"), int) or body["steps"] < 1:
                return 400, {"error": "steps must be a whole number of at least 1"}
            self.run += 1
            self.state = "starting"
            self.chunks = 0
        self._cancel.clear()
        self._later(0.02, lambda: self._set("running"))
        if self.never_answers:
            return 200, self.status()  # runs with no actions until stopped
        if self.warm_s > self.move_after_s:
            # The model answers too late to move: the run ends with no actions, and it's warm from now on.
            warm, self.warm_s = self.warm_s, 0.0
            self._later(warm, lambda: self._set("idle", "the policy is ready: start the run again, and it moves at once"))
            return 200, self.status()
        self._later(0.04 + self.warm_s, self._first_action)  # once it's running
        if self.fail_run:
            self._later(self.run_s, lambda: self._set("error", "the right arm faulted"))
        elif not self.never_ends:
            # Used up its steps: idle, then the station's own home move.
            self._later(self.run_s, lambda: (self._set("idle"), self._set("homing"),
                                             self._later(self.home_s, lambda: self._set("idle"))))
        return 200, self.status()

    def _first_action(self):
        with self.lock:
            if self.state == "running":
                self.chunks = 1

    def stop(self):
        self._cancel.set()
        self._set("idle")
        return 200, self.status()

    def home(self):
        with self.lock:
            if self.state != "idle":
                return 409, {"error": f"home refused: station is {self.state}"}
            self.state = "homing"
        time.sleep(self.home_s)  # the real station answers once the arms are home
        self._set("idle")
        return 200, self.status()


@pytest.fixture
def station():
    fake = FakeStation()

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, payload):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            fake.requests.append(("GET", self.path, None))
            self._reply(200, fake.status()) if self.path == "/status" else self._reply(404, {"error": "no route"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            fake.requests.append(("POST", self.path, body))
            routes = {"/start": lambda: fake.start(body), "/stop": fake.stop, "/home": fake.home}
            self._reply(*routes[self.path]()) if self.path in routes else self._reply(404, {"error": "no route"})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    fake.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield fake
    fake._cancel.set()
    server.shutdown()


def mover(station, **args):
    defaults = {"attempt_s": 0.2, "poll_s": 0.02, "grace_s": 1.0, "log": lambda _: None}
    return StationMover(StationClient(station.url, timeout_s=5), **{**defaults, **args})


def posts(station, path):
    return [body for method, p, body in station.requests if method == "POST" and p == path]


def test_sends_the_prompt_with_a_step_limit(station):
    m = mover(station, attempt_s=30)
    station.run_s = 0.1
    m.execute(STEP)
    assert posts(station, "/start") == [{"prompt": "pick up the knight on g1", "steps": 30 * CONTROL_HZ}]


def test_waits_for_the_run_and_its_home_move(station):
    mover(station).execute(STEP)
    assert station.status()["state"] == "idle"
    assert station.run == 1


def test_go_home_after_a_run_ends_idle(station):
    m = mover(station)
    m.execute(STEP)
    m.go_home()
    assert station.status()["state"] == "idle"


def test_go_home_waits_out_a_home_move_already_in_progress(station):
    station.home_s = 0.3
    m = mover(station)
    station.start({"prompt": "x", "steps": 1})
    m.go_home()  # the station's own home move is running, so this must wait rather than fail
    assert station.status()["state"] == "idle"


def test_refuses_to_start_unless_the_station_is_idle(station):
    station.state = "running"
    with pytest.raises(StationError, match="isn't ready"):
        mover(station).execute(STEP)
    assert posts(station, "/start") == []


def test_refuses_to_start_with_a_dead_arm(station):
    station.arms["left"] = False
    with pytest.raises(StationError, match="dead arm"):
        mover(station).execute(STEP)
    assert posts(station, "/start") == []


def test_station_error_becomes_a_mover_error(station):
    station.fail_run = True
    with pytest.raises(StationError, match="the right arm faulted"):
        mover(station).execute(STEP)


def test_stops_a_run_that_overstays_its_limit(station):
    station.never_ends = True
    m = mover(station, attempt_s=0.1, grace_s=0.2)
    m.execute(STEP)
    assert len(posts(station, "/stop")) == 1
    assert station.status()["state"] == "idle"


def test_unreachable_station_is_a_clear_error():
    client = StationClient("http://127.0.0.1:9", timeout_s=1)
    with pytest.raises(StationError, match="can't reach the station"):
        StationMover(client, log=lambda _: None).execute(STEP)


def test_refusal_message_is_passed_on(station):
    client = StationClient(station.url)
    station.state = "homing"
    with pytest.raises(StationError, match="home refused: station is homing"):
        client.home()


def test_works_as_the_orchestrators_mover(station):
    import chess

    from dreamchess.core import occupancy
    from dreamchess.engine import FallbackEngine
    from dreamchess.mock import MockHuman, RandomOpponent, SimWorld
    from dreamchess.orchestrator import Orchestrator

    board = chess.Board("7k/8/8/8/8/8/R7/1R4K1 w - - 0 1")
    world = SimWorld(board)

    class TableCamera:
        """The fake station doesn't move pieces, so make each run land the piece, as a success would."""

        def __init__(self):
            self.pending = None

        def observe(self):
            if self.pending is not None:
                world.occ = dict(self.pending)
                self.pending = None
            return dict(world.occ)

    camera = TableCamera()
    m = mover(station)
    original = m.execute

    def execute(step):
        original(step)
        camera.pending = step.expected_after

    m.execute = execute
    agent = Orchestrator(board, chess.WHITE, FallbackEngine(), m, camera, MockHuman(world),
                         RandomOpponent(world), log=lambda _: None)
    agent.play_robot_turn()
    assert world.occ == occupancy(board)
    assert station.run == 1


# --- A model that is still starting on DreamAPI ---

def test_a_long_warm_up_is_waited_out_then_the_step_runs_again(station):
    station.warm_s = 0.6  # longer than the attempt limit, and too late for the station to move
    notes = []
    m = mover(station, attempt_s=0.1, grace_s=0.2, log=notes.append)
    m.execute(STEP)
    assert posts(station, "/stop") == []  # never cut off while the model was starting
    assert len(posts(station, "/start")) == 2  # the first run moved nothing, so it ran again, by itself
    assert station.chunks == 1
    assert any("running the step again" in n for n in notes)


def test_a_model_that_never_answers_is_stopped_and_reported(station):
    station.never_answers = True
    m = mover(station, attempt_s=0.1, grace_s=0.2, first_answer_s=0.4, restarts=1)
    with pytest.raises(StationError, match="never moved"):
        m.execute(STEP)
    assert len(posts(station, "/stop")) == 2  # each run stopped once its wait for an answer ran out
