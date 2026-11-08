"""HOW, through the YAM station: send each prompt to the station's local control API.

The station (`dreamcli robot up`) already runs Dream-1's control loop with its own safety
limits, such as a cap on hand speed, and its own home move. So rather than driving the arms,
this asks the station to start a run with the step's prompt and a step limit, then waits for it
to finish. A run that uses up its steps ends on its own and sends the arms home.

The API only listens on the NUC itself (127.0.0.1:8091), so this runs there.
Written from the station's source (robots 1.0.37). It has only run against a fake station.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Callable

from .core import MoverError, Step

DEFAULT_URL = "http://127.0.0.1:8091"
# The station keeps whichever model it ran last, so "the base model" has to be asked for by name.
BASE_MODEL = "dreamlabs/dream-1-beta"
CONTROL_HZ = 20  # the station's loop rate on a YAM: one step is one command to the arms
BUSY = {"starting", "running", "stopping", "homing", "resetting"}
# A model that isn't running on DreamAPI yet takes a while to start: "a cold one can take 15 minutes", the station
# says, and it waits that out itself (up to 30) with the arms still. So allow this long for Dream-1's first answer.
FIRST_ANSWER_WAIT_S = 20 * 60
WAIT_NOTE_EVERY_S = 30.0


class StationError(MoverError):
    """The station refused a request or reported an error."""


class StationClient:
    def __init__(self, url: str = DEFAULT_URL, timeout_s: float = 70.0):
        # The station waits up to 60 s on a stop or home before answering, so allow a bit more.
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            self.url + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as err:
            try:
                message = json.loads(err.read()).get("error", "")
            except ValueError:
                message = ""
            raise StationError(f"station refused {method} {path} ({err.code}): {message}") from err
        except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
            raise StationError(f"can't reach the station at {self.url}: {err}") from err

    def status(self) -> dict:
        return self._request("GET", "/status")

    def start(self, prompt: str, steps: int, checkpoint: str | None = None) -> dict:
        """Start a run. A checkpoint, such as a fine-tuned model's ID, stays in use for later runs too."""
        body = {"prompt": prompt, "steps": steps}
        if checkpoint:
            body["checkpoint"] = checkpoint
        return self._request("POST", "/start", body)

    def stop(self) -> dict:
        return self._request("POST", "/stop", {})

    def home(self) -> dict:
        return self._request("POST", "/home", {})

    def reset(self) -> dict:
        return self._request("POST", "/reset", {})


class StationMover:
    """A Mover for the orchestrator that runs each step as one station run."""

    def __init__(
        self,
        client: StationClient | None = None,
        attempt_s: float = 30.0,
        checkpoint: str | None = None,
        poll_s: float = 0.5,
        grace_s: float = 20.0,
        first_answer_s: float = FIRST_ANSWER_WAIT_S,
        restarts: int = 3,
        log: Callable[[str], None] = print,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.client = client or StationClient()
        self.steps = max(1, round(attempt_s * CONTROL_HZ))
        self.attempt_s = attempt_s
        self.checkpoint = checkpoint  # None keeps the station's current model
        self.poll_s = poll_s
        self.grace_s = grace_s  # time to start up, home, and answer, beyond the run itself
        self.first_answer_s = first_answer_s  # how long Dream-1 may take to send its first action
        self.restarts = restarts  # runs to start again because the model answered too late for the arms to move
        self.log = log
        self.clock = clock
        self.sleep = sleep

    def _wait_until_settled(self, limit_s: float) -> dict:
        """Wait until the station isn't starting, running, stopping, or homing."""
        deadline = self.clock() + limit_s
        while True:
            status = self.client.status()
            if status.get("state") not in BUSY:
                return status
            if self.clock() >= deadline:
                return status
            self.sleep(self.poll_s)

    def _check(self, status: dict) -> None:
        if status.get("state") == "error":
            raise StationError(f"station reported an error: {status.get('last_error') or 'no details'}")

    def _run(self, step: Step) -> tuple[dict, bool]:
        """One station run of the step. Returns the settled status and whether Dream-1 sent any actions, which is
        whether the arms could have moved at all.

        The time limit counts from Dream-1's first action, not from the start: until then the model may still be
        starting on DreamAPI, the station waits with the arms still, and there is nothing to stop."""
        self.client.start(step.prompt, self.steps, self.checkpoint)
        started = self.clock()
        first_action = None
        reports_chunks = False
        noted = started
        while True:
            status = self.client.status()
            reports_chunks = reports_chunks or "chunks" in status
            if (status.get("chunks") or 0) > 0 and first_action is None:
                first_action = self.clock()
            if status.get("state") not in BUSY:
                break
            now = self.clock()
            waiting = reports_chunks and first_action is None
            limit = (started + self.first_answer_s) if waiting else (first_action or started) + self.attempt_s + self.grace_s
            if now >= limit:
                self.log("      Dream-1 never answered: stopping the run" if waiting
                         else "      the station run went past its time limit: stopping it")
                self.client.stop()
                status = self._wait_until_settled(self.grace_s)
                break
            if waiting and now - noted >= WAIT_NOTE_EVERY_S:
                noted = now
                self.log(f"      waiting for Dream-1's first answer ({now - started:.0f} s): the model is starting "
                         "on DreamAPI, and the arms wait until it answers")
            self.sleep(self.poll_s)
        moved = first_action is not None or not reports_chunks or (status.get("chunks") or 0) > 0
        return status, moved

    def execute(self, step: Step) -> None:
        before = self.client.status()
        if before.get("state") != "idle":
            raise StationError(f"station isn't ready: it's {before.get('state')}")
        dead = [name for name, alive in (before.get("arms") or {}).items() if not alive]
        if dead:
            raise StationError(f"station reports a dead arm driver: {', '.join(dead)}")

        after, moved = self._run(step)
        for _ in range(self.restarts):
            if moved or (after.get("state") == "error" and "ready" not in str(after.get("last_error") or "")):
                break  # it moved, or something else went wrong: _check() below says what
            # No actions came, so the arms never moved and the board is as it was. That happens when the model
            # answered only after a long start: the station then refuses to move on its own ("start the run again,
            # and it moves at once"). So start the same step again rather than ask anyone whether it landed.
            self.log("      Dream-1 is ready now; the arms didn't move yet, so running the step again")
            if after.get("state") == "error":
                self.client.reset()
                after = self._wait_until_settled(self.grace_s * 3)
                self._check(after)
            after, moved = self._run(step)
        self._check(after)
        if after.get("state") in BUSY:
            raise StationError(f"station still {after.get('state')} after stopping")
        if not moved:
            raise StationError("Dream-1 sent no actions, so the arms never moved")

    def go_home(self) -> None:
        """Clear the top camera's view. A run that used up its steps already went home."""
        settled = self._wait_until_settled(self.grace_s)
        self._check(settled)
        try:
            self.client.home()
        except StationError:
            # It may have started its own home move a moment ago, so let that finish and check.
            settled = self._wait_until_settled(self.grace_s)
            self._check(settled)
            if settled.get("state") != "idle":
                raise
