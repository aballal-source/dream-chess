"""Watch a move while Dream-1 makes it, and re-prompt it the moment it goes wrong, without stopping the run.

The hand tracker (tools/hand_watch.py) reports, about once a second, which square each hand is over, how high it is,
and how far its gripper is closed. From that, the supervisor sees the two moments that decide a move:

- the squeeze: the fingers close at table height. Over the right square, the piece is in hand. Over an empty square
  (grabbing air) or the wrong piece, it says so and asks again.
- the release: the fingers open. On the target square, the step is done and the run can end. Anywhere else, the piece
  is now on that square, so it sends the correction from there, worded exactly like our training demos ("pick up
  the brown rook on square c3 and place it upright in the center of square b1").

Each prompt goes to the station's /prompt route, which changes the instruction mid-run: the arms keep moving. The
check after the run (a person, or later the camera) still decides whether the board is right.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import chess

from .core import TRAY, Occupancy, Step
from .where import make_step

FIRM_GRIP = 0.22     # fingers squeezed shut on a piece; 1.0 is fully open
REOPENED = 0.30      # open again after a squeeze: the piece has been let go
TABLE_Z_M = 0.10     # a squeeze higher than this is in the air (operators pre-close on the way down)
REACH_LIMIT_S = 40.0  # no squeeze this long after a prompt: ask again


@dataclass
class Action:
    kind: str  # "prompt" (send `step.prompt` now) or "done" (the piece is on its target: the run can end)
    step: Step | None = None
    reason: str = ""


@dataclass
class Supervisor:
    step: Step  # what this run was asked to do
    pieces: dict[str, tuple[str, str]]  # what's on each square before the step, (color code, name), from the game
    robot_color: chess.Color
    goal: Occupancy
    max_prompts: int = 4  # corrections and repeats per run, before leaving it to the check after
    log: Callable[[str], None] = lambda _: None
    current: Step = field(init=False)
    phase: str = field(init=False, default="reach")  # reach -> holding -> done; "returning" for a wrong piece
    prompts: int = field(init=False, default=0)
    asked_at: float = field(init=False, default=0.0)
    gripped: str | None = field(init=False, default=None)  # the square the fingers closed on
    low_square: str | None = field(init=False, default=None)  # the last square the hand was over near the table
    wrong_piece_home: str | None = field(init=False, default=None)
    then: Step | None = field(init=False, default=None)  # the move to go back to once a wrong piece is returned
    events: list[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.current = self.step

    def _say(self, text: str) -> None:
        self.events.append(text)
        self.log(f"      SUPERVISE {text}")

    def _ask(self, step: Step, t: float, why: str) -> Action | None:
        if self.prompts >= self.max_prompts:
            self._say(f"{why}, but that's {self.max_prompts} prompts this run: leaving it to the check")
            return None
        self.prompts += 1
        self.current, self.asked_at, self.phase = step, t, "reach"
        self._say(f"{why}: asking again: {step.prompt}")
        return Action("prompt", step, why)

    def feed(self, sample: dict) -> Action | None:
        """One hand-tracker sample, {"t", "right": {square, z, grip}, "left": {...}}. Returns what to do, if anything."""
        if self.phase == "done":
            return None
        hand = sample.get(self.current.arm)
        if not hand:
            return None
        t, square, z, grip = sample.get("t", 0.0), hand.get("square"), hand.get("z", 1.0), hand.get("grip", 1.0)
        low = z < TABLE_Z_M
        if low:
            self.low_square = square

        if self.phase == "reach":
            if grip < FIRM_GRIP and low:
                self.gripped = square
                if square == self.current.source:
                    self.phase = "holding"
                    self._say(f"gripped on {square}, the right square")
                    return None
                if square is not None and square in self.pieces:
                    # The wrong piece: let it go, then bring it back to where it was.
                    self.phase, self.wrong_piece_home = "returning", square
                    self._say(f"gripped the wrong piece, the one on {square}, not {self.current.source}")
                    return None
                self.phase = "air"
                self._say(f"closed on {square or 'the edge of the board'}, where there's no piece to take")
                return None
            if t - self.asked_at > REACH_LIMIT_S:
                return self._ask(self.current, t, f"no grip in {REACH_LIMIT_S:.0f} s")
            return None

        if self.phase == "air":
            if grip > REOPENED:  # it let go of nothing: ask for the same step again
                return self._ask(self.current, t, f"missed the piece on {self.current.source}")
            return None

        if grip <= REOPENED:
            return None  # still holding
        released = self.low_square
        if self.phase == "returning":
            home = self.wrong_piece_home
            self.wrong_piece_home = None
            if released == home or released is None:
                return self._ask(self.current, t, f"it put the wrong piece back down on {home}")
            color, name = self.pieces[home]
            back = make_step(color, name, released, home, self.robot_color, self.goal, fix=True)
            self.then = self.current
            return self._ask(back, t, f"the {name} from {home} landed on {released}: putting it back first")

        # Holding the right piece, and it just let go.
        if self.current.target == TRAY:
            if released is None:
                self.phase = "done"
                self._say("let go off the board, in the tray")
                return Action("done")
        elif released == self.current.target:
            if self.then is not None:  # that was putting a wrong piece back: now the real move
                step, self.then = self.then, None
                return self._ask(step, t, f"the wrong piece is back on {released}")
            self.phase = "done"
            self._say(f"let go on {released}, the target")
            return Action("done")
        if released is None:
            self._say("let go off the board: leaving it to the check")
            self.phase = "done"
            return None
        fix = make_step(self.current.color, self.current.piece_name, released, self.current.target,
                        self.robot_color, self.goal, fix=True)
        return self._ask(fix, t, f"let go on {released}, not {self.current.target}")
