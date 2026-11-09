"""The agent loop: observe, decide, act, verify, recover.

The board only advances after the camera confirms the physical move, so the program
can never believe a move happened when it didn't. Every failure uses the same recovery
logic: compare the camera against the goal, then issue a move that closes the gap.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

import chess

from .agents import AgentChain
from .core import TRAY, Human, Mover, MoverError, Occupancy, OpponentSignal, Perception, Step, color_code, occupancy
from .engine import Engine
from .voice import SilentVoice
from .what import choose_piece
from .where import choose_destination, make_step, plan_move


class GameAborted(RuntimeError):
    pass


@dataclass
class Config:
    max_attempts: int = 6  # Dream-1 tries per goal before asking a person
    max_human_calls: int = 2  # per goal
    frames_per_observation: int = 3
    margin_cp: int = 0  # 0 always plays Stockfish's move; above 0 trades strength for easier handling
    max_opponent_reads: int = 3
    # More wrong squares than this means a bumped board or an arm in view, so don't let the
    # robot "fix" it. Ordinary failures only touch a few squares.
    max_auto_fix_squares: int = 6
    # The robot's moves in order, in SAN, to play instead of Stockfish's (a scripted collection game). A move that's
    # no longer legal, because the game left the script, falls back to Stockfish from then on.
    forced_moves: tuple[str, ...] = ()


@dataclass
class Stats:
    moves: int = 0
    dream_attempts: int = 0
    corrections: int = 0
    human_calls: int = 0


def compare(goal: Occupancy, seen: Occupancy) -> tuple[Occupancy, Occupancy]:
    """Return (missing, extra): pieces the goal has but the camera doesn't, and vice versa."""
    missing = {sq: c for sq, c in goal.items() if seen.get(sq) != c}
    extra = {sq: c for sq, c in seen.items() if goal.get(sq) != c}
    return missing, extra


def infer_opponent_move(board: chess.Board, seen: Occupancy) -> chess.Move | None:
    """Find the one legal move that produces what the camera sees.

    The camera only needs colors, not piece types. Chess rules supply the rest.
    """
    matches = []
    for move in board.legal_moves:
        board.push(move)
        if occupancy(board) == seen:
            matches.append(move)
        board.pop()
    if len(matches) == 1:
        return matches[0]
    # The four promotion choices look identical from above, so assume a queen.
    queen = [m for m in matches if m.promotion == chess.QUEEN]
    return queen[0] if queen else None


class Orchestrator:
    def __init__(
        self,
        board: chess.Board,
        robot_color: chess.Color,
        engine: Engine,
        mover: Mover,
        perception: Perception,
        human: Human,
        wait_for_opponent: OpponentSignal,
        config: Config | None = None,
        log: Callable[[str], None] = print,
        voice=None,
        agents: AgentChain | None = None,
    ):
        self.board = board
        self.robot_color = robot_color
        self.engine = engine
        self.mover = mover
        self.perception = perception
        self.human = human
        self.wait_for_opponent = wait_for_opponent
        self.config = config or Config()
        self.log = log
        self.voice = voice or SilentVoice()  # says each move and reacts; see voice.py
        # The helper agents between the checker and Dream-1 (agents.py). The default stand-ins change nothing.
        self.agents = agents or AgentChain()
        self.stats = Stats()

    def run(self, max_plies: int | None = None) -> str:
        plies = 0
        while not self.board.is_game_over(claim_draw=True):
            if max_plies is not None and plies >= max_plies:
                return "stopped"
            if self.board.turn == self.robot_color:
                self.play_robot_turn()
            else:
                self.read_opponent_turn()
            plies += 1
        return self.board.result(claim_draw=True)

    def _names(self) -> dict[str, str]:
        """Which piece is on each square, by name, as the game has it."""
        return {chess.square_name(s): chess.piece_name(p.piece_type) for s, p in self.board.piece_map().items()}

    def play_robot_turn(self) -> chess.Move:
        # Catch anything bumped since the last move before planning on top of it.
        self._achieve(occupancy(self.board), names=(self._names(), self._names()))

        scripted = self._scripted_move()
        if scripted is not None:
            piece = self.board.piece_at(scripted.from_square)
            self.log(f"WHAT  {chess.piece_name(piece.piece_type)} on {chess.square_name(scripted.from_square)} "
                     "(from the script: a move Dream-1 was trained on)")
            self.log(f"WHERE {self.board.san(scripted)} (from the script)")
            move = scripted
        else:
            choice = choose_piece(self.board, self.engine, self.config.margin_cp)
            self.log(
                f"WHAT  {chess.piece_name(choice.piece.piece_type)} on {chess.square_name(choice.square)} "
                f"(best {choice.piece_best_cp:+d} cp, {choice.crowding} neighbors, "
                f"{choice.alternatives} alternatives)"
            )
            dest = choose_destination(self.board, self.engine, choice, self.config.margin_cp)
            self.log(f"WHERE {self.board.san(dest.move)} ({dest.cp:+d} cp, risk {dest.risk})")
            move = dest.move
        self.voice.robot_move_chosen(self.board, move)

        # Follow each piece through the steps, so a correction can say "the brown rook", not just "the brown piece":
        # the camera only tells colors apart, but the game knows which piece belongs where.
        names = self._names()
        for step in plan_move(self.board, move):
            after = dict(names)
            moved = after.pop(step.source, step.piece_name) if step.source != TRAY else step.piece_name
            if step.target != TRAY:
                after[step.target] = moved
            self._achieve(step.expected_after, step, names=(names, after))
            names = after

        self.board.push(move)
        self.stats.moves += 1
        self.voice.robot_move_done(self.board)
        return move

    def _scripted_move(self) -> chess.Move | None:
        """The script's next robot move, while the game still follows the script."""
        index = self.stats.moves
        if getattr(self, "_off_script", False) or index >= len(self.config.forced_moves):
            return None
        wanted = self.config.forced_moves[index]
        try:
            move = self.board.parse_san(wanted)
            # The parser forgives a lot ("Nxe4" with nothing on e4), so insist on exactly the scripted move.
            if self.board.san(move).rstrip("+#") == wanted.rstrip("+#"):
                return move
            raise ValueError(wanted)
        except ValueError:
            self._off_script = True
            self.log(f"WHAT  the game left the script ({self.config.forced_moves[index]} isn't legal here): "
                     "Stockfish picks from now on")
            return None

    def read_opponent_turn(self) -> chess.Move:
        self.wait_for_opponent(self.board)
        for _ in range(self.config.max_opponent_reads):
            move = infer_opponent_move(self.board, self._observe())
            if move is not None:
                self.log(f"SEEN  opponent played {self.board.san(move)}")
                self.voice.opponent_moved(self.board, move)
                self.board.push(move)
                return move
        raise GameAborted("camera doesn't match any legal opponent move")

    def _observe(self) -> Occupancy:
        """Vote across several frames so one bad frame can't trigger a false correction."""
        votes: dict[str, Counter] = {}
        frames = self.config.frames_per_observation
        for _ in range(frames):
            frame = self.perception.observe()
            for sq in chess.SQUARE_NAMES:
                votes.setdefault(sq, Counter())[frame.get(sq)] += 1
        result: Occupancy = {}
        for sq, counter in votes.items():
            value, _ = counter.most_common(1)[0]
            if value is not None:
                result[sq] = value
        return result

    def _achieve(self, goal: Occupancy, first: Step | None = None,
                 names: tuple[dict[str, str], dict[str, str]] | None = None) -> None:
        """Keep acting until the camera shows `goal`, or escalate to a person. `names` says which piece is on each
        square before and after this step, so a correction can name the piece it moves."""
        step = first
        attempts = 0
        self.agents.new_goal()
        human_calls = 0
        while True:
            if step is not None:
                attempts += 1
                self.stats.dream_attempts += 1
                self.log(f"HOW   [{attempts}/{self.config.max_attempts}] {step.prompt}")
                try:
                    self.mover.execute(step)
                except MoverError as err:
                    self.log(f"      mover error: {err}")
                self.mover.go_home()  # the arms block the top camera

            missing, extra = compare(goal, self._observe())
            if not missing and not extra:
                return
            # The checker's verdict, in words, rides on the CHECK line back up to the person and the page:
            # "the brown queen went to b2 instead of c2". The correction below goes back down to Dream-1.
            from .checker import describe_failure

            verdict = describe_failure(step, missing, extra, (names or ({}, {}))[1])
            self.log(f"CHECK missing={sorted(missing)} extra={sorted(extra)}: {verdict}")
            self.agents.failed(step, verdict)
            failed = step

            can_retry = (
                attempts < self.config.max_attempts
                and len(missing) + len(extra) <= self.config.max_auto_fix_squares
            )
            step = self._corrective_step(missing, extra, goal, first, names) if can_retry else None
            if step is not None:
                step = self.agents.revise(step, verdict, failed)
                self.stats.corrections += 1
                self.voice.fixing()
                continue

            if human_calls >= self.config.max_human_calls:
                raise GameAborted(f"board still wrong after help: missing={missing} extra={extra}")
            human_calls += 1
            self.stats.human_calls += 1
            self.voice.asking_help()
            attempts = 0
            self.log(f"HELP  asking a person to fix the board: missing={sorted(missing)} extra={sorted(extra)}")
            self.human.request_fix(f"Please fix the board. Missing: {missing}. Extra: {extra}.", goal)

    def _corrective_step(self, missing: Occupancy, extra: Occupancy, goal: Occupancy, first: Step | None,
                         names: tuple[dict[str, str], dict[str, str]] | None = None) -> Step | None:
        """Pick the next physical action, or None if only a person can fix this."""
        before, after = names or ({}, {})
        for source, color in sorted(extra.items()):
            candidates = [sq for sq, c in missing.items() if c == color and sq != source]
            if candidates:
                target = min(
                    candidates,
                    key=lambda sq: (chess.square_distance(chess.parse_square(sq), chess.parse_square(source)), sq),
                )
                # The piece that belongs on the target is the one to bring there.
                return make_step(color, after.get(target, "piece"), source, target, self.robot_color, goal, fix=True)
        if extra:  # a stray with nowhere to go, e.g. a capture that wasn't removed
            source, color = sorted(extra.items())[0]
            return make_step(color, before.get(source, "piece"), source, TRAY, self.robot_color, goal, fix=True)
        if first is not None and first.source == TRAY and first.target in missing:
            return first  # the spare queen didn't arrive, so try again
        return None  # a piece vanished from view, likely off the table
