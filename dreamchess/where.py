"""WHERE: choose the destination square, then expand the move into physical steps.

One chess move is not always one pick-and-place. Captures, castling, en passant, and
promotion all need several, and each step gets verified separately.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess

from .core import TRAY, Occupancy, Step, color_code, color_word, occupancy
from .engine import Engine
from .what import PieceChoice, candidate_moves, crowding, within_margin

CAPTURE_RISK = 3  # a capture adds a whole extra pick-and-place


@dataclass(frozen=True)
class Destination:
    move: chess.Move
    cp: int
    risk: int


def choose_destination(
    board: chess.Board, engine: Engine, choice: PieceChoice, margin_cp: int = 0
) -> Destination:
    moves = [m for m in candidate_moves(board) if m.from_square == choice.square]
    ranked = engine.rank(board, moves)

    def risk(move: chess.Move) -> int:
        return crowding(board, move.to_square, ignore=move.from_square) + (
            CAPTURE_RISK if board.is_capture(move) else 0
        )

    # Compare against the best move overall, so WHAT and WHERE can't each spend the
    # margin and add up to twice the allowed loss.
    eligible = [s for s in ranked if within_margin(s.cp, choice.global_best_cp, margin_cp)]
    eligible = eligible or ranked[:1]  # engine scores vary a little between searches
    best = min(eligible, key=lambda s: (risk(s.move), -s.cp, s.move.uci()))
    return Destination(best.move, best.cp, risk(best.move))


def arm_for(square: str, robot_color: chess.Color) -> str:
    """Seated behind its own pieces, White has the a-file on its left; Black the h-file."""
    queenside = square[0] in "abcd"
    return "left" if queenside == (robot_color == chess.WHITE) else "right"


# How a square is named in the prompt (set from the app's --prompt-style):
#   "squares": "on square b8", the wording our demos were recorded with;
#   "both":    "at Column (b), Row (8) (in the row nearest you, the second square in from your right)", the board's
#              printed labels plus where the square is (a little better than square names on the robot);
#   "spatial": the description alone.
# Dream-1 sees the board through its cameras but has no reason to know what "b8" means: the printed edge labels are
# tiny in its pictures. Rows and columns counted from where it sits are things it can see.
PROMPT_STYLE = "squares"
ROBOT_SIDE = chess.BLACK  # which side's edge the robot sits behind, for "nearest you" and "your right"
ORDINALS = ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
NUMBERS = ["no", "one", "two", "three", "four", "five", "six", "seven"]
SHORT = ["", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]


def _row_col(square: str, side: chess.Color) -> tuple[int, int]:
    """(row counted from the robot's edge, 1 = nearest; column counted from the robot's right, 1 = rightmost)."""
    file, rank = "abcdefgh".index(square[0]), int(square[1])
    if side == chess.BLACK:  # behind rank 8, facing White: the a-file is on its right
        return 9 - rank, file + 1
    return rank, 8 - file  # behind rank 1: the h-file is on its right


def describe_square(square: str, side: chess.Color = None) -> str:
    """Where a square is, as the robot sees it, in as few words as will do: "nearest row, 2nd from your right"."""
    row, col = _row_col(square, ROBOT_SIDE if side is None else side)
    rows = "nearest row" if row == 1 else "farthest row" if row == 8 else f"{SHORT[row]} row"
    if col <= 4:
        cols = "rightmost" if col == 1 else f"{SHORT[col]} from your right"
    else:
        left = 9 - col
        cols = "leftmost" if left == 1 else f"{SHORT[left]} from your left"
    return f"{rows}, {cols}"


def describe_path(source: str, target: str, side: chess.Color = None) -> str:
    """How far and which way: "seven rows straight away from you", "two rows away from you and one to your right"."""
    side = ROBOT_SIDE if side is None else side
    (r0, c0), (r1, c1) = _row_col(source, side), _row_col(target, side)
    rows, right = r1 - r0, c0 - c1  # rows away from the robot; columns toward its right (its right is column 1)
    away = "away from you" if rows > 0 else "toward you"
    if rows and not right:
        return f"{NUMBERS[abs(rows)]} row{'s' if abs(rows) > 1 else ''} straight {away}"
    parts = []
    if rows:
        parts.append(f"{NUMBERS[abs(rows)]} row{'s' if abs(rows) > 1 else ''} {away}")
    if right:
        parts.append(f"{NUMBERS[abs(right)]} column{'s' if abs(right) > 1 else ''} to your {'right' if right > 0 else 'left'}")
    return " and ".join(parts)


def _where(square: str, style: str) -> str:
    if style == "spatial":
        return f"the {describe_square(square)}"
    if style == "both":
        return f"Column ({square[0]}), Row ({square[1]}) ({describe_square(square)})"
    return f"square {square}"


def write_prompt(arm: str, color: str, piece_name: str, source: str, target: str, style: str | None = None,
                 fix: bool = False) -> str:
    """The text Dream-1 receives: who (which arm), what (which piece), where from, where to.

    A first instruction keeps the shape of our training demos. A follow-up after a failed check (`fix`) is one
    short line, since the arm, the piece and the goal are already set: move it from where it is now to the target.
    The "squares" style is our demos' exact wording, follow-ups included."""
    style = style or PROMPT_STYLE
    item = f"{color_word(color)} {piece_name}".strip()
    tray = f"the tray on your {arm}"
    if style == "squares":  # word for word as recorded, so the fine-tunes see what they were trained on
        pick = f"pick up the spare {item} from {tray}" if source == TRAY else f"pick up the {item} on square {source}"
        place = f"put it in {tray}" if target == TRAY else f"place it upright in the center of square {target}"
        return f"Using your {arm} arm, {pick} and {place}. Move slowly and do not touch any other piece."
    here = tray if source == TRAY else _where(source, style)
    there = tray if target == TRAY else _where(target, style)
    if fix:
        return f"Using your {arm} arm, move the {item} from {here} to {there}."
    pick = f"pick up the spare {item} from {tray}" if source == TRAY else f"pick up the {item} at {here}"
    place = f"put it in {tray}" if target == TRAY else f"place it at {there}"
    return f"Using your {arm} arm, {pick} and {place}. Move slowly and do not touch any other piece."


def make_step(
    color: str, piece_name: str, source: str, target: str, robot_color: chess.Color, expected_after: Occupancy,
    fix: bool = False,
) -> Step:
    arm = arm_for(target if source == TRAY else source, robot_color)
    return Step(
        color=color,
        piece_name=piece_name,
        source=source,
        target=target,
        arm=arm,
        prompt=write_prompt(arm, color, piece_name, source, target, fix=fix),
        expected_after=expected_after,
    )


def plan_move(board: chess.Board, move: chess.Move) -> list[Step]:
    """Expand a legal move into ordered steps. Doesn't modify the board."""
    robot_color = board.turn
    us = color_code(robot_color)
    them = color_code(not robot_color)
    src = chess.square_name(move.from_square)
    dst = chess.square_name(move.to_square)
    name = chess.piece_name(board.piece_type_at(move.from_square))

    occ = occupancy(board)
    steps: list[Step] = []

    def add(color: str, piece_name: str, source: str, target: str) -> None:
        if source != TRAY:
            del occ[source]
        if target != TRAY:
            occ[target] = color
        steps.append(make_step(color, piece_name, source, target, robot_color, dict(occ)))

    if board.is_en_passant(move):
        captured = chess.square_name(chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square)))
        add(them, "pawn", captured, TRAY)
    elif board.is_capture(move):
        add(them, chess.piece_name(board.piece_type_at(move.to_square)), dst, TRAY)

    if move.promotion:
        add(us, "pawn", src, TRAY)
        add(us, "queen", TRAY, dst)
    else:
        add(us, name, src, dst)

    if board.is_castling(move):
        rank = src[1]
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        add(us, "rook", ("h" if kingside else "a") + rank, ("f" if kingside else "d") + rank)

    return steps
