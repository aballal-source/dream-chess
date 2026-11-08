"""The robot's voice: it says each move it makes and reacts to yours.

This file only decides WHAT to say, from the game itself, so it's plain code and fully testable.
HOW it's said is a Speaker: printed to the log (`--voice print`), or spoken over LiveKit
(`--voice livekit`, see livekit_voice.py). Speaking never blocks the game or the arms, and a
voice that fails only logs a warning: the game goes on silently.

    "My knight moves from g8 to f6."
    "My bishop on c5 takes your knight on f2. Check!"
    "Good check! My king steps out of the way."   (the reaction, then the robot's own move)
"""
from __future__ import annotations

import random
from typing import Callable, Protocol

import chess

ROBOT_PIECES, THEIR_PIECES = "my", "your"


class Speaker(Protocol):
    def say(self, text: str) -> None:
        """Speak `text` without waiting for it to finish."""
        ...

    def close(self) -> None: ...


class PrintSpeaker:
    """No audio: hands each line to a function, such as the page's log."""

    def __init__(self, out: Callable[[str], None] = print):
        self.out = out

    def say(self, text: str) -> None:
        self.out(f"Robot says: {text}")

    def close(self) -> None:
        pass


def _piece(board: chess.Board, square: int) -> str:
    return chess.piece_name(board.piece_type_at(square))


def describe_robot_move(board: chess.Board, move: chess.Move) -> str:
    """What the robot says about its own move, from the board BEFORE the move."""
    src, dst = chess.square_name(move.from_square), chess.square_name(move.to_square)
    piece = _piece(board, move.from_square)

    if board.is_castling(move):
        side = "kingside" if chess.square_file(move.to_square) > chess.square_file(move.from_square) else "queenside"
        rook_from, rook_to = ("h", "f") if side == "kingside" else ("a", "d")
        rank = src[1]
        text = (f"I castle {side}: my king goes from {src} to {dst}, "
                f"and my rook from {rook_from}{rank} to {rook_to}{rank}.")
    elif board.is_en_passant(move):
        taken = chess.square_name(chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square)))
        text = f"My pawn on {src} takes your pawn on {taken} en passant, landing on {dst}."
    elif board.is_capture(move):
        text = f"My {piece} on {src} takes your {_piece(board, move.to_square)} on {dst}."
    else:
        text = f"My {piece} moves from {src} to {dst}."

    if move.promotion:
        text = text[:-1] + f", and becomes a {chess.piece_name(move.promotion)}."

    after = board.copy(stack=False)
    after.push(move)
    if after.is_checkmate():
        text += " Checkmate!"
    elif after.is_check():
        text += " Check!"
    return text


# Reactions to the opponent's move, most specific first. Each has a few lines, so it doesn't repeat itself.
REACTIONS = {
    "checkmate": ["Checkmate. Well played, you got me!", "That's mate. Great game, you win!"],
    "check": ["Good check!", "Ooh, check. Nice one.", "Check! You're putting on the pressure."],
    "promotion": ["A new queen? Impressive.", "Promotion! That's a strong move."],
    "castle": ["Castling. Very solid.", "Good idea, keep that king safe."],
    "capture": ["Nice capture.", "Ouch, you took my {taken}.", "Good trade. I'll remember that {taken}."],
    "quiet": ["Good move.", "Interesting choice.", "Hmm, let me think about that.", "Okay, my turn."],
}


class RobotVoice:
    """Turns game events into lines and hands them to a Speaker. The orchestrator calls these hooks."""

    def __init__(self, speaker: Speaker, seed: int | None = None):
        self.speaker = speaker
        self.rng = random.Random(seed)
        self._fixing_said = False  # one "let me fix that" per step, not one per retry

    def _say(self, text: str) -> None:
        try:
            self.speaker.say(text)
        except Exception as err:  # a voice must never stop the game
            print(f"[voice] couldn't speak: {type(err).__name__}: {err}")

    def _pick(self, kind: str, **fields: str) -> str:
        return self.rng.choice(REACTIONS[kind]).format(**fields)

    def opponent_moved(self, board_before: chess.Board, move: chess.Move) -> None:
        after = board_before.copy(stack=False)
        after.push(move)
        if after.is_checkmate():
            kind = "checkmate"
        elif after.is_check():
            kind = "check"
        elif move.promotion:
            kind = "promotion"
        elif board_before.is_castling(move):
            kind = "castle"
        elif board_before.is_capture(move):
            kind = "capture"
        else:
            kind = "quiet"
        taken = "pawn" if board_before.is_en_passant(move) else (
            _piece(board_before, move.to_square) if board_before.is_capture(move) else "")
        self._say(self._pick(kind, taken=taken))

    def robot_move_chosen(self, board_before: chess.Board, move: chess.Move) -> None:
        self._fixing_said = False
        self._say(describe_robot_move(board_before, move))

    def robot_move_done(self, board_after: chess.Board) -> None:
        if board_after.is_checkmate():
            self._say("Good game!")
        elif board_after.is_game_over(claim_draw=True):
            self._say("It's a draw. Good game!")

    def fixing(self) -> None:
        if not self._fixing_said:
            self._fixing_said = True
            self._say("Oops, that didn't land right. Let me fix it.")

    def asking_help(self) -> None:
        self._say("I need a hand. Could someone fix the board for me?")

    def close(self) -> None:
        self.speaker.close()


class SilentVoice:
    """The default: says nothing."""

    def opponent_moved(self, board_before: chess.Board, move: chess.Move) -> None: ...
    def robot_move_chosen(self, board_before: chess.Board, move: chess.Move) -> None: ...
    def robot_move_done(self, board_after: chess.Board) -> None: ...
    def fixing(self) -> None: ...
    def asking_help(self) -> None: ...
    def close(self) -> None: ...


def make_voice(kind: str, out: Callable[[str], None] = print, room: str | None = None, model: str | None = None,
               local: bool = False, seed: int | None = None):
    """`none`, `print`, or `livekit`. LiveKit is imported only when asked for."""
    if kind == "none":
        return SilentVoice()
    if kind == "print":
        return RobotVoice(PrintSpeaker(out), seed)
    if kind == "livekit":
        from .livekit_voice import LiveKitSpeaker

        return RobotVoice(LiveKitSpeaker(room=room, model=model, local=local, log=out).start(), seed)
    raise ValueError(f"unknown voice {kind!r}: choose none, print, or livekit")
