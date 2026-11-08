import chess
import pytest

import run_robot
from dreamchess.voice import REACTIONS, RobotVoice, describe_robot_move, make_voice
from test_run_robot import BROWN_MATES, args, events, scripted


def said(fen: str, uci: str) -> str:
    return describe_robot_move(chess.Board(fen), chess.Move.from_uci(uci))


class Recorder:
    def __init__(self):
        self.lines = []

    def say(self, text):
        self.lines.append(text)

    def close(self):
        pass


def test_a_plain_move_names_the_piece_and_both_squares():
    board = chess.Board()
    board.push_san("e4")
    assert describe_robot_move(board, chess.Move.from_uci("g8f6")) == "My knight moves from g8 to f6."


def test_a_capture_names_both_pieces():
    fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"
    assert said(fen, "d5e4") == "My pawn on d5 takes your pawn on e4."


def test_castling_says_both_the_king_and_the_rook():
    fen = "r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1"
    assert said(fen, "e8g8") == "I castle kingside: my king goes from e8 to g8, and my rook from h8 to f8."
    assert said(fen, "e8c8") == "I castle queenside: my king goes from e8 to c8, and my rook from a8 to d8."


def test_en_passant_and_promotion():
    assert said("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1", "d4e3") == \
        "My pawn on d4 takes your pawn on e4 en passant, landing on e3."
    assert said("4k3/8/8/8/8/8/p7/7K b - - 0 1", "a2a1q") == \
        "My pawn moves from a2 to a1, and becomes a queen. Check!"


def test_check_and_checkmate_are_announced():
    board = chess.Board(BROWN_MATES)
    board.push_san("Kg1")
    assert describe_robot_move(board, chess.Move.from_uci("b8b1")) == "My rook moves from b8 to b1. Checkmate!"


@pytest.mark.parametrize("fen,uci,kind", [
    ("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", "a1a8", "check"),
    ("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "e4d5", "capture"),
    ("4k3/8/8/8/8/8/8/4K2R w K - 0 1", "e1g1", "castle"),
    (chess.STARTING_FEN, "e2e4", "quiet"),
    ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8", "checkmate"),
])
def test_reactions_to_the_opponent(fen, uci, kind):
    rec = Recorder()
    RobotVoice(rec, seed=0).opponent_moved(chess.Board(fen), chess.Move.from_uci(uci))
    options = [line.format(taken="pawn") for line in REACTIONS[kind]]
    assert rec.lines[0] in options


def test_fixing_is_said_once_per_move():
    rec = Recorder()
    voice = RobotVoice(rec)
    voice.fixing()
    voice.fixing()
    assert len(rec.lines) == 1
    voice.robot_move_chosen(chess.Board(), chess.Move.from_uci("e2e4"))
    voice.fixing()
    assert len(rec.lines) == 3


def test_a_broken_speaker_never_stops_the_game():
    class Broken(Recorder):
        def say(self, text):
            raise RuntimeError("no network")

    RobotVoice(Broken()).asking_help()  # doesn't raise


def test_unknown_voice_is_refused():
    with pytest.raises(ValueError):
        make_voice("shout")


def test_a_simulated_game_speaks_its_moves(tmp_path):
    out = []
    ask = scripted("Kg1")
    summary = run_robot.play(args(tmp_path, sim=True, voice="print"), ask=ask, say=out.append)
    assert summary["result"] == "0-1"
    spoken = [e["text"] for e in events(tmp_path) if e["type"] == "voice"]
    assert spoken[0] in REACTIONS["quiet"]  # the person's Kg1
    assert spoken[1] == "My rook moves from b8 to b1. Checkmate!"
    assert spoken[2] == "Good game!"


def test_no_voice_by_default(tmp_path):
    run_robot.play(args(tmp_path, sim=True), ask=scripted("Kg1"), say=lambda line: None)
    assert not [e for e in events(tmp_path) if e["type"] == "voice"]
