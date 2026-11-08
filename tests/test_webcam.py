import chess
import pytest

from dreamchess.core import occupancy
from dreamchess.synth import PIECE_SIZES, SynthCamera
from dreamchess.vision import CalibrationError
from dreamchess.webcam import BoardCamera, read_move


def after(fen, *moves):
    board = chess.Board(fen)
    for m in moves:
        board.push_uci(m)
    return occupancy(board)


# --- Reading the person's move from what the camera sees ---

def test_a_legal_move_is_read_from_the_change():
    board = chess.Board()
    assert read_move(board, after(chess.STARTING_FEN, "e2e4")) == {"move": "e2e4", "san": "e4"}


def test_a_capture_is_read_from_colors_alone():
    board = chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
    assert read_move(board, after(board.fen(), "e4d5"))["san"] == "exd5"


def test_no_change_says_so():
    result = read_move(chess.Board(), occupancy(chess.Board()))
    assert "doesn't see any change" in result["problem"]


def test_an_illegal_move_says_exactly_how_to_undo_it():
    board = chess.Board()
    seen = occupancy(board)
    seen["g5"] = seen.pop("g2")  # a pawn can't jump three squares
    result = read_move(board, seen)
    assert result["problem"] == "Your pawn moved from g2 to g5, which isn't a legal move."
    assert result["fix"] == "Please move your pawn back from g5 to g2, then make a legal move."


def test_a_bumped_robot_piece_is_named():
    board = chess.Board("1r4k1/8/8/8/8/8/r7/7K w - - 0 1")
    seen = after(board.fen(), "h1g1")
    seen["c8"] = seen.pop("b8")  # the person's sleeve nudged the brown rook
    result = read_move(board, seen)
    assert result["problem"].startswith("Your move, Kg1, looks right")
    assert "the brown rook on b8 is missing" in result["problem"]
    assert "there's a brown piece on c8" in result["problem"]


# --- The webcam, calibrated and read, on the simulated camera ---

class Table:
    """The simulated camera, from White's side of the table, where a webcam on a stand would be."""

    def __init__(self):
        self.camera = SynthCamera(tilt_deg=25, seed=4, color_noise=3.0)

    def place(self, board):
        self.camera.pieces = {chess.square_name(s): ("w" if p.color else "b", *PIECE_SIZES[p.piece_type])
                              for s, p in board.piece_map().items()}

    def read(self):
        return self.camera.read()


def test_calibrate_then_read_the_persons_move(tmp_path):
    table = Table()
    cam = BoardCamera(lambda: table, tmp_path / "webcam.npz")
    assert cam.status()["step"] == "empty"
    table.camera.pieces = {}
    assert cam.capture_empty("bottom-left")["step"] == "start"
    table.place(chess.Board())
    assert cam.capture_start()["calibrated"]
    assert (tmp_path / "webcam.npz").exists()

    board = chess.Board()
    played = board.copy()
    played.push_uci("g1f3")
    table.place(played)
    assert read_move(board, cam.read()) == {"move": "g1f3", "san": "Nf3"}
    assert cam.snapshot(grid=True)[:2] == b"\xff\xd8"

    # The saved calibration works after a restart, with no recalibrating.
    again = BoardCamera(lambda: table, tmp_path / "webcam.npz")
    assert again.status()["calibrated"] and again.read() == occupancy(played)


def test_the_start_position_needs_the_empty_board_first(tmp_path):
    cam = BoardCamera(lambda: Table(), tmp_path / "webcam.npz")
    with pytest.raises(CalibrationError, match="empty board first"):
        cam.capture_start()


def test_the_orientation_is_found_without_being_told(tmp_path):
    table = Table()  # the simulated camera on White's side: a1 is at the bottom left of its picture
    cam = BoardCamera(lambda: table, tmp_path / "webcam.npz")
    table.camera.pieces = {}
    assert cam.capture_empty()["step"] == "start"
    table.place(chess.Board())
    assert cam.capture_start()["calibrated"]
    assert cam.read() == occupancy(chess.Board())
