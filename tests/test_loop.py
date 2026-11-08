import chess
import pytest

from dreamchess.core import occupancy
from dreamchess.engine import FallbackEngine
from dreamchess.mock import MockDream, MockHuman, MockPerception, RandomOpponent, SimWorld
from dreamchess.orchestrator import Config, GameAborted, Orchestrator, infer_opponent_move
from dreamchess.what import choose_piece
from dreamchess.where import arm_for, choose_destination, plan_move


def build(fen=chess.STARTING_FEN, color=chess.WHITE, script=None, p_success=1.0, noise=0.0, fixes=True, seed=0, config=None):
    board = chess.Board(fen)
    world = SimWorld(board)
    mover = MockDream(world, seed=seed, p_success=p_success, script=script)
    human = MockHuman(world, fixes=fixes)
    agent = Orchestrator(
        board, color, FallbackEngine(depth=1), mover,
        MockPerception(world, seed=seed + 1, noise=noise), human,
        RandomOpponent(world, seed=seed + 2), config=config, log=lambda _: None,
    )
    return agent, world, mover, human


# --- WHAT and WHERE ---

BACK_RANK_MATE = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"


def test_finds_mate_in_one():
    board = chess.Board(BACK_RANK_MATE)
    engine = FallbackEngine()
    choice = choose_piece(board, engine)
    dest = choose_destination(board, engine, choice)
    assert dest.move == chess.Move.from_uci("d1d8")


def test_never_trades_forced_mate_for_easier_handling():
    board = chess.Board(BACK_RANK_MATE)
    engine = FallbackEngine()
    choice = choose_piece(board, engine, margin_cp=99_999)
    assert choose_destination(board, engine, choice, margin_cp=99_999).move == chess.Move.from_uci("d1d8")


def test_prefers_uncrowded_piece_when_scores_are_close():
    # Equal material, and the a1 rook has more room than the king.
    board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    engine = FallbackEngine(depth=1)
    choice = choose_piece(board, engine, margin_cp=10_000)
    assert choice.crowding == 0


def test_margin_limits_total_loss_across_what_and_where():
    board = chess.Board()
    engine = FallbackEngine(depth=1)
    choice = choose_piece(board, engine, margin_cp=30)
    assert choose_destination(board, engine, choice, margin_cp=30).cp >= choice.global_best_cp - 30


def test_arm_follows_seat():
    assert arm_for("a1", chess.WHITE) == "left"
    assert arm_for("h1", chess.WHITE) == "right"
    assert arm_for("a8", chess.BLACK) == "right"
    assert arm_for("h8", chess.BLACK) == "left"


# --- Turning moves into physical steps ---

SPECIAL_MOVES = [
    ("quiet", chess.STARTING_FEN, "g1f3", 1),
    ("capture", "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 2),
    ("kingside_castle", "4k3/8/8/8/8/8/8/4K2R w K - 0 1", "e1g1", 2),
    ("queenside_castle", "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1", "e1c1", 2),
    ("en_passant", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", 2),
    ("promotion", "4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q", 2),
    ("capture_promotion", "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8q", 3),
    ("black_capture", "4k3/8/8/3p4/4P3/8/8/4K3 b - - 0 1", "d5e4", 2),
]


@pytest.mark.parametrize("name,fen,uci,count", SPECIAL_MOVES, ids=[m[0] for m in SPECIAL_MOVES])
def test_steps_end_exactly_at_the_chess_result(name, fen, uci, count):
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    steps = plan_move(board, move)
    assert len(steps) == count
    after = board.copy()
    after.push(move)
    assert steps[-1].expected_after == occupancy(after)


@pytest.mark.parametrize("name,fen,uci,count", SPECIAL_MOVES, ids=[m[0] for m in SPECIAL_MOVES])
def test_robot_executes_every_special_move(name, fen, uci, count):
    board = chess.Board(fen)
    agent, world, _, _ = build(fen=fen, color=board.turn)
    move = chess.Move.from_uci(uci)
    for step in plan_move(board, move):
        agent._achieve(step.expected_after, step)
    board.push(move)
    assert world.occ == occupancy(board)


# --- Reading the opponent from the camera ---

@pytest.mark.parametrize("name,fen,uci,count", SPECIAL_MOVES, ids=[m[0] for m in SPECIAL_MOVES])
def test_infers_any_move_from_occupancy(name, fen, uci, count):
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    after = board.copy()
    after.push(move)
    assert infer_opponent_move(board, occupancy(after)) == move


def test_rejects_position_no_legal_move_can_reach():
    board = chess.Board()
    seen = occupancy(board)
    del seen["e2"]
    assert infer_opponent_move(board, seen) is None


# --- Recovery ---

@pytest.mark.parametrize("failure", ["miss", "off_by_one", "api_error"])
def test_recovers_from_failed_attempt(failure):
    agent, world, mover, human = build(script=[failure, "success", "success"], fen="4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    agent.play_robot_turn()
    assert world.occ == occupancy(agent.board)
    assert agent.stats.dream_attempts >= 2
    assert not human.calls


@pytest.mark.parametrize("failure", ["miss", "off_by_one"])
def test_a_retry_names_the_piece_not_just_its_color(failure):
    # The camera only tells colors apart, but the game knows which piece belongs where, and Dream-1 was trained
    # on prompts that name the piece ("the brown rook on square b8"), so a retry should too.
    prompts = []
    agent, world, mover, human = build(script=[failure, "success"], fen="1r4k1/8/8/8/8/8/r7/6K1 b - - 1 1",
                                       color=chess.BLACK)
    agent.log = prompts.append
    agent.play_robot_turn()
    hows = [p for p in prompts if p.startswith("HOW")]
    assert len(hows) == 2 and all("brown rook" in p for p in hows), hows
    assert world.occ == occupancy(agent.board)


def test_returns_a_knocked_neighbor_to_its_square():
    agent, world, _, human = build(fen="4k3/8/8/8/8/8/PP6/R3K3 w - - 0 1")
    goal = occupancy(agent.board)
    world.occ["a3"] = world.occ.pop("a2")  # bumped since the last move
    agent._achieve(goal)
    assert world.occ == goal
    assert not human.calls


def test_lost_piece_goes_to_a_person_and_game_continues():
    agent, world, _, human = build(script=["lost"], fen="4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    agent.play_robot_turn()
    assert len(human.calls) == 1
    assert world.occ == occupancy(agent.board)


def test_board_does_not_advance_until_move_is_verified():
    agent, world, _, _ = build(
        script=["miss"] * 50, fen="4k3/8/8/8/8/8/8/R3K3 w - - 0 1", fixes=False,
        config=Config(max_attempts=2, max_human_calls=1),
    )
    before = agent.board.fen()
    with pytest.raises(GameAborted):
        agent.play_robot_turn()
    assert agent.board.fen() == before


def test_single_bad_frames_do_not_trigger_corrections():
    agent, world, mover, human = build(noise=0.3, seed=3)
    goal = occupancy(agent.board)
    frames = [dict(goal) for _ in range(3)]
    del frames[0]["e2"]
    agent.perception.observe = lambda: frames.pop()
    agent._achieve(goal)
    assert mover.prompts == []
    assert not human.calls


# --- Whole games ---

@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("color", [chess.WHITE, chess.BLACK], ids=["white", "black"])
def test_full_game_keeps_table_and_board_in_sync(seed, color):
    agent, world, _, _ = build(color=color, p_success=0.4, noise=0.02, seed=seed)
    agent.run(max_plies=30)
    assert world.occ == occupancy(agent.board)
    assert agent.board.ply() == 30 or agent.board.is_game_over(claim_draw=True)
    assert agent.board.is_valid()


# --- Describing squares the way the robot sees them ---

def test_squares_are_described_from_the_robots_seat():
    from dreamchess.where import describe_path, describe_square

    # Brown sits behind rank 8 with the a-file on its right.
    assert describe_square("b8", chess.BLACK) == "nearest row, 2nd from your right"
    assert describe_square("h1", chess.BLACK) == "farthest row, leftmost"
    assert describe_square("e4", chess.BLACK) == "5th row, 4th from your left"
    assert describe_path("b8", "b1", chess.BLACK) == "seven rows straight away from you"
    assert describe_path("g8", "f6", chess.BLACK) == "two rows away from you and one column to your right"
    # White, behind rank 1, has the h-file on its right.
    assert describe_square("h1", chess.WHITE) == "nearest row, rightmost"


def test_the_prompt_styles():
    from dreamchess import where

    squares = where.write_prompt("right", "b", "rook", "b8", "b1", style="squares")
    assert squares == ("Using your right arm, pick up the brown rook on square b8 and place it upright in the center "
                       "of square b1. Move slowly and do not touch any other piece.")
    both = where.write_prompt("right", "b", "rook", "b8", "b1", style="both")
    assert both == ("Using your right arm, pick up the brown rook at Column (b), Row (8) (nearest row, 2nd from your "
                    "right) and place it at Column (b), Row (1) (farthest row, 2nd from your right). Move slowly and "
                    "do not touch any other piece.")
    spatial = where.write_prompt("right", "b", "rook", "b8", "b1", style="spatial")
    assert "b8" not in spatial and "b1" not in spatial and "at the nearest row, 2nd from your right" in spatial
    tray = where.write_prompt("right", "w", "pawn", "d3", where.TRAY, style="both")
    assert tray.endswith("put it in the tray on your right. Move slowly and do not touch any other piece.")


def test_a_follow_up_after_a_failed_check_is_one_short_line():
    from dreamchess import where

    assert where.write_prompt("right", "b", "rook", "b3", "b1", style="both", fix=True) == (
        "Using your right arm, move the brown rook from Column (b), Row (3) (6th row, 2nd from your right) to "
        "Column (b), Row (1) (farthest row, 2nd from your right).")
    # In our demos' wording, a fix is worded exactly as the fix demos were.
    assert where.write_prompt("right", "b", "rook", "b3", "b1", style="squares", fix=True) == (
        "Using your right arm, pick up the brown rook on square b3 and place it upright in the center of square b1. "
        "Move slowly and do not touch any other piece.")
