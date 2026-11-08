"""Play a full game of chess against the robot. Runs on the NUC, at the station.

    python run_robot.py                          # moves the arms: a person must be at the e-stop
    python run_robot.py --fen "<position>"       # start from a position, such as a short mate
    python run_robot.py --sim                    # rehearse with a simulated robot; nothing moves

The robot plays Brown (Black), so you move first. You also stand in for the top camera: after
each robot step, you say whether the piece landed. Every event goes to games/<label>.jsonl.
Ctrl-C stops the arms at once.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import time
from pathlib import Path

import chess

from dreamchess.core import color_word, occupancy
from dreamchess.engine import make_engine
from dreamchess.human_eyes import AttemptMover, GameQuit, HumanEyes, TerminalHuman, TerminalOpponent
from dreamchess.orchestrator import Config, GameAborted, Orchestrator
from dreamchess.voice import make_voice

GAMES = Path(__file__).parent / "games"
ROBOT = chess.BLACK


class GameLog:
    def __init__(self, path: Path, listener=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.listener = listener  # also hands every event to the app's page, when there is one

    def __call__(self, event: dict) -> None:
        event = {"time": datetime.datetime.now().isoformat(timespec="seconds"), **event}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        if self.listener is not None:
            self.listener(event)


def plain(line: str) -> str:
    """Turn the orchestrator's shorthand into words for the person playing."""
    kind, _, rest = line.partition(" ")
    rest = rest.strip()
    found = re.match(r"(\w+) on (\w\d) \(best ([+-]\d+) cp", rest)
    if kind == "WHAT" and found:
        return f"I'll move my {found.group(1)} on {found.group(2)}."
    found = re.match(r"(\S+) \(", rest)
    if kind == "WHERE" and found:
        return f"My move: {found.group(1)}."
    found = re.match(r"\[(\d+)/(\d+)\] (.*)", rest)
    if kind == "HOW" and found:
        return f"Robot, attempt {found.group(1)} of {found.group(2)}: {found.group(3)}"
    found = re.match(r"missing=\[(.*?)\] extra=\[(.*?)\]", rest)
    if kind == "CHECK" and found:
        missing = re.findall(r"[a-h][1-8]", found.group(1))
        extra = re.findall(r"[a-h][1-8]", found.group(2))
        parts = []
        if missing:
            parts.append("expected a piece on " + ", ".join(missing))
        if extra:
            parts.append("found an unexpected piece on " + ", ".join(extra))
        verdict = rest.split("]: ", 1)[1] if "]: " in rest else ""
        if verdict:
            return f"That didn't work: {verdict}. Fixing it."
        return "That didn't work: " + " and ".join(parts) + ". Fixing it."
    if kind == "SEEN":
        return "You " + rest.split(" ", 1)[1] + "."
    if kind == "HELP":
        return "I can't fix this alone, so I'm asking a person for help."
    return line


def add_voice_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--voice", choices=("none", "print", "livekit"), default="none",
                        help="the robot says its moves and reacts to yours: print them, or speak over LiveKit")
    parser.add_argument("--voice-room", help="with --voice livekit, the LiveKit room (default dream-chess)")
    parser.add_argument("--voice-model", help='with --voice livekit, the TTS voice, "provider/model:voice"')
    parser.add_argument("--voice-local", action="store_true", help="with --voice livekit, also play it here")


def describe_position(board: chess.Board) -> str:
    groups: dict[str, list[str]] = {}
    for square, piece in sorted(board.piece_map().items()):
        color = color_word("w" if piece.color == chess.WHITE else "b")
        groups.setdefault(f"{color} {chess.piece_name(piece.piece_type)}", []).append(chess.square_name(square))
    return "\n".join(f"  {name}: {', '.join(squares)}" for name, squares in groups.items())


def play(args, ask=input, say=print, client=None, clock=time.monotonic, on_event=None, on_ready=None) -> dict:
    """Play one game. `on_event` receives every logged event; `on_ready(board, physical, stop_arms)` is called once
    the game is set up, with a function returning the physical board as the robot currently believes it."""
    script = ()
    if getattr(args, "script", None):
        from dreamchess.scripts import load_games

        game = load_games()[args.script]
        script = tuple(game)
        args.fen = chess.STARTING_FEN  # the scripted games start from the normal starting position
        say(f"Replaying collection game {args.script}: you play White's moves from the script, and the robot plays "
            "the brown moves Dream-1 was trained on.")
    from dreamchess import where

    where.PROMPT_STYLE = getattr(args, "prompt_style", "both")  # how squares are named to Dream-1
    where.ROBOT_SIDE = ROBOT
    board = chess.Board(args.fen)
    label = args.label or datetime.datetime.now().strftime("game-%Y%m%d-%H%M%S")
    log = GameLog(Path(args.log_dir) / f"{label}.jsonl", listener=on_event)
    if args.sim:
        from dreamchess.mock import MockDream, MockHuman, MockPerception, SimWorld

        world = SimWorld(board)
        eyes = None
        mover = AttemptMover(MockDream(world, seed=args.seed, p_success=args.sim_success), None, log, clock)
        perception = MockPerception(world, noise=0)
        human = MockHuman(world)
        if getattr(args, "auto_opponent", False):
            from dreamchess.mock import RandomOpponent

            opponent = RandomOpponent(world, seed=args.seed + 2)
        else:
            opponent = TerminalOpponent(ask, say, on_move=lambda after: setattr(world, "occ", occupancy(after)),
                                        record=log, script=tuple(w for w, _ in script))
        stop_arms = lambda: None
        physical = lambda: dict(world.occ)
    else:
        from dreamchess.station_mover import BASE_MODEL, StationClient, StationMover

        client = client or StationClient(args.station)
        status = client.status()
        if status.get("state") != "idle":
            raise SystemExit(f"The station is {status.get('state')}, not idle. Wait for it, then try again.")
        say("\nThe robot plays brown. Set up the board like this, with brown nearest the robot:")
        say(describe_position(board))
        ask("Press Enter when the board is set up: ")
        if ask("This moves the arms. Type yes when a person has a hand on the e-stop: ").strip().lower() != "yes":
            raise SystemExit("Stopped: nobody confirmed the e-stop.")
        eyes = HumanEyes(board, ask, say, record=log)
        args.checkpoint = args.checkpoint or BASE_MODEL  # the station keeps its last model unless told
        watch = getattr(args, "supervise", True)
        hands = getattr(args, "hand_feed", None)
        if hands is None and watch:
            from dreamchess.hands import HandFeed

            hands = HandFeed(getattr(args, "dream_python", "~/dream-env/bin/python"),
                             getattr(args, "board_fit", "~/.dream-chess/board_fit.json"))
        if watch and not hands.ready:
            say("Not watching moves as they happen: " + (hands.problem or "the hand tracker stopped") + ".")

        def supervise(step):
            from dreamchess.supervisor import Supervisor

            pieces = {chess.square_name(s): ("w" if p.color else "b", chess.piece_name(p.piece_type))
                      for s, p in board.piece_map().items()}
            return Supervisor(step, pieces, ROBOT, step.expected_after, log=say)

        station = StationMover(client, attempt_s=args.seconds, checkpoint=args.checkpoint, log=say,
                               hands=hands if watch else None, supervise=supervise)
        mover = AttemptMover(station, eyes, log, clock)
        # The checker agent: once the top camera is calibrated, a picture after every step decides, not a person.
        board_camera = getattr(args, "board_camera", None)
        if board_camera is not None:
            from dreamchess.checker import CameraChecker

            perception = CameraChecker(board_camera, eyes, log=lambda line: narrate(line))
        else:
            perception = eyes
        human = TerminalHuman(eyes, ask, say, record=log)
        opponent = TerminalOpponent(ask, say, on_move=eyes.opponent_moved, record=log,
                                    script=tuple(w for w, _ in script))
        stop_arms = client.stop
        def physical():
            seen = getattr(board_camera, "last_seen", None) if board_camera is not None else None
            calibrated = board_camera is not None and board_camera.status()["calibrated"]
            return dict(seen) if calibrated and seen is not None else dict(eyes.belief)

    # Start the engine only once every check has passed. Stockfish runs as a separate program, so
    # an engine started and then abandoned by a refusal would keep this program from exiting.
    engine = make_engine(args.stockfish)
    say(f"Chess engine: {engine.name}")
    if engine.name != "stockfish":
        say("WARNING: Stockfish isn't installed, so the robot will play a much weaker built-in engine.")

    def narrate(line: str) -> None:
        say(plain(line))
        log({"type": "agent", "text": line})

    def spoken(line: str) -> None:
        say(line)
        if line.startswith("Robot says: "):  # not the voice's own status lines
            log({"type": "voice", "text": line.removeprefix("Robot says: ")})

    # The person answers once per check, so there's nothing to vote across.
    config = Config(frames_per_observation=1 if eyes is not None else 3, forced_moves=tuple(b for _, b in script))
    voice = None
    try:
        voice = make_voice(getattr(args, "voice", "none"), out=spoken, room=getattr(args, "voice_room", None),
                           model=getattr(args, "voice_model", None), local=getattr(args, "voice_local", False),
                           seed=args.seed)
        agent = Orchestrator(board, ROBOT, engine, mover, perception, human, opponent, config=config, log=narrate,
                             voice=voice)
    except BaseException:
        engine.close()
        if voice is not None:
            voice.close()
        raise
    log({"type": "game_start", "fen": board.fen(), "robot": "brown", "engine": engine.name,
         "sim": bool(args.sim), "checkpoint": args.checkpoint, "seconds_per_attempt": args.seconds})
    if on_ready is not None:
        on_ready(board, physical, stop_arms)

    started = clock()
    ending = "finished"
    try:
        agent.run(args.max_plies)
    except GameQuit:
        ending = "quit"
    except KeyboardInterrupt:
        stop_arms()
        say("\nStopped the arms.")
        ending = "stopped with Ctrl-C"
    except GameAborted as err:
        ending = f"aborted: {err}"
    finally:
        engine.close()
        voice.close()  # lets its last line finish

    minutes = (clock() - started) / 60
    stats = agent.stats
    summary = {
        "type": "game_end", "ending": ending, "result": board.result(claim_draw=True),
        "moves": [m.uci() for m in board.move_stack], "robot_moves": stats.moves,
        "dream_attempts": stats.dream_attempts, "corrections": stats.corrections,
        "human_rescues": stats.human_calls, "minutes": round(minutes, 1),
        "robot_moves_per_hour": round(stats.moves / (minutes / 60), 1) if minutes > 0 and stats.moves else 0,
    }
    log(summary)
    say(f"\nGame {ending}. Result: {summary['result']}")
    say(f"Robot moves: {stats.moves}, Dream-1 attempts: {stats.dream_attempts}, "
        f"automatic fixes: {stats.corrections}, human rescues: {stats.human_calls}, minutes: {summary['minutes']}")
    say(f"Saved to {log.path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fen", default=chess.STARTING_FEN, help="starting position; default is a normal game")
    parser.add_argument("--label", help="a name for the game log; default is the date and time")
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="time limit per Dream-1 attempt, not counting time the arms wait on DreamAPI")
    parser.add_argument("--checkpoint", help="the model to run, such as a fine-tuned ID; default: the base model")
    parser.add_argument("--stockfish", help="path to Stockfish (or set STOCKFISH_PATH)")
    parser.add_argument("--max-plies", type=int, default=None, help="stop after this many half-moves")
    parser.add_argument("--script", type=int, choices=(1, 2, 3),
                        help="replay collection game 1, 2 or 3: the robot plays the moves Dream-1 was trained on")
    parser.add_argument("--station", default="http://127.0.0.1:8091")
    parser.add_argument("--log-dir", default=str(GAMES))
    parser.add_argument("--sim", action="store_true", help="rehearse with a simulated robot")
    parser.add_argument("--sim-success", type=float, default=0.4, help="how often the simulated robot succeeds")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-supervise", dest="supervise", action="store_false",
                        help="don't watch moves as they happen (or re-prompt Dream-1 mid-run)")
    parser.add_argument("--board-fit", default="~/.dream-chess/board_fit.json")
    parser.add_argument("--prompt-style", choices=("squares", "both", "spatial"), default="both",
                        help="name squares as in our demos (b8), describe where they are from the robot's seat, or both")
    parser.add_argument("--dream-python", default="~/dream-env/bin/python")
    add_voice_args(parser)
    play(parser.parse_args())


if __name__ == "__main__":
    main()
