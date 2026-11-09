import argparse
import json
import threading
import time
import urllib.error
import urllib.request

import chess
import pytest

import app
from test_station_mover import FakeStation, station  # noqa: F401  (the fixture)

# White's only move is Kg1; then Brown mates in one with Rb1#.
BROWN_MATES = "1r4k1/8/8/8/8/8/r7/7K w - - 0 1"


def args(tmp_path, **extra):
    defaults = dict(sim=True, auto_opponent=False, pace=0.0, fen=BROWN_MATES, checkpoint=None, seconds=0.2,
                    stockfish=None, station=None, sim_success=1.0, seed=1, max_plies=None, port=0,
                    log_dir=str(tmp_path), label=None)
    return argparse.Namespace(**{**defaults, **extra})


def wait_for(session, check, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = session.state(0)
        if check(state):
            return state
        time.sleep(0.05)
    raise AssertionError(f"timed out; last state: running={state['running']} pending={state['pending']}")


@pytest.mark.parametrize("prompt,kind", [
    ("Your move, such as e4, Nf3, or e2e4 (or 'quit'): ", "move"),
    ("  Did the brown rook from b8 end up on b1? y/n: ", "yesno"),
    ("This moves the arms. Type yes when a person has a hand on the e-stop: ", "confirm"),
    ("Press Enter when the board is set up: ", "continue"),
    ("  Anything else knocked out of place? Type 'from to', like e4 e3, or press Enter for no: ", "bumps"),
    ("  Where is it now? A square, 'same' if it didn't move, 'tray', or 'gone' if it left the board: ", "where"),
])
def test_questions_get_the_right_controls(prompt, kind):
    assert app.question_kind(prompt) == kind


def test_simulated_game_against_a_person_on_the_page(tmp_path):
    session = app.Session(args(tmp_path))
    session.start()
    state = wait_for(session, lambda s: s["pending"] and s["pending"]["kind"] == "move")
    assert session.answer(state["pending"]["id"], "Kg1")
    state = wait_for(session, lambda s: not s["running"])
    assert state["summary"]["result"] == "0-1"
    kinds = [e["type"] for e in session.state(0)["events"]]
    assert "game_start" in kinds and "attempt" in kinds and "game_end" in kinds
    texts = [e.get("text", "") for e in session.state(0)["events"] if e["type"] == "agent"]
    assert any(t.startswith("WHAT") for t in texts) and any(t.startswith("HOW") for t in texts)


def test_auto_opponent_plays_a_whole_game_unattended(tmp_path):
    session = app.Session(args(tmp_path, fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                               auto_opponent=True, max_plies=6, sim_success=0.5))
    session.start()
    state = wait_for(session, lambda s: not s["running"], timeout=60)
    assert state["summary"]["ending"] in ("finished", "stopped")
    assert state["pending"] is None


def test_state_reports_boards_and_mismatches(tmp_path):
    session = app.Session(args(tmp_path))
    session.start()
    state = wait_for(session, lambda s: s["fen"] is not None and s["pending"])
    assert state["fen"].startswith("1r4k1/")
    assert state["physical"]["b8"] == "b"
    assert state["mismatch"] == []
    assert state["mode"] == "simulation"
    session.answer(state["pending"]["id"], "quit")
    wait_for(session, lambda s: not s["running"])


def test_answers_only_reach_the_current_question(tmp_path):
    session = app.Session(args(tmp_path))
    session.start()
    state = wait_for(session, lambda s: s["pending"])
    assert not session.answer(state["pending"]["id"] + 1, "Kg1")
    assert session.answer(state["pending"]["id"], "quit")
    state = wait_for(session, lambda s: not s["running"])
    assert state["summary"]["ending"] == "quit"


def test_live_mode_refuses_a_busy_station_and_says_why(tmp_path, station):  # noqa: F811
    station.state = "running"
    session = app.Session(args(tmp_path, sim=False, station=station.url))
    session.start()
    state = wait_for(session, lambda s: not s["running"])
    assert "not idle" in state["summary"]["ending"]
    assert any(e["type"] == "error" for e in state["events"])


def test_live_mode_asks_for_the_e_stop_and_plays_through_the_station(tmp_path, station, monkeypatch):  # noqa: F811
    from dreamchess import station_mover

    original = station_mover.StationMover
    monkeypatch.setattr(station_mover, "StationMover",
                        lambda client, **kw: original(client, poll_s=0.02, grace_s=1.0, **kw))
    session = app.Session(args(tmp_path, sim=False, station=station.url))
    session.start()
    for expected, reply in (("continue", ""), ("confirm", "yes"), ("move", "Kg1"), ("yesno", "y"), ("bumps", "")):
        state = wait_for(session, lambda s: s["pending"] and s["pending"]["kind"] == expected)
        assert session.answer(state["pending"]["id"], reply)
    state = wait_for(session, lambda s: not s["running"])
    assert state["summary"]["result"] == "0-1"
    assert station.run == 1


def test_page_actions_need_the_apps_own_header(tmp_path):
    session = app.Session(args(tmp_path))
    server = app.make_server(session, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"Dream Chess" in urllib.request.urlopen(url + "/").read()
        request = urllib.request.Request(url + "/new", data=b"{}", method="POST")
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(request)
        assert err.value.code == 403
        state = json.loads(urllib.request.urlopen(url + "/state?since=0").read())
        assert state["running"] is False
    finally:
        server.shutdown()


# --- Fixes from the code review ---

def serve(session):
    server = app.make_server(session, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def request(port, path, host, method="GET", body=None, header=True):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body, method=method)
    req.add_header("Host", host)
    if header:
        req.add_header("X-Dream-Chess", "1")
    return urllib.request.urlopen(req, timeout=5)


def test_another_sites_name_is_refused_even_with_the_header(tmp_path):
    session = app.Session(args(tmp_path))
    server, port = serve(session)
    try:
        for method, path, body in (("GET", "/state", None), ("POST", "/new", b"{}")):
            with pytest.raises(urllib.error.HTTPError) as err:
                request(port, path, f"evil.example:{port}", method, body)
            assert err.value.code == 403
        assert request(port, "/state", f"localhost:{port}").status == 200
        assert request(port, "/state", f"127.0.0.1:{port}").status == 200
    finally:
        server.shutdown()


def test_malformed_requests_get_a_clear_refusal(tmp_path):
    session = app.Session(args(tmp_path))
    server, port = serve(session)
    try:
        for body in (b"[]", b'{"id": "x"}', b"not json"):
            with pytest.raises(urllib.error.HTTPError) as err:
                request(port, "/answer", f"localhost:{port}", "POST", body)
            assert err.value.code == 400
    finally:
        server.shutdown()


def test_a_page_from_before_a_restart_starts_over(tmp_path):
    session = app.Session(args(tmp_path))
    session.say("hello")
    state = session.state(500)
    assert state["reset"] is True and state["events"][0]["text"] == "hello" and state["boot"] == session.boot


def test_shutdown_ends_the_game_cleanly(tmp_path):
    session = app.Session(args(tmp_path))
    session.start()
    wait_for(session, lambda s: s["pending"])
    session.shutdown(wait_s=10)
    state = session.state(0)
    assert state["running"] is False
    assert state["summary"]["ending"] == "quit"
    assert any(e["type"] == "game_end" for e in state["events"])  # the summary was saved, the engine closed


def test_stop_reaches_the_station_at_any_point(tmp_path, station):  # noqa: F811
    session = app.Session(args(tmp_path, sim=False, station=station.url))
    session.stop_robot()  # before any game has even started
    assert [p for m, p, b in station.requests if p == "/stop"]


def test_an_unexpected_error_stops_the_arms(tmp_path, station, monkeypatch):  # noqa: F811
    def broken(*a, **k):
        raise RuntimeError("the station sent something strange")

    monkeypatch.setattr(app.run_robot, "play", broken)
    session = app.Session(args(tmp_path, sim=False, station=station.url))
    session.start()
    state = wait_for(session, lambda s: not s["running"])
    assert "error" in state["summary"]["ending"]
    assert [p for m, p, b in station.requests if p == "/stop"]


def test_the_page_only_sees_the_board_between_moves(tmp_path):
    session = app.Session(args(tmp_path))
    session.start()
    state = wait_for(session, lambda s: s["pending"] and s["pending"]["kind"] == "move")
    before = state["fen"]
    # The opponent-move check tries moves on the live board; the page's copy must not change while it does.
    board = session.board
    board.push(chess.Move.from_uci("h1g1"))
    assert session.state(0)["fen"] == before
    board.pop()
    session.answer(state["pending"]["id"], "quit")
    wait_for(session, lambda s: not s["running"])


# --- The robot's eyes and brain on the page ---

def test_simulated_camera_shows_the_board_and_the_vision_reads_it(tmp_path):
    session = app.Session(args(tmp_path))
    server, port = serve(session)
    try:
        session.start()
        wait_for(session, lambda s: s["fen"] is not None and s["pending"])
        deadline = time.time() + 60
        while not session.vision()["ready"] and time.time() < deadline:
            time.sleep(0.2)
        vision = json.loads(request(port, "/vision", f"localhost:{port}").read())
        assert vision["source"] == "camera" and vision["agrees"] is True
        assert vision["detected"] == session.state(0)["physical"]
        for path in ("/camera/top.jpg", "/camera/top.jpg?grid=1"):
            reply = request(port, path, f"localhost:{port}")
            assert reply.headers["Content-Type"] == "image/jpeg" and reply.read()[:2] == b"\xff\xd8"
        for path in ("/camera/left.jpg", "/camera/nope.jpg"):  # the simulation has only the top camera
            with pytest.raises(urllib.error.HTTPError) as err:
                request(port, path, f"localhost:{port}")
            assert err.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as err:
            request(port, "/camera/top.jpg", f"evil.example:{port}")
        assert err.value.code == 403
        session.answer(session.state(0)["pending"]["id"], "quit")
        wait_for(session, lambda s: not s["running"])
    finally:
        server.shutdown()


def test_the_brain_shows_what_the_agent_asked_dream_1(tmp_path):
    session = app.Session(args(tmp_path))
    session.start()
    state = wait_for(session, lambda s: s["pending"] and s["pending"]["kind"] == "move")
    assert session.brain()["prompt"] is None
    session.answer(state["pending"]["id"], "Kg1")
    wait_for(session, lambda s: not s["running"])
    brain = session.brain()
    assert brain["simulated"] and brain["calls"] >= 1 and brain["state"] == "idle"
    assert "rook" in brain["prompt"] and brain["arm"] in ("left", "right")


def test_live_page_shows_the_frames_dream_1_was_handed_and_the_station_status(tmp_path, station):  # noqa: F811
    frames = tmp_path / "frames"
    (frames / "top").mkdir(parents=True)
    (frames / "top" / "000001.jpg").write_bytes(b"\xff\xd8old")
    time.sleep(0.05)
    (frames / "top" / "000002.jpg").write_bytes(b"\xff\xd8new")
    session = app.Session(args(tmp_path, sim=False, station=station.url, frames=str(frames), top_camera=False))
    assert session.camera("top", grid=True) == b"\xff\xd8new"
    assert session.camera("left", grid=False) is None
    assert session.vision()["source"] == "person"
    brain = session.brain()
    assert brain["station"]["reachable"] is True and brain["station"]["state"] == "idle"
    assert brain["simulated"] is False


def test_brain_says_when_the_station_is_unreachable(tmp_path):
    session = app.Session(args(tmp_path, sim=False, station="http://127.0.0.1:9", frames=str(tmp_path)))
    assert session.brain()["station"] == {"reachable": False}


def test_the_page_shows_the_position_to_set_up_before_the_game_starts(tmp_path, station):  # noqa: F811
    session = app.Session(args(tmp_path, sim=False, station=station.url))
    session.start()
    state = wait_for(session, lambda s: s["pending"] and s["pending"]["kind"] == "continue")
    assert state["setup"] is True
    assert state["fen"] == chess.Board(BROWN_MATES).fen()
    assert state["physical"] == {"b8": "b", "g8": "b", "a2": "b", "h1": "w"}
    session.answer(state["pending"]["id"], "")
    state = wait_for(session, lambda s: s["pending"] and s["pending"]["kind"] == "confirm")
    assert state["setup"] is True  # still setting up until the e-stop is confirmed
    session.answer(state["pending"]["id"], "yes")
    state = wait_for(session, lambda s: s["pending"] and s["pending"]["kind"] == "move")
    assert state["setup"] is False
    session.answer(state["pending"]["id"], "quit")
    wait_for(session, lambda s: not s["running"])


# --- Starting the model on DreamAPI before the first move ---

def test_the_live_robot_runs_the_base_model_by_name_unless_told_otherwise(tmp_path, station):  # noqa: F811
    session = app.Session(args(tmp_path, sim=False, station=station.url))
    assert session.model() == "dreamlabs/dream-1-beta"
    assert app.Session(args(tmp_path, sim=False, checkpoint="dream-1-ft:x-chess")).model() == "dream-1-ft:x-chess"


def test_the_brain_shows_the_model_warming_up(tmp_path, station):  # noqa: F811
    session = app.Session(args(tmp_path, sim=False, station=station.url, checkpoint="dream-1-ft:x-chess"))
    session.warm_file = tmp_path / "warm.json"
    session.warm_file.write_text(json.dumps({"state": "warming", "checkpoint": "dream-1-ft:x-chess",
                                             "since": time.time() - 90}))
    warm = session.brain()["warm"]
    assert warm["state"] == "warming" and 89 <= warm["waited_s"] <= 92
    session.warm_file.write_text(json.dumps({"state": "ready", "checkpoint": "some-other-model", "since": 1}))
    assert session.brain()["warm"] is None  # a status for another model isn't this one's


def test_without_the_dream_library_the_warm_up_says_so(tmp_path, station):  # noqa: F811
    session = app.Session(args(tmp_path, sim=False, station=station.url, dream_python=str(tmp_path / "nope")))
    session.warm_file = tmp_path / "warm.json"
    session.warm_up()
    assert session.warm_process is None
    assert session.brain()["warm"]["state"] == "off"


def test_replaying_a_collection_game_suggests_whites_move_and_plays_the_trained_reply(tmp_path):
    session = app.Session(args(tmp_path, script=1, fen=chess.STARTING_FEN, max_plies=2))
    session.start()
    state = wait_for(session, lambda s: s["pending"] and s["pending"]["kind"] == "move")
    assert "[script: play e4, e2 to e4]" in state["pending"]["prompt"]
    assert state["fen"] == chess.Board().fen()
    session.answer(state["pending"]["id"], "e4")
    state = wait_for(session, lambda s: not s["running"])
    texts = [e.get("text", "") for e in state["events"] if e["type"] == "agent"]
    assert any(t.startswith("WHERE Nf6 (from the script)") for t in texts)
# --- The webcam reads the person's move ---

def webcam_session(tmp_path):
    from test_webcam import Table

    from dreamchess.webcam import BoardCamera

    table = Table()
    session = app.Session(args(tmp_path))
    session.webcam = BoardCamera(lambda: table, tmp_path / "webcam.npz")
    table.camera.pieces = {}
    assert session.webcam_action("empty", {"a1": "bottom-left"})["ok"]
    table.place(chess.Board())
    assert session.webcam_action("start", {})["calibrated"]
    return session, table


def test_the_webcam_reads_the_persons_move_and_the_game_goes_on(tmp_path):
    session, table = webcam_session(tmp_path)
    session.start()
    state = wait_for(session, lambda s: s["pending"] and s["pending"]["kind"] == "move")
    played = chess.Board(BROWN_MATES)
    played.push_uci("h1g1")
    table.place(played)
    result = session.camera_move(state["pending"]["id"])
    assert result["ok"] and result["san"] == "Kg1"
    state = wait_for(session, lambda s: not s["running"])
    assert state["summary"]["result"] == "0-1"
    assert any(e.get("text") == "The camera read your move: Kg1." for e in state["events"])


def test_the_webcam_says_how_to_undo_an_illegal_move(tmp_path):
    session, table = webcam_session(tmp_path)
    session.start()
    state = wait_for(session, lambda s: s["pending"] and s["pending"]["kind"] == "move")
    wrong = chess.Board(BROWN_MATES)
    wrong.remove_piece_at(chess.H1)
    wrong.set_piece_at(chess.H2, chess.Piece(chess.KING, chess.WHITE))  # into the rook's line: illegal
    table.place(wrong)
    result = session.camera_move(state["pending"]["id"])
    assert not result["ok"]
    assert result["problem"] == "Your king moved from h1 to h2, which isn't a legal move."
    assert result["fix"] == "Please move your king back from h2 to h1, then make a legal move."
    assert session.state(0)["pending"]["id"] == state["pending"]["id"]  # still waiting for a real move
    session.answer(state["pending"]["id"], "quit")
    wait_for(session, lambda s: not s["running"])


def test_restart_ends_the_game_and_starts_a_fresh_one(tmp_path):
    session = app.Session(args(tmp_path))
    session.start()
    first = wait_for(session, lambda s: s["pending"] and s["pending"]["kind"] == "move")
    assert session.restart()
    state = wait_for(session, lambda s: s["pending"] and s["pending"]["kind"] == "move"
                     and s["pending"]["id"] != first["pending"]["id"])
    assert state["running"] and state["fen"] == chess.Board(BROWN_MATES).fen()
    endings = [e for e in state["events"] if e["type"] == "game_end"]
    assert endings and endings[-1]["ending"] == "quit"  # the first game was ended, and its summary saved
    session.answer(state["pending"]["id"], "quit")
    wait_for(session, lambda s: not s["running"])


def test_restart_stops_the_arms_first_on_the_real_robot(tmp_path, station):  # noqa: F811
    session = app.Session(args(tmp_path, sim=False, station=station.url, top_camera=False))
    session.start()
    wait_for(session, lambda s: s["pending"])
    session.restart()
    assert [p for m, p, b in station.requests if p == "/stop"]
    session.shutdown()


# --- The robot's voice on the page ---

def test_what_the_robot_says_reaches_the_page(tmp_path):
    session = app.Session(args(tmp_path, voice="print"))
    session.start()
    state = wait_for(session, lambda s: s["pending"] and s["pending"]["kind"] == "move")
    assert state["voice"] == "print"
    session.answer(state["pending"]["id"], "Kg1")
    state = wait_for(session, lambda s: not s["running"])
    said = [e["text"] for e in state["events"] if e["type"] == "voice"]
    assert "My rook moves from b8 to b1. Checkmate!" in said


def test_the_listen_pass_is_only_given_with_the_livekit_voice(tmp_path):
    server, port = serve(app.Session(args(tmp_path)))
    try:
        with pytest.raises(urllib.error.HTTPError) as err:
            request(port, "/voice/token", f"localhost:{port}")
        assert err.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as err:  # and never to another site's page
            request(port, "/voice/token", f"evil.example:{port}")
        assert err.value.code == 403
    finally:
        server.shutdown()


def test_the_listen_pass_can_only_listen(tmp_path, monkeypatch):
    api = pytest.importorskip("livekit.api")
    import base64

    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret-for-tests-only-0123456789")
    found = app.Session(args(tmp_path, voice="livekit")).voice_token()
    payload = found["token"].split(".")[1]
    grants = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))["video"]
    assert grants["roomJoin"] and grants["room"] == "dream-chess" and grants["canPublish"] is False
    assert found["url"] == "wss://example.livekit.cloud" and api


# --- The public demo site ---

def test_the_public_site_only_ever_runs_the_simulation(tmp_path, station):  # noqa: F811
    with pytest.raises(ValueError, match="must never reach a real robot"):
        app.make_server(app.Session(args(tmp_path, sim=False, station=station.url)), 0, public=True)


def test_visitors_can_watch_and_restart_but_nothing_else(tmp_path):
    session = app.Session(args(tmp_path, public=True, auto_opponent=True, max_plies=200))
    server = app.make_server(session, 0, public=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        session.start()
        page = request(port, "/", f"dream-chess.hf.space:{port}", header=False)  # any visitor's address
        assert page.status == 200
        state = json.loads(request(port, "/state", f"dream-chess.hf.space:{port}").read())
        assert state["public"] is True
        for path in ("/answer", "/stop", "/new", "/webcam/empty"):
            with pytest.raises(urllib.error.HTTPError) as err:
                request(port, path, f"dream-chess.hf.space:{port}", "POST", b"{}")
            assert err.value.code == 403
        first = json.loads(request(port, "/restart", f"dream-chess.hf.space:{port}", "POST", b"{}").read())
        again = json.loads(request(port, "/restart", f"dream-chess.hf.space:{port}", "POST", b"{}").read())
        assert first["ok"] is True and again["ok"] is False  # one restart per 15 s, for everyone
    finally:
        session.shutdown()
        server.shutdown()


# --- The landing page's services panel and the stats page ---

def test_the_services_panel_lists_every_part_and_marks_the_agent_slots(tmp_path):
    session = app.Session(args(tmp_path, voice="print"))
    services = {s["name"]: s for s in session.brain()["services"]}
    assert {"Dream-1 · DreamAPI", "Chess engine", "Watcher agent", "Prompt writer agent", "Top camera checker",
            "Voice · LiveKit"} <= set(services)
    assert services["Watcher agent"]["state"] == "placeholder"
    assert services["Voice · LiveKit"]["detail"] == "Text only, no audio"


def test_the_stats_page_counts_the_game_and_lists_every_instruction(tmp_path):
    session = app.Session(args(tmp_path, fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                               auto_opponent=True, max_plies=4, sim_success=0.5))
    session.start()
    wait_for(session, lambda s: not s["running"], timeout=60)
    stats = session.stats()
    assert stats["robot_moves"] == 2
    assert stats["dream_calls"] == len([p for p in stats["prompts"] if p["kind"] != "supervisor"])
    assert stats["prompts"] and stats["prompts"][0]["attempt"] == "1/6"
    server, port = serve(session)
    try:
        assert b"Stats" in request(port, "/stats", f"localhost:{port}").read()
        assert json.loads(request(port, "/stats.json", f"localhost:{port}").read())["robot_moves"] == 2
    finally:
        server.shutdown()
