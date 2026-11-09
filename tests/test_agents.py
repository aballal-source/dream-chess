"""The helper agents between the checker and Dream-1 (dreamchess/agents.py)."""
import chess

from dreamchess.agents import AgentChain, LLMPromptWriter, LLMWatcher, RuleWatcher, TemplatePrompts, make_agents
from dreamchess.core import Step
from dreamchess.mock import MockDream, MockHuman, MockPerception, RandomOpponent, SimWorld
from dreamchess.engine import make_engine
from dreamchess.orchestrator import Config, Orchestrator


def step(prompt="pick up the brown rook on square b8 and place it on b1"):
    return Step("b", "rook", "b8", "b1", "right", prompt, {"b1": "b"})


def test_the_stand_ins_change_nothing_and_log_nothing():
    lines = []
    chain = make_agents("rules", log=lines.append)
    s = step()
    assert chain.revise(s, "the brown rook went to b3 instead of b1", s) is s
    assert lines == []
    assert [a["placeholder"] for a in chain.status()] == [False, False]


def test_the_llm_placeholders_fall_back_and_say_so_once():
    lines = []
    chain = make_agents("llm", model="some-model", log=lines.append)
    s = step()
    for _ in range(2):
        chain.failed(s, "the brown rook went to b3 instead of b1")
        assert chain.revise(s, "the brown rook went to b3 instead of b1", s).prompt == s.prompt
    assert sum(1 for line in lines if line.startswith("AGENT")) == 2  # one per agent, not per call
    assert any(line.startswith("HANDOFF checker -> watcher") for line in lines)
    assert all(not a["connected"] and a["placeholder"] for a in chain.status())


def test_a_connected_prompt_writer_rewords_the_correction():
    class Shorter(LLMPromptWriter):
        connected = True

        def _ask_model(self, system, user, picture=None):
            assert "b3 instead of b1" in user and "Earlier:" in user
            return "Using your right arm, slide the brown rook from b3 to b1."

    chain = AgentChain(RuleWatcher(), Shorter("test"), log=lambda _: None)
    chain.failed(step(), "the brown rook went to b3 instead of b1")
    revised = chain.revise(step(), "the brown rook went to b3 instead of b1", step())
    assert revised.prompt == "Using your right arm, slide the brown rook from b3 to b1."
    assert revised.source == "b8" and revised.target == "b1"


def test_the_watcher_sees_the_picture():
    seen = {}

    class Eyes(LLMWatcher):
        connected = True

        def _ask_model(self, system, user, picture=None):
            seen["picture"] = picture
            return "the rook is lying on its side on b3"

    chain = AgentChain(Eyes("test"), TemplatePrompts(), picture=lambda: b"jpeg")
    chain.revise(step(), "the brown rook went to b3 instead of b1", step())
    assert seen["picture"] == b"jpeg"


def test_a_game_runs_the_same_with_the_llm_placeholders():
    def game(agents):
        board = chess.Board()
        world = SimWorld(board)
        engine = make_engine(None)
        try:
            agent = Orchestrator(board, chess.BLACK, engine, MockDream(world, seed=3, p_success=0.5),
                                 MockPerception(world, noise=0), MockHuman(world), RandomOpponent(world, seed=5),
                                 config=Config(), log=lambda _: None, agents=agents)
            agent.run(max_plies=6)
        finally:
            engine.close()
        return [m.uci() for m in board.move_stack], agent.stats.corrections

    assert game(make_agents("rules")) == game(make_agents("llm"))


def test_hand_offs_read_as_words_for_the_person_playing():
    import run_robot

    assert run_robot.plain("HANDOFF checker -> watcher: the rook went to b3") == "Checker to watcher: the rook went to b3"
    assert run_robot.plain("AGENT watcher isn't connected (x): using the rule-based stand-in") == \
        "Watcher isn't connected (x): using the rule-based stand-in."
