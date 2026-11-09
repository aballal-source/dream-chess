"""The helper agents between the checker and Dream-1: slots for the LLM agents the system still needs.

What the hackathon showed (see docs/NEXT_STEPS.md): every retry was a fixed template. Nothing looked at the board
and changed how the instruction was worded, so Dream-1 got the same kind of sentence after every miss. This module is
where that changes. After a failed check, the orchestrator hands the failure to a chain of two agents:

    checker --verdict--> watcher --what happened--> prompt writer --next instruction--> Dream-1

- The **watcher** looks at what went wrong (the checker's verdict, and later the top-camera picture) and says what
  actually happened, in words Dream-1's next instruction can use: "the rook is lying on its side on b3".
- The **prompt writer** turns that into the next instruction for Dream-1, given every attempt so far.

Each has a working stand-in today, so the robot plays exactly as before:
- `RuleWatcher` passes the checker's verdict through unchanged.
- `TemplatePrompts` keeps the template sentence the orchestrator wrote (where.write_prompt).

`LLMWatcher` and `LLMPromptWriter` are the placeholders for the real agents. They have the interfaces the chain calls
and say they aren't connected; until someone fills in `_ask_model`, they fall back to the stand-ins. Choose them with
`--agents llm`. Each hand-off is logged as a HANDOFF line, which the page shows over the robot's live view.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .core import Step


@dataclass
class Attempt:
    """One try at a step, as the agents see it."""

    prompt: str
    verdict: str


class Watcher(Protocol):
    name: str
    connected: bool

    def watch(self, step: Step | None, verdict: str, picture: bytes | None) -> str:
        """What actually happened, in words. `picture` is the top camera's last frame, when there is one."""
        ...


class PromptWriter(Protocol):
    name: str
    connected: bool

    def rewrite(self, step: Step, happened: str, history: list[Attempt]) -> str:
        """The next instruction for Dream-1. `step.prompt` is the template sentence; return it to keep it."""
        ...


class RuleWatcher:
    """Today's behavior: the checker's verdict is all we know."""

    name = "Rule-based (the checker's verdict)"
    connected = True

    def watch(self, step, verdict, picture):
        return verdict


class TemplatePrompts:
    """Today's behavior: the template sentence our training demos used."""

    name = "Template (where.py)"
    connected = True

    def rewrite(self, step, happened, history):
        return step.prompt


class NotConnected(RuntimeError):
    """An LLM agent's model isn't wired up yet."""


class _LLMAgent:
    """Shared plumbing for the LLM placeholders: one model call, and a fallback while it isn't connected."""

    connected = False

    def __init__(self, model: str | None = None, log: Callable[[str], None] = lambda _: None):
        self.model = model
        self.log = log
        self._warned = False

    def _ask_model(self, system: str, user: str, picture: bytes | None = None) -> str:
        # TODO: call a model here (a vision-capable LLM for the watcher). Return its text answer.
        # Keep it fast (the arms wait on it) and set `connected = True` on the class once it works.
        raise NotConnected(f"{type(self).__name__} has no model connected yet")

    def _fallback(self, why: Exception) -> None:
        if not self._warned:
            self._warned = True
            self.log(f"AGENT {self.role} isn't connected ({why}): using the rule-based stand-in")


class LLMWatcher(_LLMAgent):
    """PLACEHOLDER: a vision LLM that looks at the top-camera picture after a failed step and says what happened."""

    role = "watcher"

    @property
    def name(self) -> str:
        return f"LLM watcher ({self.model or 'no model'})"

    def watch(self, step, verdict, picture):
        system = ("You watch a robot arm play chess. Given the intended step, the checker's verdict and a top-camera "
                  "picture, say in one short sentence where the piece is now and how it sits (upright, tipped, "
                  "between squares). Name squares the way the board's edge labels do.")
        intended = step.prompt if step is not None else "(no step: checking the board between moves)"
        try:
            return self._ask_model(system, f"Intended: {intended}\nChecker: {verdict}", picture)
        except NotConnected as err:
            self._fallback(err)
            return verdict


class LLMPromptWriter(_LLMAgent):
    """PLACEHOLDER: an LLM that rewrites Dream-1's next instruction from what happened and every attempt so far."""

    role = "prompt writer"

    @property
    def name(self) -> str:
        return f"LLM prompt writer ({self.model or 'no model'})"

    def rewrite(self, step, happened, history):
        system = ("You write one instruction at a time for Dream-1, a vision-language-action model driving two robot "
                  "arms. It can see the board but knows nothing about chess. Keep the instruction short, name the "
                  "arm, the piece's color and type, and describe both squares from the robot's seat. Change the "
                  "wording when earlier attempts failed the same way.")
        tries = "\n".join(f"- {a.prompt} -> {a.verdict}" for a in history) or "- none yet"
        try:
            return self._ask_model(system, f"Template: {step.prompt}\nWhat happened: {happened}\nEarlier:\n{tries}")
        except NotConnected as err:
            self._fallback(err)
            return step.prompt


@dataclass
class AgentChain:
    """Runs a failed step through the watcher and the prompt writer, logging each hand-off."""

    watcher: Watcher = field(default_factory=RuleWatcher)
    writer: PromptWriter = field(default_factory=TemplatePrompts)
    log: Callable[[str], None] = lambda _: None
    picture: Callable[[], bytes | None] = lambda: None  # the top camera's last frame, if there is one
    history: list[Attempt] = field(default_factory=list)

    @property
    def stand_ins_only(self) -> bool:
        return isinstance(self.watcher, RuleWatcher) and isinstance(self.writer, TemplatePrompts)

    def new_goal(self) -> None:
        self.history.clear()

    def failed(self, step: Step | None, verdict: str) -> None:
        if step is not None:
            self.history.append(Attempt(step.prompt, verdict))

    def revise(self, step: Step, verdict: str, failed: Step | None = None) -> Step:
        """The correction to send Dream-1, after the watcher and the prompt writer have had their say.

        With only the stand-ins, this returns `step` unchanged and logs nothing, so the game is exactly as before."""
        if self.stand_ins_only:
            return step
        self.log(f"HANDOFF checker -> watcher: {verdict}")
        happened = self.watcher.watch(failed, verdict, self.picture())
        self.log(f"HANDOFF watcher -> prompt writer: {happened}")
        prompt = self.writer.rewrite(step, happened, list(self.history))
        if prompt != step.prompt:
            self.log("HANDOFF prompt writer -> Dream-1: a reworded instruction")
            return dataclasses.replace(step, prompt=prompt)
        self.log("HANDOFF prompt writer -> Dream-1: the template instruction")
        return step

    def status(self) -> list[dict]:
        return [{"role": "watcher", "name": self.watcher.name, "connected": self.watcher.connected,
                 "placeholder": isinstance(self.watcher, LLMWatcher)},
                {"role": "prompt writer", "name": self.writer.name, "connected": self.writer.connected,
                 "placeholder": isinstance(self.writer, LLMPromptWriter)}]


def make_agents(kind: str = "rules", model: str | None = None, log: Callable[[str], None] = lambda _: None) -> AgentChain:
    """`rules`: today's stand-ins. `llm`: the LLM placeholders (they fall back to the stand-ins until connected)."""
    if kind == "llm":
        return AgentChain(LLMWatcher(model, log), LLMPromptWriter(model, log), log)
    if kind != "rules":
        raise ValueError(f"no such agents: {kind}")
    return AgentChain(log=log)
