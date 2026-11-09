# Next steps: what's still missing

After the hackathon (Nov 7-8), the robot didn't complete a move in any of its 10 live runs (see `AUDIT.md`). The
lesson: **teamwork first, plan second, build third, collaborate and tune last.** This is the list of what the
project still needs, in that order. Parts marked **placeholder** exist in the code as a slot with a stand-in, so the
app runs today and each piece can be filled in on its own.

## 1. Plan together, then measure the baseline first

- **Run the baseline before building anything else.** `docs/EVAL.md` describes 28 trials of the base model, and they
  were never run (`evals/results/` is empty). Start even simpler: type plain instructions into Dream's live view
  ("pick up the brown rook and put it on the square in front of it") and record what the base model can and can't
  do. Every later choice (what agents to build, what data to collect) should come from those results.
- **Test the fine-tunes the same way.** `chess-1k` and `chess-2k` (58 cleaned episodes) never ran on the robot.
  Compare them with the base model on the same in-distribution moves, several tries each (`app.py --script 2`).
- **Fill the gaps in the training data.** Few rook moves, few long or far-rank moves, no castling, no physical tray.

## 2. Move to a new NUC and a different control loop

The hackathon NUC is being wiped, and the next YAM station runs a different control loop. Everything chess-related
(the orchestrator, Stockfish, the prompts, the agents, the page, the simulation) runs anywhere with
`python app.py --sim`. What's tied to the old station:

| Part | Tied to | Where |
|---|---|---|
| **Sending a step to the arms** (How) | Dream's station API on `127.0.0.1:8091` (`start`, `/prompt`, `stop`, `home`) | `dreamchess/station_mover.py` |
| **Starting the model before the first move** | That API, plus the Dream library's Python in `~/dream-env` | `tools/warm_dream.py`, `--dream-python` |
| **The robot's eyes and the checker's pictures** | Frames the station saves to `~/.dreamlabs/last-run_inputs` | `dreamchess/app_vision.py`, `dreamchess/station_camera.py`, `--frames` |
| **The supervisor's hand tracking** | Joint states in `inputs.jsonl` there, the arm model (MuJoCo) in `~/dream-env`, and `~/.dreamlabs/station.yaml` | `tools/hand_watch.py`, `tools/fit_board.py`, `dreamchess/hands.py` |
| **Saved calibrations** | `~/.dream-chess/board_fit.json` (fitted to game 2's demos), `~/.dream-chess/top-calibration.npz` | `--board-fit`, `--top-calibration` |
| **Data collection** | The old NUC's dataset paths | `collect_watch.py` |

The seam for a new control loop is the `Mover` interface in `dreamchess/core.py`: `execute(step)` makes one
pick-and-place from `step.prompt`, and `go_home()` clears the top camera's view. Write one adapter for the new loop
(as `station_mover.py` is for Dream's station, and `dream_mover.py` is for calling DreamAPI directly), and pass it in
`run_robot.play()`. The cameras and hand tracking need their own adapters the same way. Re-fit the board and
re-calibrate the top camera on the new station: the saved files belong to the old board position.

## 3. The LLM agents (placeholders)

The agents live in `dreamchess/agents.py`. After a failed check, the orchestrator hands the failure down a chain:
checker -> watcher -> prompt writer -> Dream-1. Each hand-off is logged as a `HANDOFF` line, shown on the page over
the robot's eyes and in the agent chain. Choose the agents with `--agents`:

| Agent | Today's stand-in (`--agents rules`, default) | Placeholder (`--agents llm`) | To fill it in |
|---|---|---|---|
| **Watcher** | `RuleWatcher`: passes the checker's verdict through | `LLMWatcher` | Implement `_ask_model` with a vision LLM. It gets the step, the verdict and the top camera's last picture, and says where the piece is and how it sits |
| **Prompt writer** | `TemplatePrompts`: keeps the template sentence from `where.py` | `LLMPromptWriter` | Implement `_ask_model`. It gets the template, what happened, and every attempt at this goal, and returns the next instruction |

Until `_ask_model` is filled in, the placeholders say once that they aren't connected and fall back to the
stand-ins, so a game plays exactly as with `--agents rules` (tested in `tests/test_agents.py`). Set `connected = True`
on the class once it works; the page then shows it as connected instead of dashed. `--agent-model` names the model.

Keep them fast: the arms wait on every call.

## 4. Cameras (blocked on the station)

- **Reading your move from the camera.** The station holds all three cameras and saves frames only while Dream-1 is
  moving, so today you type your move. The fix is the station's camera plug-in (a camera source in
  `station.yaml`), which needs the organizers' approval and a station restart.
- **Top-camera calibration** has never finished on the real camera (640x360 frames). Calibrate it and check the
  read-back before trusting the checker.

## 5. The voice

The LiveKit voice made no sound in the demo, and the cause isn't known. The page now plays it from an **Audio** switch
on the robot's avatar: the switch is the click browsers require before playing sound, and it now starts the audio
inside that click. Still to check on the NUC with `--voice livekit`: that the robot's voice agent actually publishes
audio into the room, and that the page hears it.

## 6. The page

Done after the hackathon: the landing page is the robot's eyes (with the agents' hand-offs over them) and the game
(the robot's move marked on the board, avatars for both players, the voice on the robot's avatar), then connected
services and metrics, the agent chain, the top camera's view, and the activity log. Stats and every instruction sent
to Dream-1 are on `/stats`.

Still to do: show the real agent chain once the LLM agents are connected, and the station's live video (Dream's live
view streams it; our page shows the frames handed to Dream-1, about one a second).
