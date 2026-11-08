# Runbook: the robot and the app, at the station

From arriving at the station to a game on screen. Nothing here has been run on the real arms yet, so the
first real game is also the first test of the base model through our pipeline.

## 1. Connect and update

On your laptop (Git Bash), log in to the NUC with the app's page forwarded:

```bash
ssh -L 8800:127.0.0.1:8800 dream-nuc
```

Then, on the NUC:

```bash
cd ~/dream-chess
git pull
```

## 2. Check nothing else holds the arms

Only one program can drive the arms at a time. Stop `dhk-collect` if it's running (Ctrl+C in its
window). The Dream station must be **running and idle**, because the app sends each move through it:

```bash
curl -s http://127.0.0.1:8091/status | python3 -m json.tool | head -5
```

You want `"state": "idle"`. If nothing answers, the station isn't running. Ask whoever runs it to start it.

## 3. Rehearse first: nothing moves

```bash
.venv/bin/python app.py --sim
```

Open http://localhost:8800 on your laptop. You play White from the page, and a simulated robot plays
Brown. Press Ctrl+C on the NUC to stop it.

## 4. Play the real robot

Set up the board with brown nearest the robot (rank 8 closest, the h-file at the robot's left), a tray
spot beside each arm, and **a person with a hand on the e-stop**. Then:

```bash
.venv/bin/python app.py                                    # the base model (dreamlabs/dream-1-beta)
.venv/bin/python app.py --checkpoint "dream-1-ft:..."      # or a fine-tuned model
```

The station keeps whichever model it ran last, so the app always names the one it wants: with no
`--checkpoint`, that's the base model.

**A model nobody has run lately takes a while to start on DreamAPI, up to 15 minutes for a fine-tune.** The app
starts it as soon as it launches (`tools/warm_dream.py`, from a client of its own that never touches the arms), and
"The robot's brain" shows it warming up with a timer. Set up the board meanwhile. If the robot's move begins before
the model is ready, the arms wait without a time limit, and once Dream-1 answers the app starts the step again by
itself, so nobody has to say whether a piece landed when nothing moved.

The page asks for everything the terminal used to: confirm the board is set up, confirm the e-stop,
make your move (type it, or click a piece and then its square), and, after each robot step, whether
the piece landed. **Stop the arms** on the page, or Ctrl+C on the NUC, stops the arms at once.

"The robot's eyes" shows the frames the station hands Dream-1 (from `~/.dreamlabs/last-run_inputs`; change it
with `--frames`), and "The robot's brain" shows the station's status. If the brain says *Station offline*, the
station isn't running. The board itself still comes from your answers until the station shares its camera.

For a short first test, start from a mate in two with `--fen "<position>"`.

## 5. Notes

- Stockfish isn't installed on the NUC, so the robot plays the weaker built-in engine until someone
  with the admin password runs `sudo apt install stockfish`. The page shows which engine is in use.
- Every game is logged to `games/`, ending with a summary of moves, Dream-1 attempts, fixes, and
  rescues.
- A fine-tuned model's first run can take up to an hour to start on Dream's side. Start it early.
