"""Start a Dream-1 model on DreamAPI before the robot needs it, and keep it running. The arms are never involved.

    ~/dream-env/bin/python tools/warm_dream.py <checkpoint> <status.json> [--parent <pid>]

A model that isn't running on DreamAPI starts on its first request and refuses requests (`pool_warming`) until it's
up, which for a fine-tune nobody has run lately can take up to 15 minutes. The station only asks once a robot move
begins, so without this the first move waits out the whole start. This sends that first request instead, as soon as
the app starts: from a client of its own, with blank camera frames and a resting pose, throwing the answer away. Once
the model answers, it asks again every few minutes so it stays running, until the app (`--parent`) exits.

It writes what it knows to <status.json> for the app's page: {"state": "warming" | "ready" | "error", ...}.
Runs with the Dream library's Python (~/dream-env), not the app's.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path

KEEP_WARM_S = 180.0  # how often to ask a running model again, so it isn't shut down for being idle


def write(path: Path, **status) -> None:
    status["checked_at"] = time.time()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status))
    tmp.replace(path)  # whole, so the app never reads half a file


def blank_request(client, checkpoint: str) -> None:
    """Hand the client one observation of the shape the model reads: blank frames, a resting pose, a short prompt."""
    import cv2
    import numpy as np

    jpeg = base64.b64encode(cv2.imencode(".jpg", np.zeros((224, 224, 3), np.uint8))[1].tobytes()).decode()
    model = next((m for m in client.get_models() if m.id == checkpoint), None)
    if model is None:
        raise LookupError(f"{checkpoint} isn't one of this account's models")
    config = model.embodiment_config
    if config:  # a fine-tune: its own cameras and state parts
        roles = list(config["video"])
        state = {f"state.{part}": [0.0] * (end - start) for part, (start, end) in config["state"].items()}
    else:  # a Dream-1 base model on a YAM, which also reads each hand's pose
        roles = ["overhead", "wrist_l", "wrist_r"]
        state = {"state.right_arm": [0.0] * 6, "state.right_gripper": [0.0],
                 "state.left_arm": [0.0] * 6, "state.left_gripper": [0.0]}
        for side in ("left", "right"):
            state[f"state.{side}_eef_xyz"] = [0.3, 0.0, 0.2]
            state[f"state.{side}_eef_rot6"] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            state[f"state.{side}_grip"] = [0.0]
    client.set_observation({f"video.{role}": jpeg for role in roles}, {**state, "prompt": "hold still"})


def alive(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("status", type=Path)
    parser.add_argument("--parent", type=int, help="stop once this process (the app) exits")
    args = parser.parse_args()

    import dream

    started = time.time()
    write(args.status, state="warming", checkpoint=args.checkpoint, since=started, message="asking DreamAPI")
    try:
        client = dream.DreamClient()
        client.policy({"checkpoint": args.checkpoint})
        blank_request(client, args.checkpoint)
    except Exception as err:  # noqa: BLE001  (anything at all: say it on the page, don't crash the app)
        write(args.status, state="error", checkpoint=args.checkpoint, since=started, message=f"{type(err).__name__}: {err}")
        return
    ready_at = None
    while alive(args.parent):
        try:
            client.get_action()
            if ready_at is None:
                ready_at = time.time()
            write(args.status, state="ready", checkpoint=args.checkpoint, since=started, ready_at=ready_at,
                  message="running on DreamAPI")
            wait = KEEP_WARM_S
        except Exception as err:  # noqa: BLE001
            if getattr(err, "reason", None) == "pool_warming":
                ready_at = None
                write(args.status, state="warming", checkpoint=args.checkpoint, since=started,
                      message="the model is starting on DreamAPI; a cold start can take up to 15 minutes")
                wait = float(getattr(err, "retry_after_s", None) or 30.0)
            else:
                write(args.status, state="error", checkpoint=args.checkpoint, since=started,
                      message=f"{type(err).__name__}: {str(err)[:300]}")
                wait = 60.0
        client.reset_episode()  # each ask is a fresh episode: one observation, nothing carried over
        blank_request(client, args.checkpoint)
        end = time.time() + wait
        while time.time() < end and alive(args.parent):
            time.sleep(1.0)


if __name__ == "__main__":
    main()
