"""Separate wall / body crash penalties, the room-to-move vision, and endings recorded during training."""
import csv
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import rewards  # noqa: E402
import vision as visions  # noqa: E402
from dashboard.store import HISTORY_KEYS, history_row  # noqa: E402
from snake_env import RIGHT, STRAIGHT, TURN_RIGHT, UP, SnakeEnv  # noqa: E402


def crash(env, snake, direction, action):
    env.set_state(snake, direction, (0, 0))
    _, r, done, info = env.step(action)
    assert done and info["collision"]
    return r, info["crash"]


def test_crash_penalties_by_kind():
    env = SnakeEnv(10, 10, reward={"apple": 10, "death_wall": 5, "death_body": 30})
    wall = crash(env, [(9, 5), (8, 5), (7, 5)], RIGHT, STRAIGHT)
    body = crash(env, [(5, 5), (4, 5), (4, 6), (5, 6), (6, 6)], RIGHT, TURN_RIGHT)
    assert wall == (-5.0, "wall") and body == (-30.0, "body")
    env = SnakeEnv(10, 10, reward="score")  # the single death penalty is unchanged
    assert crash(env, [(9, 5), (8, 5), (7, 5)], RIGHT, STRAIGHT)[0] == -10.0


def test_split_penalties_cant_mix_with_death():
    errors, _ = rewards.validate({"apple": 10, "death": 10, "death_body": 30})
    assert errors
    assert rewards.validate("split_death") == ([], [])
    assert any("loop" in w for w in rewards.validate({"apple": 10, "death_wall": 10, "death_body": 30})[1])
    assert not rewards.validate("score")[1], "the default reward stays warning-free"
    assert rewards.describe({"apple": 10, "death_body": 30}) == "apple +10, death_body -30"
    feats = {f["name"]: f for f in rewards.catalog()["features"]}
    assert feats["death_wall"]["sign"] == -1 and feats["apple"]["sign"] == 1


def test_room_vision_sees_pockets():
    assert visions.state_size("room") == 14
    env = SnakeEnv(10, 10, vision="room")
    # Heading up next to the left wall; to its right the body seals a 2-square pocket {(2,5), (2,6)}.
    env.set_state([(1, 5), (1, 6), (1, 7), (2, 7), (3, 7), (3, 6), (3, 5), (3, 4), (2, 4), (2, 3), (3, 3)], UP, (8, 8))
    s = env.observe(env)
    assert s.shape == (14,) and np.allclose(s[:11], visions.short_view(env))
    straight, left, right = s[11:]
    free = 100 - 10  # the tail square counts as free: it moves away
    assert straight == pytest.approx(left) and straight > 0.9, "both lead to the open board"
    assert right == pytest.approx(2 / free), "the pocket is 2 of the free squares"
    assert s[2] == 0.0, "the pocket is safe for now: no danger signal, only the room number warns"
    open_env = SnakeEnv(10, 10, vision="room")
    open_env.reset(seed=0)
    assert open_env.observe(open_env)[11] == pytest.approx(1.0), "early on, every square is reachable"


def test_history_rows_without_endings_load():
    old = {k: "1" for k in HISTORY_KEYS if not k.startswith("end_")}
    assert history_row(old)[-4:] == [None] * 4
    assert history_row({**old, "end_wall": "3", "end_body": "97", "end_starved": "0", "end_won": "0"})[-4:] == [3, 97, 0, 0]


@pytest.mark.parametrize("script,extra", [("train_ga.py", ["--generations", "2", "--population", "20", "--workers", "1"]),
                                          ("train_dqn.py", ["--episodes", "30"])])
def test_trainers_record_endings(tmp_path, script, extra):
    subprocess.run([sys.executable, script, "--vision", "room", "--reward", "split_death", "--live-dir", str(tmp_path / "live"),
                    "--results-dir", str(tmp_path / "results"), "--models-dir", str(tmp_path / "models"), *extra],
                   cwd=ROOT, check=True, capture_output=True, timeout=300)
    agent = script[6:-3]
    with open(tmp_path / "results" / f"{agent}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    ends = [int(last[f"end_{k}"]) for k in ("wall", "body", "starved", "won")]
    assert sum(ends) == min(100, int(last["total_games"]))


def reference_room(env):
    """Plain flood fill with sets, to check the fast version against."""
    from vision import DIRS
    blocked = set(env.occupied) - {env.snake[-1]}
    free = env.width * env.height - len(blocked)
    hx, hy = env.snake[0]
    out = []
    for turn in (0, -1, 1):
        dx, dy = DIRS[(env.direction + turn) % 4]
        start = (hx + dx, hy + dy)
        if env.is_danger(*start):
            out.append(0.0)
            continue
        seen, stack = {start}, [start]
        while stack:
            x, y = stack.pop()
            for ddx, ddy in DIRS:
                n = (x + ddx, y + ddy)
                if 0 <= n[0] < env.width and 0 <= n[1] < env.height and n not in blocked and n not in seen:
                    seen.add(n)
                    stack.append(n)
        out.append(len(seen) / free)
    return out


@pytest.mark.parametrize("width,height", [(10, 10), (7, 12)])
def test_room_matches_a_plain_flood_fill(width, height):
    env = SnakeEnv(width, height, vision="room")
    rng = np.random.default_rng(1)
    env.reset(seed=0)
    for n in range(3000):
        assert np.allclose(visions.room_shares(env), reference_room(env)), n
        # mostly-greedy play toward the apple so snakes grow and make pockets
        _, _, done, _ = env.step(int(rng.integers(3)) if rng.random() < 0.3 else STRAIGHT if not env.observe(env)[0] else TURN_RIGHT)
        if done:
            env.reset(seed=n)
