"""Inspect mode: every vision explains itself with valid drawing primitives, and recorded games replay exactly."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import vision as visions  # noqa: E402
from dashboard.inspector import Game, describe  # noqa: E402
from dashboard.replay import play_game  # noqa: E402
from policies import MLPPolicy, mlp_param_count  # noqa: E402
from snake_env import SnakeEnv  # noqa: E402

SPECS = ["short", "room", "rays:depth=4", "rays:depth=9", "window:radius=2", "full"]


def random_positions(spec, n=40):
    """Positions from random play (several snake lengths, near walls and in the middle)."""
    rng = np.random.default_rng(1)
    env = SnakeEnv(10, 10, vision=spec, seed=3)
    obs = env.reset()
    for _ in range(n):
        for _ in range(int(rng.integers(1, 6))):
            obs, _, done, _ = env.step(int(rng.integers(3)))
            if done:
                obs = env.reset(seed=int(rng.integers(1 << 30)))
        yield env, obs


@pytest.mark.parametrize("spec", SPECS)
def test_explain_gives_valid_primitives(spec):
    ok = lambda x, y: -1 <= x <= 11 and -1 <= y <= 11  # noqa: E731  (one square past the edge: the wall)
    for env, obs in random_positions(spec):
        p = visions.explain(spec, env, obs)
        assert set(p) == {"cells", "lines", "marks", "boxes", "text"} and p["text"]
        assert all(0 <= c["x"] < 10 and 0 <= c["y"] < 10 for c in p["cells"])
        assert all(ok(*l["from"]) and ok(*l["to"]) for l in p["lines"])
        assert all(ok(*m["at"]) for m in p["marks"])
        hx, hy = env.snake[0]  # boxes (the window) may reach past the board: they surround the head
        assert all(b["x0"] <= hx < b["x1"] and b["y0"] <= hy < b["y1"] for b in p["boxes"])
        json.dumps(p)  # sent as JSON
        names, pic = visions.input_names(spec), visions.picture_info(spec)
        if names:
            assert len(names) == len(obs)
        else:
            assert len(pic["channels"]) * pic["side"] ** 2 + len(pic["extra"]) == len(obs)


def test_explain_matches_the_inputs():
    for env, obs in random_positions("room"):
        p = visions.explain("room", env, obs)
        crosses = [l for l in p["lines"] if l["tone"] in ("danger", "safe")]
        assert [l["tone"] == "danger" for l in crosses] == [bool(v) for v in obs[:3]]
        shares = [m["label"] for m in p["marks"]]
        assert len(shares) == sum(1 for v in obs[:3] if not v)  # one share per move that does not crash


@pytest.mark.parametrize("spec", ["short", "rays:depth=4", "full"])
def test_recorded_game_replays_the_viewing_game(spec):
    rng = np.random.default_rng(0)
    n_in = visions.state_size(spec)
    act = MLPPolicy(rng.normal(0, 0.5, mlp_param_count(n_in)).astype(np.float32)).act
    meta = {"vision": visions.parse(spec), "width": 10, "height": 10}
    shown = play_game(act, 10, 10, 1234, spec)
    g = Game(act, meta, 1234)
    assert len(g.actions) == shown["steps"]
    for step in [0, *range(1, len(shown["heads"]) + 1, 7), 2]:
        f = g.frame(step)
        if step:
            x, y = f["snake"][0]
            assert y * 10 + x == shown["heads"][step - 1]
    size = len(json.dumps({**g.frame(1), "info": describe(meta, "ga")}))
    assert size < (12_000 if spec == "full" else 6_000)
