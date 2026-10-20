"""Reward features: each one does what its description says; conflicts are caught."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rewards  # noqa: E402
from snake_env import RIGHT, STRAIGHT, TURN_LEFT, TURN_RIGHT, UP, SnakeEnv  # noqa: E402


def env_with(spec, **kw):
    return SnakeEnv(reward=spec, seed=0, **kw)


def test_parse_presets_strings_and_dicts():
    assert rewards.parse("score") == {"apple": 10.0, "death": 10.0}
    assert rewards.parse("apple=5,hunger=0.01") == {"apple": 5.0, "hunger": 0.01}
    assert rewards.parse({"apple": 1}) == {"apple": 1.0}
    assert rewards.describe("shaped") == "shaped"
    assert rewards.parse(rewards.to_string("apple=3,starve=2")) == {"apple": 3.0, "starve": 2.0}
    with pytest.raises(ValueError):
        rewards.parse("apple=1,teleport=2")


def test_incompatible_features_are_errors():
    errors, _ = rewards.validate("apple=10,distance=0.1,potential=0.1")
    assert len(errors) == 1 and "can't be used together" in errors[0]
    errors, _ = rewards.validate("apple=10,hunger=0.01,survival=0.1")
    assert len(errors) == 1
    errors, warnings = rewards.validate("survival=0.1")
    assert not errors and len(warnings) == 2  # no apple reward; survival without starvation


def test_each_feature():
    e = env_with("apple=10,death=3,hunger=0.5")
    e.set_state([(5, 5), (4, 5), (3, 5)], RIGHT, (6, 5))
    assert e.step(STRAIGHT)[1] == pytest.approx(10 - 0.5)        # apple + hunger
    e.set_state([(9, 5), (8, 5), (7, 5)], RIGHT, (0, 0))
    assert e.step(STRAIGHT)[1] == pytest.approx(-3 - 0.5)        # death + hunger

    e = env_with("survival=0.2,starve=7")
    e.set_state([(5, 5), (5, 6), (4, 6)], UP, (0, 0))  # loops in a 2x2 square until it starves
    rs = [e.step(TURN_LEFT)[1] for _ in range(300)]
    assert rs[:-1] == [pytest.approx(0.2)] * 299 and rs[-1] == pytest.approx(0.2 - 7)

    e = env_with("distance=0.5")
    e.set_state([(5, 5), (4, 5), (3, 5)], RIGHT, (9, 5))
    assert e.step(STRAIGHT)[1] == 0.5 and e.step(TURN_RIGHT)[1] == -0.5

    e = env_with("potential=1")
    e.set_state([(5, 5), (4, 5), (3, 5)], RIGHT, (9, 5))  # distance 4 -> 3
    assert e.step(STRAIGHT)[1] == pytest.approx(0.95 * -3 + 4)


def test_win_bonus():
    e = env_with("apple=1,win=50", width=5, height=5)
    path = [(x, y) for y in range(5) for x in (range(5) if y % 2 == 0 else range(4, -1, -1))]
    snake = path[-2::-1]  # fills every square except the last one on the path, head next to it
    (hx, hy), (fx, fy) = snake[0], path[-1]
    heading = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}[(fx - hx, fy - hy)]
    e.set_state(snake, heading, path[-1])
    _, r, done, info = e.step(STRAIGHT)
    assert done and info["won"] and r == 51


def test_ga_classic_matches_old_fitness():
    e, rng = env_with("ga_classic"), np.random.default_rng(0)
    for _ in range(50):
        e.reset()
        total, done = 0.0, False
        while not done:
            _, r, done, info = e.step(int(rng.integers(3)))
            total += r
        assert total == pytest.approx(info["score"] * 100 + info["steps"] * 0.1)
