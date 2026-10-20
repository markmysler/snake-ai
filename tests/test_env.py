"""Environment tests. Run with `python -m pytest -s tests/`."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from snake_env import (DOWN, LEFT, RIGHT, STATE_SIZE, STRAIGHT, TURN_LEFT,  # noqa: E402
                       TURN_RIGHT, UP, SnakeEnv)


def play_random(env, rng):
    state, done, rewards, n = env.reset(), False, [], 0
    while not done:
        state, r, done, info = env.step(int(rng.integers(3)))
        rewards.append(r)
        n += 1
        assert state.shape == (STATE_SIZE,) and state.dtype == np.float32
        assert len(env.occupied) == len(env.snake), "snake overlaps itself"
        assert len(env.snake) == env.init_length + env.score
    return rewards, info, n


def test_random_games_rewards():
    rng = np.random.default_rng(0)
    for mode, allowed in (("score", {10.0, -10.0, 0.0}), ("shaped", {10.0, -10.0, 0.1, -0.1})):
        env = SnakeEnv(reward=mode, seed=1)
        for _ in range(300):
            rewards, info, n = play_random(env, rng)
            assert set(rewards) <= allowed, (mode, set(rewards))
            assert info["steps"] == n
            assert info["score"] == rewards.count(10.0)
            assert (rewards[-1] == -10.0) == info["collision"]
            assert info["collision"] or info["truncated"] or info["won"]


def test_wall_collision():
    env = SnakeEnv(seed=0)
    env.set_state([(9, 5), (8, 5), (7, 5)], RIGHT, (0, 0))
    _, r, done, info = env.step(STRAIGHT)
    assert done and r == -10.0 and info["collision"] and info["score"] == 0


def test_body_collision():
    env = SnakeEnv(seed=0)
    s = env.set_state([(5, 5), (5, 6), (4, 6), (4, 5), (4, 4)], UP, (0, 9))
    assert s[1] == 1.0, "danger-left flag should be set"
    _, r, done, info = env.step(TURN_LEFT)
    assert done and r == -10.0 and info["collision"]


def test_moving_into_tail_is_legal():
    env = SnakeEnv(seed=0)
    s = env.set_state([(5, 5), (5, 6), (4, 6), (4, 5)], UP, (0, 9))
    assert s[1] == 0.0, "the tail cell is not dangerous"
    _, r, done, _ = env.step(TURN_LEFT)
    assert not done and env.snake[0] == (4, 5) and env.direction == LEFT


def test_eating_grows_snake():
    env = SnakeEnv(seed=0)
    env.set_state([(5, 5), (4, 5), (3, 5)], RIGHT, (6, 5))
    _, r, done, info = env.step(STRAIGHT)
    assert r == 10.0 and not done and info["score"] == 1 and len(env.snake) == 4
    assert env.food not in env.occupied


def test_stall_limit():
    for mode in ("score", "shaped"):
        env = SnakeEnv(reward=mode, seed=0)
        # A length-3 snake turning left forever circles a 2x2 square.
        env.set_state([(5, 5), (5, 6), (4, 6)], UP, (0, 0))
        done, n = False, 0
        while not done:
            _, r, done, info = env.step(TURN_LEFT)
            n += 1
            assert r != -10.0
        assert n == 100 * 3 and info["truncated"] and not info["collision"]


def test_shaped_reward_sign():
    env = SnakeEnv(reward="shaped", seed=0)
    env.set_state([(5, 5), (4, 5), (3, 5)], RIGHT, (9, 5))
    assert env.step(STRAIGHT)[1] == 0.1   # closer
    assert env.step(TURN_RIGHT)[1] == -0.1  # heading down: farther
    assert env.direction == DOWN


def test_seed_reproducible():
    def trajectory(seed):
        env, rng = SnakeEnv(seed=seed), np.random.default_rng(42)
        out = [env.reset().tolist()]
        for _ in range(500):
            s, r, done, _ = env.step(int(rng.integers(3)))
            out.append((s.tolist(), r, env.food))
            if done:
                out.append(env.reset().tolist())
        return out
    assert trajectory(7) == trajectory(7)
    assert trajectory(7) != trajectory(8)


def test_speed():
    env, rng = SnakeEnv(seed=0), np.random.default_rng(0)
    actions = rng.integers(3, size=200_000).tolist()
    env.reset()
    t = time.perf_counter()
    for a in actions:
        if env.step(a)[2]:
            env.reset()
    rate = len(actions) / (time.perf_counter() - t)
    print(f"\nenv speed: {rate:,.0f} random steps/sec")
    assert rate > 20_000
