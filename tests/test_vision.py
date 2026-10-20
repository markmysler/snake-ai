"""Vision tests: the full view must be centred on the head and rotated so 'ahead' is always up."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from snake_env import DIRS, DOWN, LEFT, RIGHT, UP, SnakeEnv  # noqa: E402
from vision import full_radius, state_size  # noqa: E402


def view(env):
    side = 2 * full_radius(env.width, env.height) + 1
    return env.observe(env).reshape(3, side, side), side // 2


def place(env, head, direction, food):
    """Length-3 snake with its body straight behind the head."""
    dx, dy = DIRS[direction]
    snake = [(head[0] - i * dx, head[1] - i * dy) for i in range(3)]
    return env.set_state(snake, direction, food)


def test_shape_and_range():
    for w, h in ((10, 10), (12, 8)):
        env = SnakeEnv(w, h, vision="full", seed=0)
        s = env.reset()
        assert s.shape == (state_size("full", w, h),) == (env.state_size,)
        assert s.dtype == np.float32 and s.min() >= 0 and s.max() <= 1


def test_food_ahead_is_always_up_and_left_is_always_left():
    env = SnakeEnv(vision="full", seed=0)
    for d in (UP, RIGHT, DOWN, LEFT):
        dx, dy = DIRS[d]
        lx, ly = DIRS[(d - 1) % 4]  # the snake's left
        place(env, (5, 5), d, (5 + 2 * dx, 5 + 2 * dy))
        v, c = view(env)
        assert v[2, c - 2, c] == 1.0, f"food ahead not above centre for heading {d}"
        place(env, (5, 5), d, (5 + 3 * lx, 5 + 3 * ly))
        v, c = view(env)
        assert v[2, c, c - 3] == 1.0, f"food on the left not left of centre for heading {d}"


def test_walls_body_and_tail():
    env = SnakeEnv(vision="full", seed=0)
    place(env, (0, 5), UP, (9, 9))  # head on the left edge, facing up
    v, c = view(env)
    assert v[0, c, c - 1] == 1.0, "wall on the snake's left"
    assert v[0, c, c + 1] == 0.0, "free cell on the snake's right"
    assert v[0, c + 1, c] == 1.0 and v[1, c + 1, c] == 2 / 3, "neck is blocked, 2 of 3 moves left"
    assert v[0, c + 2, c] == 0.0 and v[1, c + 2, c] == 1 / 3, "tail moves away: not blocked"


def test_full_vision_game_runs_and_speed():
    env, rng = SnakeEnv(vision="full", seed=0), np.random.default_rng(0)
    env.reset()
    t, n = time.perf_counter(), 20_000
    for a in rng.integers(3, size=n).tolist():
        if env.step(a)[2]:
            env.reset()
    rate = n / (time.perf_counter() - t)
    print(f"\nfull-vision env speed: {rate:,.0f} steps/sec")
    assert rate > 5_000


def test_specs_parse_and_size():
    import vision
    assert vision.parse("window:radius=3") == {"type": "window", "radius": 3}
    assert vision.parse({"type": "rays"}) == {"type": "rays", "depth": 4}
    assert vision.to_string({"type": "window", "radius": 1}) == "window:radius=1"
    assert vision.state_size("window:radius=2") == 3 * 25 + 4
    assert vision.state_size("full", 10, 10) == 3 * 19 * 19
    for spec in ("short", "full", "rays:depth=7", "window:radius=3"):
        env = SnakeEnv(vision=spec, seed=1)
        s = env.reset()
        assert s.shape == (env.state_size,) and s.min() >= 0 and s.max() <= 1, spec
    for bad in ("window:radius=0", "rays:depth=99", "eyes", "window:size=3"):
        try:
            vision.parse(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should be rejected")


def test_window_is_egocentric_and_crops():
    env = SnakeEnv(vision="window:radius=2", seed=0)
    for d in (UP, RIGHT, DOWN, LEFT):
        dx, dy = DIRS[d]
        place(env, (5, 5), d, (5 + 2 * dx, 5 + 2 * dy))
        s = env.observe(env)
        pic = s[:75].reshape(3, 5, 5)
        assert pic[2, 0, 2] == 1.0, "food 2 ahead -> top row, centre column"
        assert list(s[75:]) == [1, 0, 0, 0], "food flags say: ahead"
    place(env, (5, 5), UP, (5, 9))  # food far behind: outside the window, only the flags see it
    s = env.observe(env)
    assert s[50:75].sum() == 0 and list(s[75:]) == [0, 1, 0, 0]


def test_rays_see_walls_and_food():
    env = SnakeEnv(vision="rays:depth=4", seed=0)
    place(env, (5, 1), UP, (7, 1))  # wall 2 squares ahead, food 2 squares to the right
    s = env.observe(env)
    assert s[0] == (4 - 2 + 1) / 4, "wall ahead at distance 2"
    assert s[2 * 3 + 2] == (4 - 2 + 1) / 4, "food on the right ray at distance 2"
    assert s[4 * 3 + 1] == 1.0, "neck directly behind"
