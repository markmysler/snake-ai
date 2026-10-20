"""Tests for the pieces between trainers and dashboard: replay encoding, publishing, CSV tailing."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dashboard.replay import play_game  # noqa: E402
from dashboard.store import CsvTail  # noqa: E402
from policies import MLPPolicy, mlp_param_count  # noqa: E402
from publisher import Publisher, read_json  # noqa: E402
from train_ga import next_generation  # noqa: E402
from vision import state_size  # noqa: E402


def rebuild(game):
    """Python mirror of the browser's frame reconstruction (board.js advance())."""
    snake, food, eats, frames = list(game["init"]), game["food"], dict(map(tuple, game["eats"])), []
    for i, head in enumerate(game["heads"], start=1):
        snake.insert(0, head)
        if i in eats:
            food = eats[i]
        else:
            snake.pop()
        frames.append((list(snake), food))
    return frames


def test_replay_encoding_round_trip():
    rng = np.random.default_rng(0)
    for seed in range(20):
        vision = ("short", "full")[seed % 2]
        act = MLPPolicy(rng.normal(0, 1, mlp_param_count(state_size(vision)))).act
        game = play_game(act, 10, 10, seed=seed, vision=vision)
        frames = rebuild(game)
        assert len(frames) == len(game["heads"])
        assert len(game["eats"]) == game["score"]
        for snake, food in frames:
            assert len(set(snake)) == len(snake), "snake overlaps itself"
            assert all(0 <= c < 100 for c in snake) and food not in snake


def test_publisher_and_csv_tail(tmp_path):
    live, results = str(tmp_path / "live"), str(tmp_path / "results")
    pub = Publisher("ga", "generation", live, results)
    tail = CsvTail(os.path.join(results, "ga.csv"))
    pub.stats(1, 100, 3, 2, 0.5, extra=1)
    rows, reset = tail.poll()
    assert len(rows) == 1 and rows[0]["total_steps"] == "100" and not reset
    pub.weights(lambda f: np.save(f, np.zeros(3)), "npy", 1)
    meta = read_json(os.path.join(live, "ga_meta.json"))
    assert meta["iteration"] == 1 and os.path.exists(os.path.join(live, meta["file"]))
    pub.stats(2, 250, 6, 4, 1.5)
    assert [r["iteration"] for r in tail.poll()[0]] == ["2"]
    pub.finish()
    assert read_json(os.path.join(live, "ga_stats.json"))["final"] is True
    assert not [n for n in os.listdir(live) if n.endswith(".tmp")]
    Publisher("ga", "generation", live, results).finish()  # a new run truncates the CSV
    rows, reset = tail.poll()
    assert reset and rows == []


def test_next_generation_keeps_elites():
    rng = np.random.default_rng(0)
    pop = rng.normal(size=(50, mlp_param_count(11))).astype(np.float32)
    fit = rng.random(50)
    new = next_generation(pop, fit, rng, 0.1, 3, 0.1, 0.2)
    assert new.shape == pop.shape
    np.testing.assert_array_equal(new[:5], pop[np.argsort(-fit)[:5]])
