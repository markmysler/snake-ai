"""jev player: descriptions, the run (with a fake decision model, no API calls), replays, evaluation and limits."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jev  # noqa: E402
import runconfig  # noqa: E402
import train_jev  # noqa: E402
from dashboard.inspector import Game  # noqa: E402
from dashboard.replay import play_game  # noqa: E402
from evaluate import evaluate  # noqa: E402
from policies import load_policy  # noqa: E402
from snake_env import RIGHT, UP, SnakeEnv  # noqa: E402


def test_descriptions_are_relative_to_the_heading():
    env = SnakeEnv(10, 10, vision="short")
    obs = env.set_state([(5, 5), (5, 6), (5, 7)], UP, (2, 3))  # heading up, apple up-left
    text = jev.describe("short", env, obs)
    assert "straight is safe" in text and "ahead and to the left" in text
    obs = env.set_state([(9, 5), (8, 5), (7, 5)], RIGHT, (2, 5))  # heading right into the wall, apple behind
    text = jev.describe("short", env, obs)
    assert "straight is DEADLY" in text and "The apple is behind." in text
    env = SnakeEnv(10, 10, vision="rays:depth=4")
    obs = env.set_state([(5, 1), (5, 2), (5, 3)], UP, (5, 9))
    assert "ahead: wall 2 square(s) away" in jev.describe("rays:depth=4", env, obs)
    env = SnakeEnv(10, 10, vision="window:radius=2")
    obs = env.set_state([(5, 5), (5, 6), (5, 7)], UP, (5, 4))
    grid = jev.describe("window:radius=2", env, obs).split("\n")
    assert grid[1:6][1][2] == "A" and grid[1:6][2][2] == "H"  # the apple right ahead of the head


def safe_player(boards, prompt="rules"):
    """A fake jev: the first move whose next square is safe (short vision descriptions), else straight."""
    out = {}
    for bid, text in boards.items():
        line = text.split("\n")[0]
        order = [m for m in ("straight", "left", "right") if f"{m} is safe" in line] or ["straight"]
        i = ("left", "straight", "right").index(order[0])
        out[bid] = (i, [0.8 if j == i else 0.1 for j in range(3)])
    return out, 0.001


def run(tmp_path, monkeypatch, *args):
    monkeypatch.setattr(jev, "decide", safe_player)
    monkeypatch.setattr(sys, "argv", ["train_jev.py", "--vision", "short", "--live-dir", str(tmp_path / "live"),
                                      "--results-dir", str(tmp_path / "results"), "--models-dir", str(tmp_path / "models"), *args])
    train_jev.main()
    return str(tmp_path / "models" / "jev_best.json")


def test_a_run_plays_the_test_games_and_replays_exactly(tmp_path, monkeypatch):
    path = run(tmp_path, monkeypatch, "--games", "3", "--max-cost", "5")
    data = json.load(open(path))
    assert len(data["games"]) == 3 and data["unfinished"] == [] and data["cost_usd"] > 0
    ev = evaluate(path)
    assert ev["games"] == 3 and sum(ev["outcomes"].values()) == 3  # its own games are its test
    act, meta = load_policy(path)
    policy = act.__self__
    for g in data["games"]:
        shown = play_game(policy.for_seed(g["seed"]).act, 10, 10, g["seed"], meta["vision"])
        assert (shown["score"], shown["steps"]) == (g["score"], g["steps"])
        frame = Game(policy.for_seed(g["seed"]).act, meta, g["seed"]).frame(1)
        assert frame["action"] == g["actions"][1] and max(frame["out"]) == pytest.approx(-0.2231, abs=1e-3)  # log 0.8
        last = Game(policy.for_seed(g["seed"]).act, meta, g["seed"]).frame(g["steps"])  # after its last move
        assert last["out"] is None and last["end"] is not None


def test_spending_cap_stops_the_run(tmp_path, monkeypatch):
    path = run(tmp_path, monkeypatch, "--games", "2", "--max-cost", "0.003")
    data = json.load(open(path)) if os.path.exists(path) else None
    assert data is None or data["cost_usd"] <= 0.003 + 0.001


def test_recipe_checks(monkeypatch):
    cfg = runconfig.normalize({"agents": ["jev"], "time_budget_min": 10})
    monkeypatch.setattr(jev, "configured", lambda: False)
    errors, warnings = runconfig.validate(cfg)
    assert any("TOKENROUTER_API_KEY" in e for e in errors) and any("time limit" in w for w in warnings)
    monkeypatch.setattr(jev, "configured", lambda: True)
    assert not runconfig.validate({**cfg, "time_budget_min": 0})[1]
    assert jev.estimate(20)["usd"] < 0.1


def test_an_outage_is_waited_out_and_unfinished_games_are_kept(tmp_path, monkeypatch):
    calls = {"n": 0}

    def flaky(boards, prompt="rules"):
        calls["n"] += 1
        if calls["n"] in (3, 4):  # the API stops answering for a while
            raise jev.JevError("Could not reach jev: The read operation timed out")
        return safe_player(boards, prompt)

    monkeypatch.setattr(train_jev.time, "sleep", lambda s: None)
    monkeypatch.setattr(jev, "decide", flaky)
    monkeypatch.setattr(sys, "argv", ["train_jev.py", "--games", "2", "--live-dir", str(tmp_path / "live"),
                                      "--results-dir", str(tmp_path / "results"), "--models-dir", str(tmp_path / "models")])
    train_jev.main()
    data = json.load(open(tmp_path / "models" / "jev_best.json"))
    assert len(data["games"]) == 2 and data["stopped"] is None  # the outage didn't end the run

    def dead(boards, prompt="rules"):
        raise jev.JevError("jev request failed (401): bad key")

    monkeypatch.setattr(jev, "decide", dead)
    monkeypatch.setattr(sys, "argv", [*sys.argv[:-2], "--models-dir", str(tmp_path / "models2")])
    train_jev.main()  # a refused request ends the run at once, with nothing played
    assert not (tmp_path / "models2" / "jev_best.json").exists()
