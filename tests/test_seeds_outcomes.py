"""How test games end (wall / body / starved / won) and seed groups (one recipe, several seeds)."""
import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import experiment  # noqa: E402
import lessons  # noqa: E402
import runconfig  # noqa: E402
from evaluate import evaluate  # noqa: E402
from policies import mlp_param_count  # noqa: E402
from publisher import Publisher  # noqa: E402
from snake_env import RIGHT, TURN_RIGHT, SnakeEnv  # noqa: E402

STRAIGHT = 1


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setattr(lessons, "LESSONS_DIR", str(tmp_path))
    monkeypatch.setattr(lessons, "LIVE_ROOT", str(tmp_path / "live"))
    monkeypatch.setenv("SNAKE_AI_LESSONS", str(tmp_path))
    return tmp_path


def test_crash_says_wall_or_body():
    env = SnakeEnv(10, 10, seed=0)
    env.reset(seed=0)
    done = False
    while not done:
        _, _, done, info = env.step(STRAIGHT)
    assert info["collision"] and info["crash"] == "wall"
    # A snake heading right whose body wraps around below its head: turning right runs into it.
    env.set_state([(5, 5), (4, 5), (4, 6), (5, 6), (6, 6)], RIGHT, (0, 0))
    _, _, done, info = env.step(TURN_RIGHT)
    assert done and info["collision"] and info["crash"] == "body"


def save_ga(run_dir, seed=0):
    genome = np.random.default_rng(seed).normal(size=mlp_param_count(11)).astype(np.float32)
    np.save(os.path.join(run_dir, "models", "ga_best.npy"), genome)
    lessons.write_json(os.path.join(run_dir, "models", "ga_best.json"),
                       {"generation": 1, "hidden": 16, "width": 10, "height": 10, "vision": "short"})


def test_evaluation_records_every_game(lab):
    run_dir = lessons.new_run(runconfig.normalize({"lesson": "x", "agents": ["ga"]}), status="running")
    save_ga(run_dir)
    ev = evaluate(os.path.join(run_dir, "models", "ga_best.npy"), games=20)
    assert sum(ev["outcomes"].values()) == 20 and len(ev["per_game"]["end"]) == 20
    assert ev["mean"] == pytest.approx(np.mean(ev["per_game"]["score"]))
    assert ev["mean_steps"] == pytest.approx(np.mean(ev["per_game"]["steps"]))


def test_backfill_adds_outcomes_without_changing_scores(lab):
    run_dir = lessons.new_run(runconfig.normalize({"lesson": "x", "agents": ["ga"]}), status="running")
    save_ga(run_dir)
    pub = Publisher("ga", "generation", os.path.join(run_dir, "live"), os.path.join(run_dir, "results"))
    pub.stats(1, 500, 6, 2, 0.5)
    pub.finish()
    info = lessons.finish_run(run_dir, eval_games=10)
    old = {k: info["evaluation"]["ga"][k] for k in ("games", "mean", "median", "max", "mean_steps")}
    lessons.update_run(run_dir, evaluation={"ga": old})  # as recorded before outcomes existed
    assert lessons.backfill(verbose=False) == ["x"]
    ev = lessons.read_run(run_dir)["evaluation"]["ga"]
    assert ev["outcomes"] and all(ev[k] == old[k] for k in old)
    assert "wall/body/starved/won" in (lab / "x" / "LESSON.md").read_text(encoding="utf-8")
    assert lessons.backfill(verbose=False) == [], "nothing left to do"


def test_seed_group_spread_and_lesson(lab):
    config = runconfig.normalize({"lesson": "s", "agents": ["ga"], "seed": 5})
    dirs = lessons.new_group(config, 3, status="running")
    infos = [lessons.read_run(d) for d in dirs]
    assert [i["config"]["seed"] for i in infos] == [5, 6, 7]
    assert {i["group"]["id"] for i in infos} == {infos[0]["id"]} and infos[2]["group"]["index"] == 2
    for d, mean in zip(dirs, (10.0, 12.0, 14.0)):
        lessons.update_run(d, status="finished", evaluation={"ga": {"games": 100, "mean": mean, "median": mean, "max": 20,
                                                                    "mean_steps": 100.0}})
    s = lessons.group_spread(lessons.group_members(infos[0]["id"]))["ga"]
    assert (s["n"], s["mean"], s["min"], s["max"]) == (3, 12.0, 10.0, 14.0) and s["std"] == pytest.approx(2.0)
    lessons.update_lesson("s")
    text = (lab / "s" / "LESSON.md").read_text(encoding="utf-8")
    assert "Seed group" in text and "12.00 ± 2.00" in text and "seed 3 of 3" in text


def test_group_is_analysed_once_when_complete(lab):
    dirs = lessons.new_group(runconfig.normalize({"lesson": "s", "agents": ["ga"]}), 2, status="queued")
    first, second = (lessons.read_run(d) for d in dirs)
    lessons.update_run(dirs[0], status="finished")
    assert not experiment.group_complete(lessons.read_run(dirs[0])), "the other seed is still training"
    lessons.update_run(dirs[1], status="finished")
    assert experiment.group_complete(lessons.read_run(dirs[1]))
    assert not experiment.group_complete(lessons.read_run(dirs[0])), "only one run claims the analysis"
    assert experiment.group_complete(lessons.read_run(lessons.new_run(runconfig.normalize({"lesson": "t"}))))


def test_api_launches_a_seed_group(lab):
    from dashboard.server import create_app
    client = TestClient(create_app(parallel=0))
    res = client.post("/api/runs", json={"config": {"lesson": "g", "agents": ["dqn"], "seed": 3}, "seeds": 4}).json()
    client.app.state.hub.refresh_runs()
    runs = [r for r in client.get("/api/runs").json()["runs"] if r["lesson"] == "g"]
    assert len(runs) == 4 and {r["group"]["id"] for r in runs} == {res["id"]}
    assert sorted(r["config"]["seed"] for r in runs) == [3, 4, 5, 6]
    assert len(lessons.new_group(runconfig.normalize({"lesson": "h"}), 99)) == lessons.MAX_SEEDS
