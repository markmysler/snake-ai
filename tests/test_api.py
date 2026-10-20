"""Dashboard API: catalog, planning, launching, editing and deleting runs (no training started)."""
import os
import shutil
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import lessons  # noqa: E402

REAL = os.path.join(os.path.dirname(__file__), "..", "lessons")


@pytest.fixture
def client(tmp_path, monkeypatch):
    shutil.copytree(os.path.join(REAL, "01-short-vision"), tmp_path / "01-short-vision")
    monkeypatch.setattr(lessons, "LESSONS_DIR", str(tmp_path))
    monkeypatch.setenv("SNAKE_AI_LESSONS", str(tmp_path))
    import ideas
    monkeypatch.setattr(ideas, "IDEAS_DIR", str(tmp_path / "ideas"))
    from dashboard.server import create_app
    return TestClient(create_app(parallel=0))  # parallel=0: queued runs are never started


def test_catalog_and_system(client):
    cat = client.get("/api/catalog").json()
    assert {v["name"] for v in cat["visions"]} >= {"short", "full", "rays", "window"}
    assert {a["name"] for a in cat["algorithms"]} == {"ga", "dqn", "jev"}
    feats = {f["name"]: f for f in cat["rewards"]["features"]}
    assert "potential" in feats["distance"]["incompatible"]
    assert client.get("/api/system").json()["cpus"] >= 1


def test_plan_validates_and_sizes(client):
    r = client.post("/api/plan", json={"config": {"vision": "window:radius=3", "agents": ["ga", "dqn"],
                                                  "reward": "apple=10,distance=0.1,potential=0.1"}}).json()
    assert r["errors"] and r["plan"]["inputs"] == 3 * 49 + 4
    assert set(r["plan"]["agents"]) == {"ga", "dqn"} and r["plan"]["max_parallel"] >= 1
    assert client.post("/api/plan", json={"config": {"vision": "eyes"}}).json()["errors"]


def test_runs_launch_edit_delete(client):
    runs = client.get("/api/runs").json()["runs"]
    assert runs[0]["agents"][0]["reward"] == "ga_classic" and runs[0]["evaluation"]["ga"]["mean"] > 0
    bad = client.post("/api/runs", json={"config": {"reward": "hunger=0.1,survival=0.1"}})
    assert bad.status_code == 400
    new = client.post("/api/runs", json={"config": {"lesson": "My Test!", "agents": ["dqn"], "vision": "rays"}}).json()
    assert new["id"].startswith("my-test/") and new["id"] in new["queue"]["queued"]
    assert client.patch(f"/api/run?id={new['id']}", json={"name": "renamed"}).status_code == 200
    assert client.get(f"/api/run?id={new['id']}").json()["name"] == "renamed"
    assert client.delete(f"/api/run?id={new['id']}").status_code == 200
    assert client.get(f"/api/run?id={new['id']}").status_code == 404


def test_replay_game_from_a_saved_run(client):
    rid = client.get("/api/runs").json()["runs"][0]["id"]
    g = client.get(f"/api/game?run={rid}&agent=dqn_score").json()
    assert g["meta"]["vision"]["type"] == "short" and len(g["game"]["heads"]) > 0
    assert "Short vision" in client.get("/api/lesson?name=01-short-vision").json()["title"]


def test_inspect_frames_follow_the_replayed_game(client):
    rid = client.get("/api/runs").json()["runs"][0]["id"]
    g = client.get(f"/api/game?run={rid}&agent=dqn_score").json()
    game, version = g["game"], g["meta"]["version"]
    url = f"/api/frame?run={rid}&agent=dqn_score&seed={game['seed']}&version={version}"
    w = game["w"]
    first = client.get(url + "&step=0").json()
    assert first["moves"] == game["steps"] and first["info"]["outputs"] == "q" and len(first["info"]["names"]) == 11
    assert [c[1] * w + c[0] for c in first["snake"]] == game["init"]
    for step in (1, 2, 7, 3):  # forward and back
        f = client.get(url + f"&step={step}").json()
        x, y = f["snake"][0]
        assert y * w + x == game["heads"][step - 1]
        assert f["action"] == max(range(3), key=lambda i: f["out"][i])  # the move is the highest output
    last = client.get(url + f"&step={game['steps']}").json()
    assert last["out"] is None and last["end"] in ("wall", "body", "starved", "won", "limit") and last["score"] == game["score"]
    assert client.get(url.replace(version, "missing") + "&step=0").status_code == 410


def test_run_again_keeps_a_per_agent_reward(client):
    base = client.get("/api/runs").json()["runs"][0]
    assert base["config"]["reward"] is None  # lesson 01: each agent has its own reward
    new = client.post("/api/runs", json={"config": {**base["config"], "vision": "room"}}).json()
    again = client.get(f"/api/run?id={new['id']}").json()["config"]
    assert again["reward"] is None and [a["reward"] for a in again["agents"]] == [a["reward"] for a in base["config"]["agents"]]


def test_read_only_demo_refuses_every_change(tmp_path, monkeypatch):
    shutil.copytree(os.path.join(REAL, "01-short-vision"), tmp_path / "01-short-vision")
    monkeypatch.setattr(lessons, "LESSONS_DIR", str(tmp_path))
    import ideas
    monkeypatch.setattr(ideas, "IDEAS_DIR", str(tmp_path / "ideas"))
    from dashboard.server import create_app
    app = create_app(parallel=2, read_only=True)
    c = TestClient(app)
    rid = c.get("/api/runs").json()["runs"][0]["id"]
    assert c.get("/api/catalog").json()["read_only"] is True
    assert app.state.hub.jobs.concurrency == 0
    changes = [("post", "/api/runs", {"config": {}}), ("patch", f"/api/run?id={rid}", {"name": "x"}),
               ("delete", f"/api/run?id={rid}", None), ("post", f"/api/run/stop?id={rid}", None),
               ("post", "/api/queue", {"concurrency": 2}), ("post", f"/api/analysis?id={rid}", None),
               ("post", f"/api/analysis/launch?id={rid}&index=0", None), ("patch", "/api/idea?id=I1", {"status": "done"}),
               ("post", "/api/ideas/review", None), ("post", "/api/course/run?lesson=c01-two-ways-to-learn", None)]
    for method, url, body in changes:
        res = getattr(c, method)(url, **({"json": body} if body is not None else {}))
        assert res.status_code == 403, (method, url, res.status_code)
    assert c.get(f"/api/run?id={rid}").status_code == 200  # still there
    assert c.post("/api/plan", json={"config": {}}).status_code == 200  # planning a recipe is harmless
    g = c.get(f"/api/game?run={rid}&agent=ga").json()["game"]
    assert c.get(f"/api/frame?run={rid}&agent=ga&seed={g['seed']}&step=3").status_code == 200  # inspect works
