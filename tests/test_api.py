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
    from dashboard.server import create_app
    return TestClient(create_app(parallel=0))  # parallel=0: queued runs are never started


def test_catalog_and_system(client):
    cat = client.get("/api/catalog").json()
    assert {v["name"] for v in cat["visions"]} >= {"short", "full", "rays", "window"}
    assert {a["name"] for a in cat["algorithms"]} == {"ga", "dqn"}
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
