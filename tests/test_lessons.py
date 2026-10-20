"""Run library: runs are recorded with stats, models and evaluation; old models stay playable."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ideas  # noqa: E402
import lessons  # noqa: E402
import runconfig  # noqa: E402
from policies import load_policy, mlp_param_count  # noqa: E402
from publisher import Publisher  # noqa: E402
from vision import state_size  # noqa: E402


@pytest.fixture(autouse=True)
def tmp_lessons(tmp_path, monkeypatch):
    monkeypatch.setattr(lessons, "LESSONS_DIR", str(tmp_path))
    monkeypatch.setattr(ideas, "IDEAS_DIR", str(tmp_path / "ideas"))
    return tmp_path


def fake_training(run_dir, vision):
    pub = Publisher("ga", "generation", os.path.join(run_dir, "live"), os.path.join(run_dir, "results"))
    pub.stats(1, 500, 6, 2, 0.5)
    pub.finish()
    genome = np.random.default_rng(0).normal(size=mlp_param_count(state_size(vision))).astype(np.float32)
    np.save(os.path.join(run_dir, "models", "ga_best.npy"), genome)
    lessons.write_json(os.path.join(run_dir, "models", "ga_best.json"),
                       {"generation": 1, "hidden": 16, "width": 10, "height": 10, "vision": vision})


def test_run_is_recorded_with_evaluation(tmp_lessons):
    config = runconfig.normalize({"lesson": "02-test", "vision": "rays:depth=3", "agents": ["ga"]})
    run_dir = lessons.new_run(config, status="running")
    fake_training(run_dir, "rays:depth=3")
    info = lessons.finish_run(run_dir, eval_games=3)
    assert info["status"] == "finished" and info["final"]["ga"]["total_steps"] == 500
    assert info["evaluation"]["ga"]["games"] == 3
    text = (tmp_lessons / "02-test" / "LESSON.md").read_text(encoding="utf-8")
    assert lessons.START in text and "vision **rays:depth=3**" in text
    assert "02-test/LESSON.md" in (tmp_lessons / "README.md").read_text(encoding="utf-8")
    assert [r["id"] for r in lessons.list_runs()] == [info["id"]]


def test_notes_survive_and_runs_move_and_delete(tmp_lessons):
    run_dir = lessons.new_run(runconfig.normalize({"lesson": "01-x", "agents": ["ga"]}), status="running")
    path = tmp_lessons / "01-x" / "LESSON.md"
    path.write_text(path.read_text(encoding="utf-8").replace("_Fill this in", "MY NOTES _Fill this in"),
                    encoding="utf-8")
    fake_training(run_dir, "short")
    info = lessons.finish_run(run_dir, eval_games=0)
    assert "MY NOTES" in path.read_text(encoding="utf-8")
    new_id = lessons.edit_run(info["id"], name="first try", lesson="03 Other Lesson")
    assert new_id.startswith("03-other-lesson/")
    moved = lessons.read_run(lessons.run_path(new_id))
    assert moved["name"] == "first try" and moved["config"]["lesson"] == "03-other-lesson"
    lessons.delete_run(new_id)
    assert lessons.list_runs() == []
    assert not (tmp_lessons / "03-other-lesson").exists(), "untouched empty lesson is removed"
    assert (tmp_lessons / "01-x" / "LESSON.md").exists(), "a lesson with your notes is kept"


def test_active_runs_cannot_be_deleted():
    run_dir = lessons.new_run(runconfig.normalize({"lesson": "x", "agents": ["dqn"]}), status="queued")
    with pytest.raises(ValueError):
        lessons.delete_run(lessons.read_run(run_dir)["id"])


def test_models_without_vision_metadata_load_as_short(tmp_path):
    path = tmp_path / "ga_best.npy"
    np.save(path, np.zeros(mlp_param_count(11), np.float32))  # saved before visions existed
    act, meta = load_policy(str(path))
    assert meta["vision"] == {"type": "short"} and act(np.zeros(11, np.float32)) in (0, 1, 2)


def test_scrub_keeps_only_what_results_need(tmp_lessons):
    run_dir = lessons.new_run(runconfig.normalize({"lesson": "x", "agents": ["dqn"]}), status="finished")
    raw = lessons.read_json(os.path.join(run_dir, "run.json"))
    raw["config"]["imported_from"] = ["/home/someone/results"]
    lessons.write_json(os.path.join(run_dir, "run.json"), raw)
    lessons.update_run(run_dir, pid=1234, live_dir=r"C:\Users\someone\AppData\Local\Temp\snake-ai-live\x",
                       machine={"cpus": 8, "ram_total_mb": 16000, "ram_available_mb": 4000, "on_battery": False,
                                "platform": "Windows-11-10.0.26200-SP0", "python": "3.13.7"},
                       plan={"parallel": 1, "cores_per_run": 8, "ram_budget_mb": 3200, "max_parallel": 4, "warnings": [],
                             "agents": {"dqn": {"algorithm": "dqn", "cores": 1, "flag": "threads", "mem_mb": 365,
                                                "params": {"buffer": 50000}}}})
    assert lessons.scrub(verbose=False)
    raw = lessons.read_json(os.path.join(run_dir, "run.json"))
    text = str(raw)
    assert "someone" not in text and "Windows" not in text and "ram_" not in text and "pid" not in raw
    assert raw["machine"] == {"cpus": 8, "on_battery": False}
    assert raw["plan"]["agents"]["dqn"] == {"algorithm": "dqn", "cores": 1, "flag": "threads", "params": {"buffer": 50000}}
    assert lessons.scrub(verbose=False) == [], "nothing left to scrub"


def test_deleting_a_run_removes_links_to_it(tmp_lessons):
    src = lessons.new_run(runconfig.normalize({"lesson": "a", "agents": ["ga"]}), status="finished")
    src_id = lessons.read_run(src)["id"]
    child = lessons.new_run(runconfig.normalize({"lesson": "b", "agents": ["ga"]}), status="finished",
                            parent={"run": src_id, "recommendation": 0, "title": "Try rays"})
    child_id = lessons.read_run(child)["id"]
    lessons.write_json(os.path.join(child, "analysis.json"),
                       {"state": "done", "recommendations": [{"title": "x", "launched": [src_id, "b/other"]}]})
    ideas.add_from_analysis(src_id, [{"title": "An idea", "kind": "vision"}], "m", "now")
    lessons.delete_run(src_id)
    assert lessons.read_run(child)["parent"] == {"run": None, "recommendation": 0, "title": "Try rays", "deleted": True}
    assert lessons.read_json(os.path.join(child, "analysis.json"))["recommendations"][0]["launched"] == ["b/other"]
    idea = ideas.load()[0]
    assert idea["title"] == "An idea" and idea["source"]["run"] is None and idea["source"]["deleted"]
    assert "source run was deleted" in (tmp_lessons / "ideas" / "IDEAS.md").read_text(encoding="utf-8")
    assert child_id
