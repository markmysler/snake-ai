"""AI analysis: answers are recorded, recipes are checked, and a recommendation launches in one step.
The LLM is replaced by a canned answer, so these tests need no network or API key."""
import os
import shutil
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import analysis  # noqa: E402
import ideas  # noqa: E402
import lessons  # noqa: E402
import llm  # noqa: E402

REAL = os.path.join(os.path.dirname(__file__), "..", "lessons")
RUN = "01-short-vision/2026-10-15_041628"

ANSWER = {
    "headline": "The GA beat both DQNs.", "summary": "GA **51.6**. <script>x</script>", "diagnosis": "Selection.",
    "learned": "Evolution works with 11 inputs.", "prediction_check": "null",
    "recommendations": [
        {"title": "Try rays", "why": "See further.", "change": "vision", "prediction": "higher", "question": "Do rays help?",
         "lesson": "Rays Vision", "recipe": {"vision": {"type": "rays", "depth": 6}}},
        {"title": "Longer", "why": "Still improving.", "change": "time", "prediction": "a bit higher",
         "lesson": "same", "recipe": {"time_budget_min": 20}},
        {"title": "X-ray vision", "why": "See through the body.", "change": "vision", "prediction": "", "lesson": "same",
         "recipe": {"vision": {"type": "x-ray"}}},
    ],
    "ideas": [{"title": "Learning-rate schedule", "kind": "training", "what": "Decay lr over time.",
               "why": "DQN plateaued.", "expected": "+2 apples", "effort": "small"},
              {"title": "Odd", "kind": "telepathy", "effort": "huge"}]}


@pytest.fixture
def lab(tmp_path, monkeypatch):
    shutil.copytree(os.path.join(REAL, "01-short-vision"), tmp_path / "01-short-vision")
    monkeypatch.setattr(lessons, "LESSONS_DIR", str(tmp_path))
    monkeypatch.setenv("SNAKE_AI_LESSONS", str(tmp_path))
    monkeypatch.setattr(ideas, "IDEAS_DIR", str(tmp_path / "ideas"))
    monkeypatch.setattr(llm, "status", lambda: {"configured": True, "model": "fake/model"})
    monkeypatch.setattr(llm, "chat_json", lambda messages, **kw: (ANSWER, {"model": "fake/model"}))
    return tmp_path


def test_analysis_is_recorded_and_recipes_checked(lab):
    a = analysis.analyze(RUN)
    assert a["state"] == "done" and a["result"]["headline"] == ANSWER["headline"]
    rays, longer = a["recommendations"]  # the impossible recipe is not offered as a run...
    assert rays["changes"] == ["vision short → rays:depth=6"] and rays["new_lesson"] and not rays["errors"]
    assert rays["config"]["reward"] is None and rays["config"]["agents"][0]["reward"]  # per-agent rewards kept
    assert longer["changes"] == ["training time (min) 10 → 20"] and not longer["new_lesson"]
    assert a["recommended"] == 0
    titles = [i["title"] for i in a["ideas"]]  # ...it becomes an idea, next to the model's own ideas
    assert titles == ["Learning-rate schedule", "Odd", "X-ray vision"]
    stored = ideas.load()
    assert [i["id"] for i in stored] == ["idea-1", "idea-2", "idea-3"] and stored[1]["kind"] == "other"
    assert stored[1]["effort"] == "medium" and "can't run it" in stored[2]["note"]
    assert stored[0]["source"]["run"] == RUN and stored[0]["status"] == "new"
    text = (lab / "01-short-vision" / "LESSON.md").read_text(encoding="utf-8")
    assert lessons.ASTART in text and "The GA beat both DQNs." in text
    assert "<script>" not in text and "&lt;script&gt;" in text  # AI text can't inject HTML
    assert "## What we learned" in text.split(lessons.AEND)[1], "the hand-written part stays after the block"


def test_launch_creates_the_next_lesson(lab):
    analysis.analyze(RUN)
    run_dir = analysis.launch(RUN, 0)
    info = lessons.read_run(run_dir)
    assert info["lesson"] == "02-rays-vision" and info["status"] == "queued"
    assert info["config"]["vision"] == {"type": "rays", "depth": 6} and info["name"] == "Try rays"
    assert info["parent"]["run"] == RUN and info["auto_analyze"]
    lesson = (lab / "02-rays-vision" / "LESSON.md").read_text(encoding="utf-8")
    assert lesson.startswith("# 02 · Try rays") and "Do rays help?" in lesson and lessons.START in lesson
    assert analysis.read(lessons.run_path(RUN))["recommendations"][0]["launched"] == [info["id"]]
    with pytest.raises(ValueError):
        analysis.launch(RUN, 2)  # there are only two launchable recipes


def test_reanalysis_replaces_and_failure_is_recorded(lab, monkeypatch):
    analysis.analyze(RUN)
    def boom(messages, **kw):
        raise llm.LLMError("rate limited")
    monkeypatch.setattr(llm, "chat_json", boom)
    with pytest.raises(llm.LLMError):
        analysis.analyze(RUN)
    a = analysis.read(lessons.run_path(RUN))  # the earlier analysis is kept, with a note
    assert a["state"] == "done" and a["result"]["headline"] and "rate limited" in a["last_error"]
    os.remove(analysis.path(lessons.run_path(RUN)))
    with pytest.raises(llm.LLMError):
        analysis.analyze(RUN)
    a = analysis.read(lessons.run_path(RUN))
    assert a["state"] == "error" and "rate limited" in a["error"]


def test_api(lab):
    from dashboard.server import create_app
    client = TestClient(create_app(parallel=0))
    assert client.get("/api/catalog").json()["llm"] == {"configured": True, "model": "fake/model"}
    assert client.get(f"/api/analysis?id={RUN}").json()["analysis"] is None
    analysis.analyze(RUN)
    got = client.get(f"/api/analysis?id={RUN}").json()["analysis"]
    assert got["recommendations"][0]["lesson_target"] == "02-rays-vision"
    client.app.state.hub.refresh_runs()  # (the 2 s poller doesn't run under TestClient)
    run = next(r for r in client.get("/api/runs").json()["runs"] if r["id"] == RUN)
    assert run["analysis"]["state"] == "done" and run["analysis"]["headline"] == ANSWER["headline"]
    new = client.post(f"/api/analysis/launch?id={RUN}&index=1").json()["id"]
    assert new.startswith("01-short-vision/")
    assert client.post(f"/api/analysis/launch?id={RUN}&index=2").status_code == 400
    assert run["analysis"]["recommended"]["title"] == "Try rays" and run["analysis"]["ideas"] == 3
    # "Edit first": an edited recommendation is launched through the normal form, with its origin
    edited = {**got["recommendations"][0]["config"], "seed": 7}
    r = client.post("/api/runs", json={"config": edited, "origin": {"run": RUN, "index": 0}}).json()
    info = lessons.read_run(lessons.run_path(r["id"]))
    assert info["config"]["seed"] == 7 and info["parent"]["recommendation"] == 0
    assert (lab / "02-rays-vision" / "LESSON.md").read_text(encoding="utf-8").startswith("# 02 · Try rays")


def test_reanalysis_replaces_only_untouched_ideas(lab):
    analysis.analyze(RUN)
    ideas.set_status("idea-1", "planned")
    analysis.analyze(RUN)
    stored = ideas.load()
    assert [(i["id"], i["status"]) for i in stored] == [("idea-1", "planned"), ("idea-4", "new"),
                                                        ("idea-5", "new"), ("idea-6", "new")]
    assert (lab / "ideas" / "IDEAS.md").read_text(encoding="utf-8").count("Learning-rate schedule") == 2


def test_ideas_review(lab, monkeypatch):
    analysis.analyze(RUN)
    review = {"summary": "Build the schedule first.",
              "build_next": [{"title": "LR schedule", "kind": "training", "ideas": ["idea-1", "idea-99"],
                              "what": "w", "why": "y", "how": "train_dqn.py", "effort": "small", "first_experiment": "e"}],
              "later": [{"title": "X-ray", "ideas": ["idea-3"], "reason": "big"}],
              "drop": [{"ideas": ["idea-2"], "reason": "unclear"}]}
    monkeypatch.setattr(llm, "chat_json", lambda messages, **kw: (review, {"model": "fake/model"}))
    from dashboard.server import create_app
    client = TestClient(create_app(parallel=0))
    rev = ideas.review()
    assert rev["result"]["build_next"][0]["ideas"] == ["idea-1"], "unknown ids are dropped"
    got = client.get("/api/ideas").json()
    assert got["review"]["name"] == rev["name"] and len(got["reviews"]) == 1
    assert {i["status"] for i in got["ideas"]} == {"reviewed"}
    assert client.patch("/api/idea?id=idea-2", json={"status": "dismissed"}).json()["status"] == "dismissed"
    assert client.patch("/api/idea?id=idea-2", json={"status": "maybe"}).status_code == 400
    def boom(messages, **kw):
        raise llm.LLMError("down")
    monkeypatch.setattr(llm, "chat_json", boom)
    with pytest.raises(llm.LLMError):
        ideas.review()
    kept = ideas.read_review()
    assert kept["state"] == "done" and kept["name"] == rev["name"] and kept["last_error"] == "down"
