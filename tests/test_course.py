"""The course: recipes that change one thing each, generated LESSON.md blocks, and queueing seed groups."""
import os
import shutil
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import course  # noqa: E402
import lessons  # noqa: E402
import runconfig  # noqa: E402

REAL = os.path.join(ROOT, "lessons")


@pytest.fixture
def lib(tmp_path, monkeypatch):
    shutil.copy(os.path.join(REAL, "course.json"), tmp_path / "course.json")
    monkeypatch.setattr(lessons, "LESSONS_DIR", str(tmp_path))
    return tmp_path


def test_every_recipe_is_valid_and_changes_one_setting(lib):
    c = course.load()
    assert len(c["lessons"]) >= 9
    for entry in c["lessons"]:
        for r in entry["recipes"] + entry.get("try", []):
            assert not runconfig.validate(r["full"])[0], (entry["lesson"], r.get("name"))
    assert not [p for p in course.check(c) if "one setting" in p]


def test_lesson_files_are_generated_and_keep_your_notes(lib):
    written = course.write_lessons()
    assert len(written) == len(course.load()["lessons"])
    p = lib / "c02-you-get-what-you-reward" / "LESSON.md"
    text = p.read_text(encoding="utf-8")
    assert text.startswith("# 02 · You get what you reward") and "**Question:**" in text and "## Try this" in text
    assert "../c01-two-ways-to-learn/LESSON.md" in text and "../c03-hints-and-shaping/LESSON.md" in text
    assert not course.learned_written("c02-you-get-what-you-reward")
    p.write_text(text.replace(course.TODO, "The snake learned to loop. _Not a fill-in"), encoding="utf-8")
    assert course.write_lessons() == []  # nothing generated changed
    assert "The snake learned to loop." in p.read_text(encoding="utf-8")
    assert course.learned_written("c02-you-get-what-you-reward")
    lessons.update_lesson("c02-you-get-what-you-reward")  # the results block is added without touching the rest
    after = p.read_text(encoding="utf-8")
    assert course.CSTART in after and course.TSTART in after and lessons.START in after


def test_queue_makes_tagged_seed_groups_once(lib):
    dirs = course.queue(3, auto_analyze=False)  # 3 recipes x 3 seeds
    assert len(dirs) == 9
    runs = lessons.list_runs()
    assert {r["course"]["recipe"] for r in runs} == {"no hint", "distance hint", "potential hint"}
    assert all(r["group"]["size"] == 3 and r["status"] == "queued" for r in runs)
    assert course.queue(3, auto_analyze=False) == []  # already queued
    trial = course.queue(1, auto_analyze=False, minutes=0.5)
    entry = course.find(course.load(), 1)
    assert len(trial) == 3 and course.recipe_runs(entry, entry["recipes"][0]) == []  # trials don't count
    info = lessons.read_run(trial[0])
    assert info["config"]["time_budget_min"] == 0.5 and info["course"]["trial"]


def test_a_run_is_started_only_once(lib):
    import experiment
    run_dir = course.queue(1, auto_analyze=False)[0]
    open(os.path.join(run_dir, "logs", "started"), "w").close()  # another process got it first
    assert experiment.execute(run_dir)["status"] == "queued"


def test_course_api(lib, monkeypatch):
    from fastapi.testclient import TestClient
    import ideas
    monkeypatch.setattr(ideas, "IDEAS_DIR", str(lib / "ideas"))
    from dashboard.server import create_app
    client = TestClient(create_app(parallel=0))
    cat = client.get("/api/catalog").json()
    assert cat["course"]["lessons"][0]["recipes"][0]["config"]["lesson"] == "c01-two-ways-to-learn"
    res = client.post("/api/course/run?lesson=c01-two-ways-to-learn").json()
    assert res["queued"] == 3 and len(res["queue"]["queued"]) == 3
    assert client.post("/api/course/run?lesson=nope").status_code == 400
    assert all(r["course"]["recipe"] == "baseline" for r in client.get("/api/runs").json()["runs"])
