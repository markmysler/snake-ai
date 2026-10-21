"""The static demo (tools/export_site.py): the server's answers saved as files, replay games and Inspect frames."""
import gzip
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import lessons  # noqa: E402

REAL = os.path.join(os.path.dirname(__file__), "..", "lessons")


def test_export_writes_a_working_static_site(tmp_path, monkeypatch):
    shutil.copytree(os.path.join(REAL, "01-short-vision"), tmp_path / "lessons" / "01-short-vision")
    monkeypatch.setattr(lessons, "LESSONS_DIR", str(tmp_path / "lessons"))
    import ideas
    monkeypatch.setattr(ideas, "IDEAS_DIR", str(tmp_path / "ideas"))
    from export_site import export
    out = tmp_path / "site"
    export(out, n_games=1, log=lambda *_: None)
    data = out / "data"

    html = (out / "index.html").read_text(encoding="utf-8")
    assert "window.SNAKE_SITE = true" in html and '"/static/' not in html  # relative paths: works under /snake-ai/
    assert (out / "static" / "js" / "app.js").is_file() and (out / ".nojekyll").is_file()
    catalog = json.loads((data / "catalog.json").read_text())
    assert catalog["read_only"] and not catalog["llm"]["configured"]
    assert json.loads((data / "events.json").read_text())["system"] is None  # this machine is never published

    run = json.loads((data / "runs.json").read_text())["runs"][0]
    watch = json.loads((data / "watch" / f"{run['id']}.json").read_text())["watch"]
    assert watch["run"]["id"] == run["id"] and watch["history_keys"]
    games = json.loads((data / "game" / run["id"] / "dqn_score.json").read_text())
    game = games[0]["game"]
    frames = json.loads(gzip.decompress((data / "frame" / run["id"] / "dqn_score" / f"{game['seed']}.json.gz").read_bytes()))
    assert frames["info"]["outputs"] == "q" and frames["info"]["names"]
    assert len(frames["frames"]) == game["steps"] + 1  # every move, and the end
    assert frames["frames"][-1]["out"] is None and frames["frames"][-1]["score"] == game["score"]
    assert json.loads((data / "lesson" / "01-short-vision.json").read_text())["markdown"]
