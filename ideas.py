"""The ideas list: improvements the AI analyses proposed that the lab can't do yet.

An analysis splits its advice in two:
- recipe changes use only what exists (visions, rewards, algorithm settings), so they can be launched
  (analysis.py);
- ideas need new code: a new vision, reward feature, algorithm, setting, chart or tool. They are
  collected here, from every analysed run, so they can pile up and be judged together.

A review asks the LLM which ideas to build next: it merges duplicates, weighs them against what the
recorded runs showed, and ranks them. Reviews are kept, so you can see how the advice changed.

    ideas/
      ideas.json            every idea: id, title, kind, what, why, expected, effort, source run, status
      review.json           the latest review (or the one running)
      reviews/<time>.json   earlier reviews
      IDEAS.md              readable version of all of it (regenerated)

    python ideas.py list
    python ideas.py review                  # ask the LLM which ideas to build (needs .env, see llm.py)
    python ideas.py status <idea-id> planned|done|dismissed|new
"""
import argparse
import glob
import json
import os
import sys
import time

import lessons
import llm
import system

ROOT = os.path.dirname(os.path.abspath(__file__))
IDEAS_DIR = os.environ.get("SNAKE_AI_IDEAS") or os.path.join(ROOT, "ideas")
KINDS = ("vision", "reward", "algorithm", "setting", "training", "evaluation", "dashboard", "other")
EFFORTS = ("small", "medium", "large")
STATUSES = ("new", "reviewed", "planned", "done", "dismissed")
OPEN = ("new", "reviewed", "planned")

REVIEW_SYSTEM = """You are the technical lead of a hands-on machine learning lab where learners train small
networks to play Snake (a genetic algorithm and DQN), change one setting at a time and compare runs.

After analysing runs, an AI tutor proposed improvements that need new code. You get every open idea,
what the lab can already do (its catalog), the best results so far, and the lessons recorded.

Decide what to build next:
- Merge ideas that are the same or overlap; say which idea ids each item covers.
- Rank by value for learners and for the results, per unit of effort. Prefer ideas backed by evidence
  from several runs over a guess from one.
- Drop ideas the lab already supports (point to the existing option) or that wouldn't teach anything.
- Be concrete about how to build each one in this code base: visions register in vision.py, reward
  features in rewards.py, algorithms in algorithms.py (plus a trainer script), and every new option
  appears in the dashboard automatically.

Answer with a single JSON object, no prose around it:
{
  "summary": "2-4 sentences: the overall picture and the single most valuable next step",
  "build_next": [
    {"title": "short name of the feature", "kind": "vision|reward|algorithm|setting|training|evaluation|dashboard|other",
     "ideas": ["idea ids it covers"], "what": "what to build, 1-3 sentences",
     "why": "why it is worth it now, citing the evidence", "how": "where and how to add it in this code base",
     "effort": "small|medium|large", "first_experiment": "the first run to try once it exists"}
  ],
  "later": [{"title": "...", "ideas": ["ids"], "reason": "why not now"}],
  "drop": [{"ideas": ["ids"], "reason": "already possible with ..., or why it isn't useful"}]
}
Put at most 5 items in build_next, best first. Every open idea id should appear in exactly one list."""


# ---- storage ----------------------------------------------------------------------------------
def _path(*parts):
    return os.path.join(IDEAS_DIR, *parts)


def load():
    return lessons.read_json(_path("ideas.json"), {"ideas": []})["ideas"]


def _number(idea_id):
    return int(idea_id[5:]) if idea_id.startswith("idea-") and idea_id[5:].isdigit() else 0


def save(items):
    os.makedirs(IDEAS_DIR, exist_ok=True)
    last = max([lessons.read_json(_path("ideas.json"), {}).get("last", 0), *(_number(i["id"]) for i in items)])
    lessons.write_json(_path("ideas.json"), {"last": last, "ideas": items})  # ids are never reused
    write_markdown(items)


def _next_number(items):
    last = lessons.read_json(_path("ideas.json"), {}).get("last", 0)
    return max([last, *(_number(i["id"]) for i in items)]) + 1


def clean(p):
    """One proposal from the LLM, normalized."""
    text = lambda k, n=600: str(p.get(k) or "").strip()[:n]  # noqa: E731
    kind, effort = text("kind").lower(), text("effort").lower()
    return {"title": text("title", 120) or "Untitled idea", "kind": kind if kind in KINDS else "other",
            "what": text("what"), "why": text("why"), "expected": text("expected"),
            "effort": effort if effort in EFFORTS else "medium"}


def add_from_analysis(run_id, proposals, model, created):
    """Record an analysis' ideas. Re-analysing a run replaces its ideas nobody has acted on yet.
    Returns the ideas as stored (with ids)."""
    items = load()
    items = [i for i in items if not (i["source"]["run"] == run_id and i["status"] == "new")]
    n = _next_number(items)
    added = []
    for p in proposals:
        idea = {"id": f"idea-{n}", **clean(p), "status": "new", "created": created,
                "source": {"run": run_id, "model": model}, "review": None}
        if p.get("note"):
            idea["note"] = str(p["note"])[:300]
        added.append(idea)
        n += 1
    save(items + added)
    return added


def set_status(idea_id, status):
    if status not in STATUSES:
        raise ValueError(f"status must be one of {', '.join(STATUSES)}")
    items = load()
    for i in items:
        if i["id"] == idea_id:
            i["status"] = status
            save(items)
            return i
    raise KeyError(idea_id)


def open_ideas():
    return [i for i in load() if i["status"] in OPEN]


# ---- reviews ------------------------------------------------------------------------------------
def read_review(name=None):
    """The latest review (name=None) or an archived one. A running review whose process died reads as an error."""
    r = lessons.read_json(_path("reviews", f"{name}.json") if name else _path("review.json"))
    if r and r.get("state") == "running" and not system.pid_alive(r.get("pid")):
        r = {**r, "state": "error", "error": "The review stopped before it finished."}
    return r


def reviews():
    """Archived reviews, newest first: [{name, created, model, items}]."""
    out = []
    for path in sorted(glob.glob(_path("reviews", "*.json")), reverse=True):
        r = lessons.read_json(path) or {}
        out.append({"name": os.path.splitext(os.path.basename(path))[0], "created": r.get("created"),
                    "model": r.get("model"), "ideas": r.get("ideas_reviewed", 0)})
    return out


def best_results():
    """The best tested model per algorithm, per vision and overall, for the reviewer's context."""
    rows = []
    for r in lessons.list_runs():
        for agent_id, ev in (r.get("evaluation") or {}).items():
            agent = next((a for a in r["config"]["agents"] if a["id"] == agent_id), None)
            if agent:
                rows.append({"run": r["id"], "agent": agent_id, "algorithm": agent["algorithm"],
                             "vision": r["config"]["vision"], "reward": r["config"]["reward"] or agent.get("reward"),
                             "test_mean": ev["mean"]})
    rows.sort(key=lambda x: -x["test_mean"])
    return rows[:15]


def context():
    import analysis
    return {"open_ideas": [{k: i[k] for k in ("id", "title", "kind", "what", "why", "expected", "effort", "status")}
                           | {"from_run": i["source"]["run"]} for i in open_ideas()],
            "catalog": analysis.catalog(), "best_results": best_results(),
            "lessons": [{"name": n, "title": lessons.lesson_title(n)} for n in lessons.lessons()]}


def _clean_review(raw, valid_ids):
    ids = lambda xs: [x for x in (xs or []) if x in valid_ids]  # noqa: E731
    text = lambda d, k, n=900: str(d.get(k) or "").strip()[:n]  # noqa: E731
    build = []
    for b in (raw.get("build_next") or [])[:5]:
        if isinstance(b, dict):
            kind, effort = text(b, "kind").lower(), text(b, "effort").lower()
            build.append({"title": text(b, "title", 120), "kind": kind if kind in KINDS else "other",
                          "ideas": ids(b.get("ideas")), "what": text(b, "what"), "why": text(b, "why"),
                          "how": text(b, "how"), "effort": effort if effort in EFFORTS else "medium",
                          "first_experiment": text(b, "first_experiment")})
    later = [{"title": text(x, "title", 120), "ideas": ids(x.get("ideas")), "reason": text(x, "reason")}
             for x in (raw.get("later") or []) if isinstance(x, dict)]
    drop = [{"ideas": ids(x.get("ideas")), "reason": text(x, "reason")}
            for x in (raw.get("drop") or []) if isinstance(x, dict)]
    return {"summary": text(raw, "summary", 1500), "build_next": build, "later": later, "drop": drop}


def review(verbose=False):
    """Ask the LLM which open ideas to build. Saves review.json (+ an archived copy)."""
    os.makedirs(_path("reviews"), exist_ok=True)
    st = llm.status()
    ctx = context()
    if not ctx["open_ideas"]:
        raise ValueError("there are no open ideas to review")
    started = time.time()
    archived = reviews()
    previous = read_review(archived[0]["name"]) if archived else None  # the last good review
    lessons.write_json(_path("review.json"), {"state": "running", "pid": os.getpid(), "model": st["model"],
                                               "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    try:
        if verbose:
            print(f"[ideas] asking {st['model']} to review {len(ctx['open_ideas'])} ideas...", flush=True)
        raw, usage = llm.chat_json([{"role": "system", "content": REVIEW_SYSTEM},
                                    {"role": "user", "content": "Review these ideas.\n\n" + json.dumps(ctx, indent=1)}],
                                   max_tokens=16000)
        valid = {i["id"] for i in ctx["open_ideas"]}
        result = _clean_review(raw, valid)
        if not result["summary"] and not result["build_next"]:
            raise llm.LLMError("The LLM's review was empty")
    except Exception as e:  # keep the last good review visible, with a note
        failed = time.strftime("%Y-%m-%d %H:%M:%S")
        lessons.write_json(_path("review.json"), {**previous, "last_error": str(e), "failed": failed}
                           if previous else
                           {"state": "error", "error": str(e), "model": st["model"], "failed": failed})
        raise
    name = time.strftime("%Y-%m-%d_%H%M%S")
    rev = {"state": "done", "name": name, "model": usage.get("model") or st["model"],
           "created": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - started, 1),
           "usage": usage, "ideas_reviewed": len(valid), "result": result}
    lessons.write_json(_path("reviews", f"{name}.json"), rev)
    lessons.write_json(_path("review.json"), rev)
    items = load()
    for i in items:  # every idea in this review has now been looked at
        if i["id"] in valid:
            i["review"] = name
            if i["status"] == "new":
                i["status"] = "reviewed"
    save(items)
    if verbose:
        print(f"[ideas] done in {rev['seconds']}s: {result['summary']}", flush=True)
    return rev


# ---- markdown -----------------------------------------------------------------------------------
def _md(text):
    return lessons._md(text)


def write_markdown(items=None):
    items = load() if items is None else items
    rev = read_review()
    lines = ["# Ideas", "",
             "Improvements the AI analyses proposed that need new code (new visions, rewards, algorithms, settings",
             "or tools). Recipe changes that the lab can already run are in each lesson instead. This file is",
             "regenerated by `ideas.py`: change an idea's status in the dashboard's Ideas tab or with",
             "`python ideas.py status <id> <status>`.", ""]
    if rev and rev.get("state") == "done":
        r = rev["result"]
        lines += [f"## Latest review ({rev['created']}, {rev['model']})", "", _md(r["summary"]), "", "**Build next:**", ""]
        for n, b in enumerate(r["build_next"], 1):
            lines.append(f"{n}. **{_md(b['title'])}** ({b['kind']}, {b['effort']} effort; covers "
                         f"{', '.join(b['ideas']) or '-'}). {_md(b['what'])} _Why:_ {_md(b['why'])} "
                         f"_How:_ {_md(b['how'])} _First experiment:_ {_md(b['first_experiment'])}")
        if r["later"]:
            lines += ["", "**Later:** " + "; ".join(f"{_md(x['title'])} ({_md(x['reason'])})" for x in r["later"])]
        lines.append("")
    for status in STATUSES:
        group = [i for i in items if i["status"] == status]
        if not group:
            continue
        lines += [f"## {status.capitalize()} ({len(group)})", ""]
        for i in group:
            lines.append(f"- **{i['id']} · {_md(i['title'])}** ({i['kind']}, {i['effort']}) {_md(i['what'])} "
                         f"_Why:_ {_md(i['why'])} _From run_ `{i['source']['run']}`.")
        lines.append("")
    os.makedirs(IDEAS_DIR, exist_ok=True)
    with open(_path("IDEAS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---- CLI ----------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("review")
    s = sub.add_parser("status")
    s.add_argument("id")
    s.add_argument("status", choices=STATUSES)
    a = p.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    if a.cmd == "list":
        for i in load():
            print(f"{i['id']:9s} {i['status']:9s} {i['kind']:10s} {i['effort']:6s} {i['title']}  (from {i['source']['run']})")
    elif a.cmd == "review":
        try:
            rev = review(verbose=True)
        except Exception as e:
            sys.exit(f"[ideas] review failed: {e}")
        for n, b in enumerate(rev["result"]["build_next"], 1):
            print(f"{n}. {b['title']} [{b['kind']}, {b['effort']}] covers {', '.join(b['ideas'])}\n   {b['what']}\n   how: {b['how']}\n")
    elif a.cmd == "status":
        print(set_status(a.id, a.status))


if __name__ == "__main__":
    main()
