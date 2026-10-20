"""The course: an ordered path through the lessons, each answering one question with one change.

    python course.py list                    # the lessons and what is trained so far
    python course.py write                   # (re)write each course lesson's LESSON.md from course.json
    python course.py run 3                   # queue lesson 3's recipes as seed groups (the dashboard trains them)
    python course.py run all --wait --parallel 2   # queue the whole course and train it here, 2 runs at a time
    python course.py check                   # what is still missing before the course is finished

lessons/course.json lists the lessons in order. Each has a question, the one change it makes, a
prediction prompt, background, terms from the glossary, its recipes (a recipe is the course defaults
plus a few changes) and "Try this" exercises. A lesson's recipes are compared with the lesson named in
"compare" (or, without one, with its own first recipe), and each must differ from it in one setting.

Course lessons are ordinary lessons: their LESSON.md has a block generated from course.json at the top,
the usual generated results, your "What we learned", and the exercises at the bottom.
"""
import argparse
import os
import sys
import time

import lessons
import runconfig

CSTART = "<!-- COURSE:START (generated from course.json by course.py - edit course.json instead) -->"
CEND = "<!-- COURSE:END -->"
TSTART = "<!-- COURSE:TRY:START (generated from course.json by course.py) -->"
TEND = "<!-- COURSE:TRY:END -->"
TODO = "_Fill this in"
DONE = ("finished", "stopped")


def path():
    return os.path.join(lessons.LESSONS_DIR, "course.json")


def load():
    """The course, or None when the library has none. Recipes get their full, normalized config."""
    c = lessons.read_json(path())
    if not c:
        return None
    for n, entry in enumerate(c["lessons"], 1):
        entry["n"] = n
        entry.setdefault("seeds", c.get("seeds", 3))
        for r in entry["recipes"]:
            r["full"] = config(c, entry, r["config"], r["name"])
        for t in entry.get("try", []):
            t["full"] = config(c, entry, t["config"], t["title"], t["why"])
    for entry in c["lessons"]:  # each run's note says what its recipe changes (shown in Runs and LESSON.md)
        for r in entry["recipes"]:
            b_entry, b = baseline(c, entry, r)
            r["full"]["note"] = (f"{entry['title']}. Changes from “{b['name']}”"
                                 + ("" if b_entry is entry else f" ({b_entry['title'].split(' · ')[0]})")
                                 + f": {'; '.join(same_except_seed(b['full'], r['full']))}." if b is not None
                                 else f"{entry['title']}. {entry['change']}")
    return c


def config(course, entry, changes, name="", note=""):
    return runconfig.normalize({**course["defaults"], **changes, "lesson": entry["lesson"], "name": name, "note": note})


def find(course, key):
    """A lesson by number (1-based) or folder name."""
    for entry in course["lessons"]:
        if str(key) in (str(entry["n"]), entry["lesson"]):
            return entry
    raise ValueError(f"no course lesson {key!r} (1-{len(course['lessons'])} or a folder name)")


def recipe_ref(course, ref):
    """'<lesson>/<recipe name>' -> (lesson entry, recipe)."""
    lesson, _, name = ref.partition("/")
    entry = find(course, lesson)
    return entry, next(r for r in entry["recipes"] if r["name"] == name)


def baseline(course, entry, recipe):
    """What a recipe is compared with: the lesson's "compare" recipe, else the lesson's first recipe."""
    if entry.get("compare"):
        return recipe_ref(course, entry["compare"])
    first = entry["recipes"][0]
    return (entry, first) if first is not recipe else (None, None)


def same_except_seed(a, b):
    import analysis
    return [c for c in analysis.changes(a, b) if not c.startswith("seed ")]


# ---- runs ----------------------------------------------------------------------------
def recipe_runs(entry, recipe, runs=None):
    """Runs trained for a course recipe (newest group last). Quick trials (run --minutes) don't count."""
    runs = lessons.list_runs() if runs is None else runs
    return [r for r in runs if r["lesson"] == entry["lesson"] and (r.get("course") or {}).get("recipe") == recipe["name"]
            and not r["course"].get("trial")]


def queue(key, parallel=1, again=False, auto_analyze=None, minutes=None):
    """Queue a lesson's recipes (every lesson for key="all") as seed groups. Recipes that already have runs
    (finished, or still training) are skipped unless again=True. Returns the queued run dirs."""
    course = load()
    if not course:
        raise ValueError("this library has no course.json")
    if auto_analyze is None:
        import llm
        auto_analyze = llm.status()["configured"]
    entries = course["lessons"] if str(key) == "all" else [find(course, key)]
    write_lessons(course)  # before the first run, so the lesson starts from the course's write-up
    runs, dirs = lessons.list_runs(), []
    for entry in entries:
        for recipe in entry["recipes"]:
            have = [r for r in recipe_runs(entry, recipe, runs) if r["status"] in (*DONE, *lessons.ACTIVE)]
            if have and not again:
                continue
            cfg = recipe["full"] if minutes is None else {**recipe["full"], "time_budget_min": float(minutes)}
            tag = {"recipe": recipe["name"], **({"trial": True} if minutes is not None else {})}
            dirs += lessons.new_group(cfg, entry["seeds"], status="queued", parallel=parallel,
                                      course=tag, auto_analyze=auto_analyze)
    write_lessons()
    return dirs


def train_queue(parallel=1, poll_s=5.0):
    """Train every queued run here, `parallel` at a time, until none are left (like the dashboard does).
    Safe to run while the dashboard is open: a run is only ever started once."""
    from dashboard.jobs import Jobs
    jobs = Jobs(concurrency=parallel)
    while True:
        runs = lessons.list_runs()
        active = [r for r in runs if r["status"] in lessons.ACTIVE]
        if not active and not jobs.procs:
            return
        jobs.tick(runs)
        print(f"[course] {time.strftime('%H:%M')} {sum(r['status'] == 'running' for r in runs)} training, "
              f"{sum(r['status'] == 'queued' for r in runs)} queued", flush=True)
        time.sleep(poll_s)


def status(course=None):
    """Per lesson: each recipe's runs, whether "What we learned" is written, and what is missing."""
    course = course or load()
    runs = lessons.list_runs()
    out = []
    for entry in course["lessons"]:
        recipes = []
        for r in entry["recipes"]:
            rr = recipe_runs(entry, r, runs)
            recipes.append({"name": r["name"], "runs": len(rr), "done": sum(x["status"] in DONE for x in rr),
                            "active": sum(x["status"] in lessons.ACTIVE for x in rr),
                            "groups": list(dict.fromkeys((x.get("group") or {}).get("id") or x["id"] for x in rr))})
        out.append({"n": entry["n"], "lesson": entry["lesson"], "title": entry["title"], "recipes": recipes,
                    "learned": learned_written(entry["lesson"])})
    return out


def learned_written(lesson):
    try:
        with open(os.path.join(lessons.LESSONS_DIR, lesson, "LESSON.md"), encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return False
    learned = text.split("## What we learned", 1)[1] if "## What we learned" in text else ""
    learned = learned.split(TSTART, 1)[0]
    return bool(learned.strip()) and TODO not in learned


def check(course=None):
    """Problems left before the course is finished, as plain sentences."""
    course = course or load()
    problems = []
    for entry in course["lessons"]:
        for r in entry["recipes"]:
            b_entry, b = baseline(course, entry, r)
            if b is not None:
                diff = same_except_seed(b["full"], r["full"])
                if len(diff) != 1:
                    problems.append(f"{entry['lesson']}: “{r['name']}” should change one setting from "
                                    f"{b_entry['lesson']}/{b['name']}, but changes {len(diff)}: {'; '.join(diff) or 'nothing'}")
    for s in status(course):
        for r in s["recipes"]:
            if r["done"] < (find(course, s["n"])["seeds"]):
                problems.append(f"{s['lesson']}: “{r['name']}” has {r['done']} finished run(s)"
                                f" of {find(course, s['n'])['seeds']} seeds")
        if not s["learned"]:
            problems.append(f"{s['lesson']}: “What we learned” is not written yet")
    return problems


# ---- LESSON.md -------------------------------------------------------------------------
def _terms(ids):
    import glossary
    names = {e["id"]: e["term"] for e in glossary.entries()}
    return ", ".join(names.get(i, i) for i in ids)


def _recipe_line(course, entry, r):
    d = runconfig.describe(r["full"])
    agents = " + ".join(a["label"].split(" (")[0] for a in d["agents"])
    seeds = f"seeds {r['full']['seed']}–{r['full']['seed'] + entry['seeds'] - 1}"
    b_entry, b = baseline(course, entry, r)
    vs = ""
    if b is not None:
        where = "" if b_entry is entry else f" of lesson {b_entry['title'].split(' · ')[0]}"
        vs = f" Changes from “{b['name']}”{where}: {'; '.join(same_except_seed(b['full'], r['full']))}."
    return (f"- **{r['name']}**: vision {d['vision_spec']}, reward {d['reward']}, {agents}, "
            f"{r['full']['time_budget_min']:g} min × {entry['seeds']} {seeds}.{vs}")


def _nav(course, entry):
    ls = course["lessons"]
    i = entry["n"] - 1
    prev = f"[← {ls[i - 1]['title']}](../{ls[i - 1]['lesson']}/LESSON.md)" if i else ""
    nxt = f"[{ls[i + 1]['title']} →](../{ls[i + 1]['lesson']}/LESSON.md)" if i + 1 < len(ls) else ""
    return " · ".join(filter(None, [prev, f"Course lesson {entry['n']} of {len(ls)}", nxt]))


def head_block(course, entry):
    lines = [CSTART, _nav(course, entry), "",
             f"**Question:** {entry['question']}", "",
             f"**Change:** {entry['change']}", "",
             f"**Your prediction:** {entry['predict']} _Write yours down before you look at the results._", "",
             "## Background", "", entry["explain"], ""]
    if entry.get("watch"):
        lines += [f"**What to watch:** {entry['watch']}", ""]
    lines += ["**Recipes**", "", *(_recipe_line(course, entry, r) for r in entry["recipes"]), ""]
    if entry.get("pilot"):
        lines += [f"Earlier pilot runs of this question: {', '.join(f'[{p}](../{p}/LESSON.md)' for p in entry['pilot'])}.", ""]
    if entry.get("terms"):
        lines += [f"**Terms** (explained in the dashboard's Learn tab): {_terms(entry['terms'])}.", ""]
    return "\n".join(lines + [CEND])


def try_block(course, entry):
    lines = [TSTART, "## Try this", ""]
    for i, t in enumerate(entry.get("try", []), 1):
        lines.append(f"{i}. **{t['title']}.** {t['why']} _(Runs tab → this lesson → Try this.)_")
    lines += ["", _nav(course, entry), TEND]
    return "\n".join(lines)


def _replace(text, start, end, block):
    if start in text and end in text:
        head, rest = text.split(start, 1)
        return head + block + rest.split(end, 1)[1]
    return None


def lesson_text(course, entry, text=None):
    """LESSON.md with the course blocks (re)generated; everything else is kept."""
    head, tail = head_block(course, entry), try_block(course, entry)
    if text is None:
        return (f"# {entry['title']}\n\n{head}\n\n## Results\n\n{lessons.START}\n{lessons.END}\n\n## What we learned\n\n"
                f"{TODO} after the runs: what happened, why, and whether your prediction held. The AI analysis of "
                f"each seed group (above) is a draft to start from._\n\n{tail}\n")
    lines = text.split("\n", 1)
    text = f"# {entry['title']}\n" + (lines[1] if len(lines) > 1 else "")
    text = _replace(text, CSTART, CEND, head) or text.replace("\n", f"\n\n{head}\n", 1)
    return _replace(text, TSTART, TEND, tail) or text.rstrip("\n") + f"\n\n{tail}\n"


def write_lessons(course=None):
    course = course or load()
    if not course:
        return []
    written = []
    for entry in course["lessons"]:
        p = os.path.join(lessons.LESSONS_DIR, entry["lesson"], "LESSON.md")
        old = None
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                old = f.read()
        new = lesson_text(course, entry, old)
        if new != old:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(new)
            written.append(entry["lesson"])
    return written


def lessons_in_order():
    """Course lesson folders in course order (empty without a course)."""
    c = lessons.read_json(path())
    return [e["lesson"] for e in (c or {}).get("lessons", [])]


def payload():
    """What the dashboard shows: the course with each lesson's recipes and exercises (configs included)."""
    course = load()
    if not course:
        return None
    return {"title": course["title"], "about": course.get("about", ""),
            "lessons": [{"n": e["n"], "lesson": e["lesson"], "title": e["title"], "question": e["question"],
                         "change": e["change"], "seeds": e["seeds"], "terms": e.get("terms", []),
                         "compare": e.get("compare"),
                         "recipes": [{"name": r["name"], "config": r["full"]} for r in e["recipes"]],
                         "try": [{"title": t["title"], "why": t["why"], "config": t["full"], "seeds": t.get("seeds", e["seeds"])}
                                 for t in e.get("try", [])]}
                        for e in course["lessons"]]}


# ---- CLI -----------------------------------------------------------------------------
def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("write")
    sub.add_parser("check")
    r = sub.add_parser("run")
    r.add_argument("lesson", help="lesson number, folder name, or 'all'")
    r.add_argument("--parallel", type=int, default=1, help="runs training at the same time (recorded in each run, "
                   "so wall-clock results stay comparable)")
    r.add_argument("--again", action="store_true", help="queue recipes that already have runs too")
    r.add_argument("--minutes", type=float, help="override the training time (for a quick trial; not the course)")
    r.add_argument("--no-analyze", action="store_true", help="skip the AI analysis of each finished seed group")
    r.add_argument("--wait", action="store_true", help="train the queue here instead of in the dashboard")
    a = p.parse_args()

    course = load()
    if not course:
        raise SystemExit(f"no course found at {path()}")
    if a.cmd == "list":
        for s in status(course):
            parts = [f"{r['name']} {r['done']}/{find(course, s['n'])['seeds']}" + (f" (+{r['active']} training)" if r["active"] else "")
                     for r in s["recipes"]]
            print(f"{s['n']:2d}. {s['title']:50s} {' | '.join(parts)}{'' if s['learned'] else '  [what we learned: to do]'}")
    elif a.cmd == "write":
        print("written:", ", ".join(write_lessons(course)) or "nothing changed")
    elif a.cmd == "check":
        problems = check(course)
        print("\n".join(problems) or "The course is complete.")
        raise SystemExit(1 if problems else 0)
    elif a.cmd == "run":
        dirs = queue(a.lesson, parallel=a.parallel, again=a.again, auto_analyze=False if a.no_analyze else None,
                     minutes=a.minutes)
        print(f"queued {len(dirs)} run(s)" + ("" if a.wait else "; open the dashboard to train them "
                                               f"(set 'runs at a time' to {a.parallel})"))
        if a.wait:
            train_queue(a.parallel)
            lessons.write_index()
            print("[course] done")


if __name__ == "__main__":
    main()
