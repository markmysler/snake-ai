"""AI analysis of a run: what happened, why, what it teaches, and which runs to try next.

    python analysis.py run <lesson>/<run-id>          # analyze a finished run (needs .env, see llm.py)
    python analysis.py show <lesson>/<run-id>
    python analysis.py launch <lesson>/<run-id> <n>   # queue recommendation n (1 = first)

The model gets the run's recipe, its learning curves, its test results, the lesson's write-up and
the other recorded runs, plus the catalog of everything that can be changed. It answers in JSON.
Every recommended recipe is then rebuilt and checked here, exactly like the New run form does,
and the list of changes shown to you is computed from the recipes, not taken from the model.

The result is saved in the run's folder (analysis.json) and written into the lesson's LESSON.md.
Launching a recommendation that asks a new question creates the next lesson, with its question,
change and prediction already filled in.
"""
import argparse
import copy
import csv
import json
import os
import re
import sys
import time

import algorithms
import lessons
import ideas
import llm
import rewards
import runconfig
import system
import vision as visions

FILE = "analysis.json"
MAX_RELATED = 12
CURVE_POINTS = 12

SYSTEM = """You are the tutor of a hands-on machine learning lab. Learners train small neural networks to
play Snake with a genetic algorithm (GA, neuroevolution) and Deep Q-learning (DQN), change one setting
at a time, and compare runs. You analyse one recorded run for a learner who is new to the field.

Be concrete and honest:
- Quote the numbers that support each claim (test score, training mean, steps, time).
- The test score (100 fixed-seed games with the saved best model) is what counts. The training mean
  of the last 100 games includes exploration moves (DQN). For the GA it follows the elites it keeps,
  playing each new generation's fresh games; "population_mean" is the whole generation, mostly mutated
  children. GA curves without "population_mean" (older runs) are the last children only, so they look
  worse than the networks the GA kept.
- One run is one seed: small differences between runs may be noise. Say so when it matters. If the run
  belongs to a seed group (the same recipe trained with several seeds), judge effects against the spread
  across those seeds, not against one seed.
- Test results also say how each test game ended (wall, body, starved, won) and the steps per game and
  per apple: use them to tell a snake that crashes from one that wanders until it starves.
- The algorithms train for the same wall-clock time, not the same number of steps, so compare
  sample efficiency (score per env step) and speed (steps per second) separately.
- Explain jargon the first time you use it, in a few words.
- If the run carried a prediction, say whether it held.

Then give two kinds of advice, kept strictly apart:

1. "recommendations": 2 or 3 next runs the lab can launch TODAY, most instructive first (the first one
   is offered as the one-click "Run recommended"). Each changes ONE setting from this run (several
   parameters count as one setting only if they are meant to move together), tests a clear hypothesis,
   and keeps the training time unless time is the thing being tested. Use only the visions, reward
   features and algorithm parameters listed in the catalog, inside their ranges.
2. "ideas": 0 to 3 improvements that would need NEW CODE because the catalog has no way to do them:
   a new vision, reward feature, algorithm, algorithm setting, training feature (e.g. a schedule),
   evaluation, or dashboard view. Only propose an idea when this run gives a reason for it. Never put
   something the catalog can already do in "ideas".

Answer with a single JSON object, no prose around it:
{
  "headline": "one sentence: the main result, with its key number",
  "summary": "markdown, 1-3 short paragraphs: what each agent achieved and how its learning curve behaved",
  "diagnosis": "markdown, 1-2 short paragraphs: the most likely reasons, tied to this run's settings",
  "learned": "1-2 sentences: what a learner should take away from this run",
  "prediction_check": "1 sentence on whether the run's prediction held, or null if it had none",
  "recommendations": [
    {
      "title": "imperative, at most 7 words",
      "why": "1-2 sentences: the hypothesis and why it is worth testing",
      "change": "the one change in plain words",
      "prediction": "what you expect to happen, with a number where possible",
      "question": "the question this next run answers",
      "lesson": "same" to add the run to this run's lesson, or a short-kebab-case name (no number) for a new lesson,
      "recipe": {only the fields that change, from: "vision", "reward", "agents", "time_budget_min", "width", "height", "seed"}
    }
  ],
  "ideas": [
    {
      "title": "short name of the feature, at most 7 words",
      "kind": "vision|reward|algorithm|setting|training|evaluation|dashboard|other",
      "what": "what would be built, 1-2 sentences",
      "why": "the evidence from this run that motivates it",
      "expected": "what it should change, with a number where possible",
      "effort": "small|medium|large"
    }
  ]
}
Recipe field formats:
- "vision": {"type": "<vision name>", "<param>": value}
- "reward": the complete set of features to use, {"<feature>": weight}; weights are positive, the sign is
  fixed by the feature ("sign" in the catalog: -1 = subtracted)
- "agents": the complete list of algorithms, [{"id": "<keep this run's id>", "algorithm": "<name>",
  "params": {only the parameters that change}}]
Use "same" for "lesson" when the run repeats this lesson's question (for example a longer run or another seed)."""


# ---- storage ------------------------------------------------------------------------
def path(run_dir):
    return os.path.join(run_dir, FILE)


def read(run_dir):
    """The saved analysis, or None. A "running" analysis whose process died reads as an error."""
    a = lessons.read_json(path(run_dir))
    if a and a.get("state") == "running" and not system.pid_alive(a.get("pid")):
        a = {**a, "state": "error", "error": "The analysis stopped before it finished."}
    return a


def write(run_dir, obj):
    lessons.write_json(path(run_dir), obj)


def recommended(a):
    """Index of the recipe change offered as "Run recommended": the first one that can be launched."""
    if a.get("recommended") is not None:
        return a["recommended"]
    return next((i for i, r in enumerate(a.get("recommendations") or []) if r.get("config") and not r.get("errors")), None)


def brief(run_dir):
    """What the run list needs: state, headline, when, by which model, and the recommended next run."""
    a = read(run_dir)
    if not a:
        return None
    out = {"state": a.get("state"), "headline": (a.get("result") or {}).get("headline"),
           "created": a.get("created"), "model": a.get("model"), "error": a.get("error"),
           "last_error": a.get("last_error"), "failed": a.get("failed"), "recommended": None,
           "ideas": len(a.get("ideas") or [])}
    i = recommended(a) if a.get("state") == "done" else None
    if i is not None:
        rec = a["recommendations"][i]
        out["recommended"] = {"index": i, "title": rec["title"], "changes": rec["changes"],
                              "prediction": rec["prediction"], "launched": rec.get("launched", []),
                              "lesson": next_lesson(rec["lesson_slug"]) if rec["new_lesson"] else rec["lesson_slug"]}
    return out


# ---- what the model is told ------------------------------------------------------------
def read_curve(run_dir, agent_id, points=CURVE_POINTS):
    """The agent's learning curve, thinned to `points` rows (always keeps the last one)."""
    try:
        with open(os.path.join(run_dir, "results", f"{agent_id}.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return []
    if not rows:
        return []
    idx = sorted({round(i * (len(rows) - 1) / max(points - 1, 1)) for i in range(points)})
    return [{"minutes": round(float(rows[i]["elapsed_s"]) / 60, 2), "env_steps": int(rows[i]["total_steps"]),
             "games": int(rows[i]["total_games"]), rows[i]["iteration_label"]: int(rows[i]["iteration"]),
             "best_game": int(rows[i]["best_score"]), "mean_last_100": float(rows[i]["mean_score_100"]),
             "steps_per_sec": float(rows[i]["steps_per_sec"]),
             **({"population_mean": float(rows[i]["pop_mean_score"])} if rows[i].get("pop_mean_score") else {})}
            for i in idx]


def lesson_text(lesson, limit=3000):
    """The hand-written part of LESSON.md (generated blocks removed)."""
    try:
        with open(os.path.join(lessons.LESSONS_DIR, lesson, "LESSON.md"), encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return ""
    for start, end in ((lessons.START, lessons.END), (lessons.ASTART, lessons.AEND)):
        if start in text and end in text:
            head, rest = text.split(start, 1)
            text = head + "(generated block omitted)" + rest.split(end, 1)[1]
    return text.strip()[:limit]


def recipe_brief(c):
    d = runconfig.describe(c)
    return {"vision": c["vision"], "inputs": d["inputs"], "reward": c["reward"] if c["reward"] else "per agent",
            "agents": [{"id": a["id"], "algorithm": a["algorithm"], "params": a["params"],
                        **({"reward": a["reward"]} if a.get("reward") else {})} for a in c["agents"]],
            "time_budget_min": c["time_budget_min"], "width": c["width"], "height": c["height"], "seed": c["seed"]}


def run_brief(info):
    return {"id": info["id"], "name": info.get("name") or None, "note": info.get("note") or None,
            "status": info["status"], "recipe": recipe_brief(info["config"]),
            "test": {k: {"mean": v["mean"], "median": v["median"], "max": v["max"],
                         "steps_per_game": round(v["mean_steps"]),
                         **({"steps_per_apple": round(v["steps_per_apple"]), "ended": v["outcomes"]}
                            if "outcomes" in v else {})}
                     for k, v in info.get("evaluation", {}).items()},
            "final_training": {k: {"env_steps": v["total_steps"], "games": v["total_games"],
                                   "minutes": round(v["elapsed_s"] / 60, 1), "mean_last_100": v["mean_score_100"]}
                               for k, v in info.get("final", {}).items()}}


def seed_group(info):
    """The spread of this recipe across seeds, when the run is one of a seed group."""
    g = info.get("group")
    if not g:
        return None
    members = lessons.group_members(g["id"])
    return {"size": g["size"], "this_run_is_seed": g["index"] + 1,
            "test_mean_across_seeds": {aid: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items() if k != "values"}
                                       | {"values": [round(x, 2) for x in s["values"]]}
                                       for aid, s in lessons.group_spread(members).items()},
            "runs": [{"id": m["id"], "seed": m["config"]["seed"], "status": m["status"]} for m in members]}


def catalog():
    return {"visions": [{"name": v.name, "description": v.description, "params": v.params}
                        for v in visions.VISIONS.values()],
            "reward_features": [{k: f[k] for k in ("name", "default", "when", "description", "incompatible", "sign")}
                                for f in rewards.catalog()["features"]],
            "reward_presets": rewards.PRESETS,
            "algorithms": [{"name": a["name"], "description": a["description"],
                            "params": [{k: p[k] for k in ("name", "default", "min", "max", "help")} for p in a["params"]]}
                           for a in algorithms.catalog()],
            "limits": {"width/height": [6, 30], "time_budget_min": [0.1, runconfig.MAX_BUDGET_MIN]}}


def context(info):
    c = info["config"]
    others = [r for r in lessons.list_runs() if r["id"] != info["id"] and r["status"] not in lessons.ACTIVE]
    others.sort(key=lambda r: (r["lesson"] != info["lesson"], r.get("created", "")), reverse=False)
    machine = info.get("machine", {})
    return {
        "run": {**run_brief(info), "lesson": info["lesson"],
                "curves": {a["id"]: read_curve(info["path"], a["id"]) for a in c["agents"]},
                "seed_group": seed_group(info),
                "machine": {"cpus": machine.get("cpus"), "on_battery": machine.get("on_battery"),
                            "runs_sharing_the_machine": info.get("parallel", 1),
                            "cores_per_agent": {k: v["cores"] for k, v in (info.get("plan") or {}).get("agents", {}).items()}}},
        "environment": {
            "board": f"{c['width']}x{c['height']}", "start_length": 3,
            "max_apples": c["width"] * c["height"] - 3,
            "actions": "relative: straight, turn left, turn right",
            "game_ends": "crash into wall or body, or 100 x length moves without eating (starving)",
            "ga": "fitness = mean total reward over its games; rank-based tournament selection, so only the "
                  "ordering of rewards matters; network 16 hidden units",
            "dqn": "double DQN, Huber loss, replay buffer, epsilon-greedy exploration decaying over "
                   "eps_decay_steps, target network synced every target_sync steps; the saved best model "
                   "is chosen by greedy evaluation every eval_every steps; network 128x128",
        },
        "lesson_write_up": lesson_text(info["lesson"]),
        "other_runs": [run_brief(r) for r in others[:MAX_RELATED]],
        "catalog": catalog(),
    }


# ---- turning the answer into launchable runs ------------------------------------------
def apply(base, recipe):
    """This run's config with the recommended changes applied, normalized (raises ValueError)."""
    cfg = copy.deepcopy(base)
    cfg["name"] = cfg["note"] = ""
    recipe = recipe or {}
    for key in ("vision", "time_budget_min", "width", "height", "seed"):
        if recipe.get(key) is not None:
            cfg[key] = recipe[key]
    if recipe.get("reward") is not None:
        cfg["reward"] = recipe["reward"]
        for a in cfg["agents"]:
            a.pop("reward", None)
    if recipe.get("agents"):
        agents = []
        for a in recipe["agents"]:
            a = {"algorithm": a} if isinstance(a, str) else dict(a)
            same = [b for b in cfg["agents"] if b["algorithm"] == a.get("algorithm")]
            match = next((b for b in same if b["id"] == a.get("id")), same[0] if same else None)
            entry = {"id": a.get("id") or (match or {}).get("id") or a.get("algorithm"),
                     "algorithm": a.get("algorithm"),
                     "params": {**(match["params"] if match else {}), **(a.get("params") or {})}}
            reward = a.get("reward") or (match or {}).get("reward")
            if reward and cfg["reward"] is None:
                entry["reward"] = reward
            agents.append(entry)
        cfg["agents"] = agents
    for a in cfg["agents"]:
        a.pop("cores", None)  # machine-specific: let this machine plan it
    if cfg["reward"] is None:  # per-agent rewards (lessons 01-02)
        # an agent without its own reward means the recipe needs one shared reward again
        cfg["reward"] = "score" if any(not a.get("reward") for a in cfg["agents"]) else "per-agent"
    return runconfig.normalize(cfg)


def _num(v):
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def changes(base, new):
    """Plain-English list of what differs between two normalized recipes."""
    out = []
    vb, vn = visions.to_string(base["vision"]), visions.to_string(new["vision"])
    if vb != vn:
        out.append(f"vision {vb} → {vn}")
    if base["reward"] != new["reward"]:
        if base["reward"] is None or new["reward"] is None:
            out.append(f"reward {runconfig.describe(base)['reward']} → {runconfig.describe(new)['reward']}")
        else:
            for f in rewards.FEATURES:
                a, b = base["reward"].get(f), new["reward"].get(f)
                if a == b:
                    continue
                out.append(f"reward: + {f} {_num(b)}" if a is None else f"reward: − {f}" if b is None
                           else f"reward: {f} {_num(a)} → {_num(b)}")
    old = {a["id"]: a for a in base["agents"]}
    nw = {a["id"]: a for a in new["agents"]}
    for aid in old.keys() - nw.keys():
        out.append(f"drop {aid}")
    for aid in nw.keys() - old.keys():
        out.append(f"add {aid} ({nw[aid]['algorithm']})")
    for aid in old.keys() & nw.keys():
        a, b = old[aid], nw[aid]
        if a["algorithm"] != b["algorithm"]:
            out.append(f"{aid}: {a['algorithm']} → {b['algorithm']}")
            continue
        for p, v in b["params"].items():
            if a["params"].get(p) != v:
                out.append(f"{aid} {p} {_num(a['params'].get(p))} → {_num(v)}")
        if a.get("reward") != b.get("reward") and base["reward"] is None and new["reward"] is None:
            out.append(f"{aid} reward {rewards.describe(a.get('reward'))} → {rewards.describe(b.get('reward'))}")
    for key, label in (("time_budget_min", "training time (min)"), ("seed", "seed")):
        if base[key] != new[key]:
            out.append(f"{label} {_num(base[key])} → {_num(new[key])}")
    if (base["width"], base["height"]) != (new["width"], new["height"]):
        out.append(f"board {base['width']}x{base['height']} → {new['width']}x{new['height']}")
    return out


def lesson_slug(name):
    """A recommendation's lesson name without any number prefix ("03-foo" -> "foo")."""
    return re.sub(r"^\d+[-_ ]*", "", runconfig.slug(name or "")) or "next-step"


def next_lesson(slug):
    """'NN-slug' with the next free lesson number."""
    numbers = [int(m.group(1)) for n in lessons.lessons() if (m := re.match(r"^(\d+)-", n))]
    return f"{(max(numbers) + 1 if numbers else 1):02d}-{slug}"


def build_recommendations(info, raw):
    """Validate the model's recommendations. Each keeps its text plus a checked config and change list.
    Returns (recommendations that can be launched, ideas made from the ones that can't)."""
    base = info["config"]
    out, not_possible = [], []
    for i, r in enumerate((raw or [])[:4]):
        if not isinstance(r, dict):
            continue
        rec = {k: str(r.get(k) or "").strip() for k in ("title", "why", "change", "prediction", "question")}
        rec["title"] = rec["title"] or f"Recommendation {i + 1}"
        lesson = str(r.get("lesson") or "same").strip()
        rec["new_lesson"] = lesson.lower() != "same" and lesson_slug(lesson) != base["lesson"]
        rec["lesson_slug"] = lesson_slug(lesson) if rec["new_lesson"] else base["lesson"]
        rec["recipe"] = r.get("recipe") if isinstance(r.get("recipe"), dict) else {}
        try:
            config = apply(base, rec["recipe"])
            errors, warnings = runconfig.validate(config)
        except (ValueError, TypeError, KeyError) as e:
            config, errors, warnings = None, [str(e)], []
        if errors:  # it needs something the lab doesn't have: keep it as an idea instead
            kind = next((k for k in ("vision", "reward") if k in rec["recipe"]), "setting")
            not_possible.append({"title": rec["title"], "kind": kind, "why": rec["why"], "expected": rec["prediction"],
                                 "what": f"Make this possible: {rec['change'] or json.dumps(rec['recipe'])}",
                                 "effort": "medium",
                                 "note": "Suggested as a recipe change, but the lab can't run it: " + " ".join(errors)})
            continue
        rec["config"], rec["errors"], rec["warnings"] = config, errors, warnings
        rec["changes"] = changes(base, config)
        rec["launched"] = []
        out.append(rec)
    return out, not_possible


# ---- running an analysis -------------------------------------------------------------------
def text_or_none(v):
    """Models sometimes write null as the string "null" (or "N/A")."""
    v = str(v).strip() if v is not None else ""
    return None if v.lower() in ("", "null", "none", "n/a", "na") else v


def analyze(run_id, verbose=False):
    """Ask the LLM about a run and record the answer. Returns the saved analysis."""
    run_dir = lessons.run_path(run_id)
    info = lessons.read_run(run_dir)
    if info is None:
        raise FileNotFoundError(run_id)
    if info["status"] in lessons.ACTIVE:
        raise ValueError("the run is still training; analyse it once it has finished")
    st = llm.status()
    previous = read(run_dir) or {}
    started = time.time()
    write(run_dir, {**previous, "state": "running", "pid": os.getpid(), "model": st["model"],
                    "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    try:
        ctx = context(info)
        if verbose:
            print(f"[analysis] asking {st['model']} about {run_id} "
                  f"({len(json.dumps(ctx)):,} characters of context)...", flush=True)
        raw, usage = llm.chat_json([{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": "Analyse this run.\n\n" + json.dumps(ctx, indent=1)}],
                                   max_tokens=16000)
        result = {k: text_or_none(raw.get(k)) for k in ("headline", "summary", "diagnosis", "learned", "prediction_check")}
        if not result["headline"]:
            raise llm.LLMError("The LLM's answer had no headline")
        recs, not_possible = build_recommendations(info, raw.get("recommendations"))
        proposals = [p for p in (raw.get("ideas") or []) if isinstance(p, dict)][:3] + not_possible
        created = time.strftime("%Y-%m-%d %H:%M:%S")
        model = usage.get("model") or st["model"]
        analysis = {"state": "done", "model": model, "created": created,
                    "seconds": round(time.time() - started, 1), "usage": usage, "result": result,
                    "recommendations": recs, "recommended": 0 if recs else None,
                    "ideas": ideas.add_from_analysis(run_id, proposals, model, created)}
    except Exception as e:  # recorded so the dashboard can show it and offer a retry
        failed = time.strftime("%Y-%m-%d %H:%M:%S")
        if previous.get("result"):  # keep the earlier analysis; just note that redoing it failed
            analysis = {**previous, "state": "done", "last_error": str(e), "failed": failed}
        else:
            analysis = {**previous, "state": "error", "error": str(e), "model": st["model"], "failed": failed}
        write(run_dir, analysis)
        raise
    write(run_dir, analysis)
    lessons.update_lesson(info["lesson"])
    if verbose:
        print(f"[analysis] done in {analysis['seconds']}s: {result['headline']}", flush=True)
    return analysis


def generated_lesson(lesson, rec, source_id, model):
    """LESSON.md for a lesson created from a recommendation."""
    src_lesson = source_id.split("/")[0]
    number = lesson.split("-", 1)[0]
    title = f"{number} · {rec['title']}" if number.isdigit() else rec["title"]
    return (f"# {title}\n\n"
            f"**Question:** {rec['question'] or rec['why']}\n\n"
            f"**Change:** {rec['change']} (compared with run `{source_id}` of [{src_lesson}](../{src_lesson}/LESSON.md)"
            f"{': ' + '; '.join(rec['changes']) if rec['changes'] else ''}).\n\n"
            f"**Prediction:** {rec['prediction']}\n\n"
            f"_Proposed by an AI analysis ({model}) of run `{source_id}`: {rec['why']}_\n\n"
            f"## Results\n\n{lessons.START}\n{lessons.END}\n\n"
            "## What we learned\n\n_Fill this in after watching the dashboard and reading the table above._\n")


def launch(run_id, index, config=None, parallel=1, auto_analyze=None, seeds=None):
    """Queue recommendation `index` (0-based) of a run's analysis. `config` = an edited version of it.
    `seeds` defaults to the size of the analysed run's seed group. Returns the new run's (first) directory.
    The dashboard's queue starts it."""
    run_dir = lessons.run_path(run_id)
    a = read(run_dir)
    if not a or a.get("state") != "done":
        raise ValueError("this run has no finished analysis")
    try:
        rec = a["recommendations"][index]
    except (IndexError, TypeError):
        raise ValueError(f"no recommendation {index + 1}") from None
    if config is None:
        if not rec.get("config") or rec.get("errors"):
            raise ValueError("this recommendation's recipe is not valid: " + " ".join(rec.get("errors") or []))
        config = copy.deepcopy(rec["config"])
        config["name"], config["note"] = rec["title"], rec["prediction"]
        config["lesson"] = next_lesson(rec["lesson_slug"]) if rec["new_lesson"] else rec["lesson_slug"]
    config = runconfig.normalize(config)
    errors, _ = runconfig.validate(config)
    if errors:
        raise ValueError(" ".join(errors))
    lesson_file = os.path.join(lessons.LESSONS_DIR, config["lesson"], "LESSON.md")
    if not os.path.exists(lesson_file):
        os.makedirs(os.path.dirname(lesson_file), exist_ok=True)
        with open(lesson_file, "w", encoding="utf-8") as f:
            f.write(generated_lesson(config["lesson"], rec, run_id, a.get("model", "")))
    if auto_analyze is None:
        auto_analyze = llm.status()["configured"]
    if seeds is None:
        seeds = (lessons.read_run(run_dir).get("group") or {}).get("size", 1)
    new_dir = lessons.new_group(config, seeds, status="queued", parallel=parallel, auto_analyze=bool(auto_analyze),
                                parent={"run": run_id, "recommendation": index, "title": rec["title"]})[0]
    new_id = f"{config['lesson']}/{os.path.basename(new_dir)}"
    a = read(run_dir)
    a["recommendations"][index].setdefault("launched", []).append(new_id)
    write(run_dir, a)
    lessons.write_index()
    return new_dir


def payload(run_id):
    """The analysis as the dashboard shows it: each recommendation says where it would be recorded."""
    a = read(lessons.run_path(run_id))
    if not a:
        return None
    for rec in a.get("recommendations", []):
        rec["lesson_target"] = next_lesson(rec["lesson_slug"]) if rec.get("new_lesson") else rec.get("lesson_slug")
        if rec.get("config"):
            rec["config"] = {**rec["config"], "lesson": rec["lesson_target"], "name": rec["title"], "note": rec["prediction"]}
    return a


# ---- CLI ------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "show"):
        sub.add_parser(name).add_argument("run", help="<lesson>/<run-id>")
    la = sub.add_parser("launch")
    la.add_argument("run", help="<lesson>/<run-id>")
    la.add_argument("n", type=int, help="recommendation number (1 = first)")
    a = p.parse_args()
    for stream in (sys.stdout, sys.stderr):  # the model writes ×, −, → ... which a Windows console may not have
        stream.reconfigure(encoding="utf-8", errors="replace")
    if a.cmd == "run":
        try:
            analyze(a.run, verbose=True)
        except Exception as e:
            sys.exit(f"[analysis] failed: {e}")
        a.cmd = "show"
    if a.cmd == "show":
        an = payload(a.run)
        if not an:
            sys.exit("no analysis yet: python analysis.py run " + a.run)
        if an.get("state") != "done":
            sys.exit(f"analysis {an.get('state')}: {an.get('error', '')}")
        r = an["result"]
        print(f"\n{r['headline']}\n\n{r['summary']}\n\nWhy: {r['diagnosis']}\n\nLearned: {r['learned']}\n")
        for i, rec in enumerate(an["recommendations"], 1):
            status = "INVALID: " + " ".join(rec["errors"]) if rec["errors"] else "; ".join(rec["changes"]) or "no change"
            print(f"{i}. {rec['title']} -> {rec['lesson_target']}\n   {rec['why']}\n   change: {status}\n"
                  f"   prediction: {rec['prediction']}\n")
    elif a.cmd == "launch":
        run_dir = launch(a.run, a.n - 1)
        print(f"queued {os.path.relpath(run_dir, lessons.LESSONS_DIR)}; start the dashboard "
              "(python run_all.py --dashboard-only) to train it")


if __name__ == "__main__":
    main()
