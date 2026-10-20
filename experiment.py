"""Run one experiment: start every agent's trainer, enforce the time budget, evaluate, record.

    python experiment.py --lesson 03-rays --vision rays:depth=4 --reward score --agents ga,dqn --time-budget 10
    python experiment.py --config my_run.json        # a recipe (see runconfig.py)
    python experiment.py --run lessons/<lesson>/runs/<id>   # start a queued run (the dashboard does this)
    python experiment.py --stop <lesson>/<run-id>    # ask a running experiment to stop and save

Resources (worker processes, threads, replay-buffer size) are planned for this machine by
system.plan(). Every run is recorded in lessons/ even if it is stopped early.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import lessons
import runconfig
import system
from publisher import STOP_FILE

ROOT = os.path.dirname(os.path.abspath(__file__))
STOP_GRACE_S = 30


def request_stop(run_id):
    os.makedirs(lessons.live_dir(run_id), exist_ok=True)
    open(os.path.join(lessons.live_dir(run_id), STOP_FILE), "w").close()


def stop_requested(live):
    return os.path.exists(os.path.join(live, STOP_FILE))


def execute(run_dir, verbose=False):
    """Train a recorded (queued) run to the end. Returns the final run info."""
    info = lessons.read_run(run_dir)
    config, run_id = info["config"], info["id"]
    plan = system.plan(config, parallel=info.get("parallel", 1))
    live = lessons.live_dir(run_id)
    shutil.rmtree(live, ignore_errors=True)
    os.makedirs(live)
    lessons.update_run(run_dir, status="running", pid=os.getpid(), started=time.strftime("%Y-%m-%d %H:%M:%S"),
                       plan=lessons.recorded_plan(plan), machine=lessons.recorded_machine(plan["system"]))
    dirs = {"live": live, "results": os.path.join(run_dir, "results"), "models": os.path.join(run_dir, "models")}
    print(f"[experiment] {run_id}: {runconfig.describe(config)['vision_spec']} vision, "
          f"{len(config['agents'])} agent(s), {config['time_budget_min']:g} min", flush=True)
    for w in plan["warnings"]:
        print(f"[experiment] warning: {w}", flush=True)

    procs, logs = {}, []
    for agent in config["agents"]:
        p = plan["agents"][agent["id"]]
        cmd = runconfig.trainer_command(sys.executable, config, agent, p["cores"], p["params"], dirs)
        out = None if verbose else open(os.path.join(run_dir, "logs", f"{agent['id']}.log"), "w")
        logs.append(out)
        procs[agent["id"]] = subprocess.Popen(cmd, cwd=ROOT, stdout=out,
                                              stderr=subprocess.STDOUT if out else None)
        print(f"[experiment]   {agent['id']}: {p['flag']}={p['cores']}, ~{p['mem_mb']:,.0f} MB", flush=True)

    start, status, last = time.time(), "finished", 0.0
    budget = config["time_budget_min"] * 60
    try:
        while any(p.poll() is None for p in procs.values()):
            if stop_requested(live):
                status = "stopped"
                break
            if budget and time.time() - start > budget + STOP_GRACE_S:
                break
            if verbose and time.time() - last > 15:
                last = time.time()
                print(f"[experiment] {time.time() - start:5.0f}s  " + status_line(live, config), flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        status = "interrupted"
        print("\n[experiment] Ctrl+C - stopping trainers and saving what was learned...", flush=True)
    finally:
        open(os.path.join(live, STOP_FILE), "w").close()  # every trainer exits at its next check
        end = time.time() + STOP_GRACE_S
        for p in procs.values():
            while p.poll() is None and time.time() < end:
                time.sleep(0.2)
            if p.poll() is None:
                p.kill()
        for f in logs:
            if f:
                f.close()
    failed = [a for a, p in procs.items() if p.returncode not in (0, None) and status == "finished"
              and not os.path.exists(os.path.join(run_dir, "results", f"{a}.csv"))]
    if failed:
        status = "failed"
        print(f"[experiment] trainer(s) failed: {', '.join(failed)} - see {run_dir}/logs/", flush=True)
    print(f"[experiment] {status}; evaluating saved models...", flush=True)
    info = lessons.finish_run(run_dir, status, eval_games=100)
    shutil.rmtree(live, ignore_errors=True)
    print(lessons.run_table(info), flush=True)
    print(f"[experiment] recorded in {run_dir}", flush=True)
    if info.get("auto_analyze") and status in ("finished", "stopped") and group_complete(info):
        import analysis
        import llm
        if llm.status()["configured"]:
            try:
                analysis.analyze(info["id"], verbose=True)
            except Exception as e:  # the run is saved either way; the dashboard offers a retry
                print(f"[experiment] AI analysis failed: {e}", flush=True)
    return info


def group_complete(info):
    """A seed group is analysed once, by whichever of its runs finishes last (with the whole spread)."""
    g = info.get("group")
    if not g or g["size"] < 2:
        return True
    if any(m["status"] in lessons.ACTIVE for m in lessons.group_members(g["id"])):
        return False
    os.makedirs(lessons.LIVE_ROOT, exist_ok=True)
    try:  # two runs finishing at the same moment: only one claims the analysis
        os.close(os.open(os.path.join(lessons.LIVE_ROOT, f"group-{g['id'].replace('/', '__')}.analysed"),
                         os.O_CREAT | os.O_EXCL))
        return True
    except FileExistsError:
        return False


def status_line(live, config):
    parts = []
    for a in config["agents"]:
        s = lessons.read_json(os.path.join(live, f"{a['id']}_stats.json"))
        parts.append(f"{a['id']} {s['iteration_label'][:3]} {s['iteration']} best {s['best_score']} "
                     f"mean {s['mean_score_100']:.1f}" if s else f"{a['id']} starting")
    return " | ".join(parts)


def config_from_args(a):
    if a.config:
        with open(a.config, encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}
    for key, value in (("lesson", a.lesson), ("name", a.name), ("note", a.note), ("vision", a.vision),
                       ("reward", a.reward), ("time_budget_min", a.time_budget), ("seed", a.seed),
                       ("width", a.width), ("height", a.height)):
        if value is not None:
            cfg[key] = value
    if a.agents:
        cfg["agents"] = a.agents.split(",")
    return runconfig.normalize(cfg)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", help="JSON recipe (see runconfig.py); flags below override it")
    p.add_argument("--run", help="run folder of a queued run to start")
    p.add_argument("--stop", help="<lesson>/<run-id> of a running experiment to stop")
    p.add_argument("--lesson")
    p.add_argument("--name")
    p.add_argument("--note")
    p.add_argument("--vision", help="short | full | rays:depth=N | window:radius=N")
    p.add_argument("--reward", help="preset (score, shaped, ga_classic) or e.g. apple=10,death=10,hunger=0.01")
    p.add_argument("--agents", help="comma separated algorithms, e.g. ga,dqn")
    p.add_argument("--time-budget", type=float, help="minutes")
    p.add_argument("--seed", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--analyze", action="store_true", help="ask the LLM to analyse the run when it ends (see analysis.py)")
    p.add_argument("--seeds", type=int, default=1, help="queue this many runs with seeds seed, seed+1, ... (a seed group; "
                   "the dashboard trains them)")
    p.add_argument("--quiet", action="store_true", help="trainer output to the run's logs/ only")
    a = p.parse_args()

    if a.stop:
        request_stop(a.stop)
        print(f"asked {a.stop} to stop")
        return
    if a.run:
        execute(os.path.abspath(a.run), verbose=not a.quiet)
        return
    config = config_from_args(a)
    errors, warnings = runconfig.validate(config)
    for w in warnings:
        print(f"warning: {w}")
    if errors:
        raise SystemExit("\n".join(errors))
    if a.seeds > 1:  # several runs: leave them to the dashboard's queue (it trains as many at once as allowed)
        dirs = lessons.new_group(config, a.seeds, status="queued", auto_analyze=a.analyze)
        print(f"[experiment] queued a seed group of {len(dirs)} runs in {config['lesson']}; start the dashboard "
              "(python run_all.py --dashboard-only) to train them", flush=True)
        return
    execute(lessons.new_run(config, status="running", pid=os.getpid(), auto_analyze=a.analyze), verbose=not a.quiet)


if __name__ == "__main__":
    main()
