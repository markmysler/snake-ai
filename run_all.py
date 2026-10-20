"""One command: start the dashboard, queue a training run, open the browser on it.

    python run_all.py                                        # GA + DQN, short vision, score reward, 10 min
    python run_all.py --lesson 03-rays --vision rays:depth=4 --reward shaped --time-budget 10
    python run_all.py --dashboard-only                       # just the lab (launch runs from the page)
    python run_all.py --time-budget 3 --exit-when-done --no-browser

The run trains as its own process (see experiment.py) and is recorded in lessons/. Ctrl+C
stops it early, still evaluates and saves it, then closes the dashboard.
"""
import argparse
import os
import socket
import subprocess
import sys
import time
import webbrowser

import experiment
import lessons
import runconfig
import system

ROOT = os.path.dirname(os.path.abspath(__file__))
DONE = ("finished", "stopped", "interrupted", "failed")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dashboard-only", action="store_true", help="don't queue a run")
    p.add_argument("--config", help="JSON recipe (see runconfig.py); flags below override it")
    p.add_argument("--lesson", default="scratch", help="lesson folder the run is recorded in")
    p.add_argument("--name")
    p.add_argument("--note", help="your prediction, shown in LESSON.md")
    p.add_argument("--vision", help="short | full | rays:depth=N | window:radius=N")
    p.add_argument("--reward", help="preset (score, shaped, ga_classic) or e.g. apple=10,death=10,hunger=0.01")
    p.add_argument("--agents", help="comma separated algorithms (default: ga,dqn)")
    p.add_argument("--time-budget", type=float, help="minutes of training (default 10)")
    p.add_argument("--seed", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--analyze", action="store_true", help="ask the LLM to analyse the run when it ends (see analysis.py)")
    p.add_argument("--seeds", type=int, default=1, help="train the recipe with this many seeds (a seed group)")
    p.add_argument("--parallel", type=int, default=1, help="runs that may train at the same time")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--exit-when-done", action="store_true", help="close the dashboard when the run ends")
    return p.parse_args()


def free_port(port, tries=20):
    for candidate in range(port, port + tries):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", candidate))
                return candidate
            except OSError:
                continue
    raise SystemExit(f"no free port in {port}..{port + tries - 1}")


def wait_for_port(port, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.3)
    return False


def wait_for_run(run_dir, config):
    """Print a status line every 15 s until the run is recorded. Returns its final info."""
    last = 0.0
    while True:
        info = lessons.read_run(run_dir)
        if info["status"] in DONE and info.get("finished"):
            return info
        if info["status"] == "running" and time.time() - last > 15:
            last = time.time()
            print(f"[run_all] {experiment.status_line(lessons.live_dir(info['id']), config)}", flush=True)
        time.sleep(1)


def main():
    a = parse_args()
    os.chdir(ROOT)
    info = system.detect()
    print(f"[run_all] this machine: {info['cpus']} CPUs, "
          f"{info['ram_total_mb'] or '?'} MB RAM ({info['ram_available_mb'] or '?'} MB free)"
          + (", on battery" if info["on_battery"] else ""), flush=True)
    run_id = config = None
    if not a.dashboard_only:
        config = experiment.config_from_args(a)
        errors, warnings = runconfig.validate(config)
        for w in warnings:
            print(f"[run_all] warning: {w}", flush=True)
        if errors:
            raise SystemExit("\n".join(errors))

    port = free_port(a.port)
    if port != a.port:
        print(f"[run_all] port {a.port} is busy, using {port}", flush=True)
    dash = subprocess.Popen([sys.executable, "-m", "dashboard.server", "--port", str(port),
                             "--parallel", str(a.parallel)])
    if not wait_for_port(port) or dash.poll() is not None:
        raise SystemExit("[run_all] the dashboard did not start")
    run_dirs = []
    if config:
        run_dirs = lessons.new_group(config, a.seeds, status="queued", parallel=a.parallel,  # the dashboard starts them
                                     auto_analyze=a.analyze)
        run_id = f"{config['lesson']}/{os.path.basename(run_dirs[0])}"
        seeds = f", {len(run_dirs)} seeds" if len(run_dirs) > 1 else ""
        print(f"[run_all] queued {run_id} ({runconfig.describe(config)['vision_spec']} vision, "
              f"{config['time_budget_min']:g} min{seeds})", flush=True)
    url = f"http://127.0.0.1:{port}/" + (f"#watch/{run_id}" if run_id else "#runs")
    print(f"[run_all] dashboard: {url}", flush=True)
    if not a.no_browser:
        webbrowser.open(url)

    try:
        for run_dir in run_dirs:
            final = wait_for_run(run_dir, config)
            print(f"\n[run_all] {final['id']} {final['status']}: saved best models on 100 fixed-seed games\n\n"
                  f"{lessons.run_table(final)}\n\n[run_all] recorded in {run_dir}", flush=True)
        if run_dirs and a.exit_when_done:
            return
        print(f"[run_all] dashboard still running at {url} - Ctrl+C to quit", flush=True)
        dash.wait()
    except KeyboardInterrupt:
        active = [lessons.read_run(d) for d in run_dirs]
        active = [i for i in active if i and i["status"] in ("queued", "running")]
        if active:
            print("\n[run_all] Ctrl+C - stopping; a run that has started is still evaluated and saved...", flush=True)
            for info in active:
                if info["status"] == "running":
                    experiment.request_stop(info["id"])
                else:  # not started yet: nothing to save
                    lessons.update_run(info["path"], status="cancelled")
                    lessons.delete_run(info["id"])
            try:
                for info in active:
                    if info["status"] == "running":
                        print(lessons.run_table(wait_for_run(info["path"], config)), flush=True)
            except KeyboardInterrupt:
                print("[run_all] not waiting; the run finishes saving in the background.", flush=True)
    finally:
        dash.terminate()
        try:
            dash.wait(timeout=10)
        except subprocess.TimeoutExpired:
            dash.kill()
        print("[run_all] dashboard stopped.", flush=True)


if __name__ == "__main__":
    main()
