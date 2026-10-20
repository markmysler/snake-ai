"""The run queue: starts queued runs as separate experiment.py processes, `concurrency` at a time.

Experiments are detached from the dashboard: closing the dashboard does not stop training,
and a restarted dashboard picks queued and running runs up again from their run.json files.
"""
import os
import subprocess
import sys

import lessons

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def spawn(cmd, log_path):
    """Start a detached process (keeps going if the dashboard stops) that logs to log_path."""
    log = open(log_path, "w")
    kwargs = {"cwd": ROOT, "stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
    if sys.platform == "win32":  # own hidden console: no window, and Ctrl+C here doesn't reach it
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kwargs)
    log.close()
    return proc


class Jobs:
    def __init__(self, concurrency=1):
        self.concurrency = concurrency
        self.procs = {}  # run id -> Popen started by this dashboard

    def tick(self, runs):
        """Start queued runs while there are free slots. `runs` = lessons.list_runs()."""
        for rid, p in list(self.procs.items()):
            if p.poll() is not None:
                del self.procs[rid]
        running = sum(1 for r in runs if r["status"] == "running" or r["id"] in self.procs)
        queued = sorted((r for r in runs if r["status"] == "queued" and r["id"] not in self.procs),
                        key=lambda r: r.get("created", ""))
        for info in queued[:max(0, self.concurrency - running)]:
            self.start(info)

    def start(self, info):
        self.procs[info["id"]] = spawn([sys.executable, "experiment.py", "--run", info["path"], "--quiet"],
                                       os.path.join(info["path"], "logs", "experiment.log"))

    def review_ideas(self):
        """Start an AI review of the ideas list (python ideas.py review). Returns the process."""
        import ideas
        os.makedirs(ideas.IDEAS_DIR, exist_ok=True)
        return spawn([sys.executable, "ideas.py", "review"], os.path.join(ideas.IDEAS_DIR, "review.log"))

    def analyze(self, info):
        """Start an AI analysis of a finished run (python analysis.py run <id>). Returns the process."""
        os.makedirs(os.path.join(info["path"], "logs"), exist_ok=True)
        return spawn([sys.executable, "analysis.py", "run", info["id"]],
                     os.path.join(info["path"], "logs", "analysis.log"))

    def stop(self, info):
        """Running: ask it to stop and save. Queued: remove it (nothing was trained yet)."""
        if info["status"] == "running" or info["id"] in self.procs:
            from experiment import request_stop
            request_stop(info["id"])
        elif info["status"] == "queued":
            lessons.update_run(info["path"], status="cancelled")
            lessons.delete_run(info["id"])
