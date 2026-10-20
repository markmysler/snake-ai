"""Live publishing shared by all trainers.

Each trainer writes, never waiting on any reader:
  live/<method>_stats.json      latest stats record (+ extras)
  live/<method>_weights.<ext>   current best policy weights
  live/<method>_meta.json       version/iteration of those weights (written after the weights)
  live/ga_population.{npy,json} GA only: last generation ranked by fitness
  results/<method>.csv          every stats record, for reloading charts later
All live files use atomic temp-file + rename writes. A file live/STOP asks trainers to stop.
"""
import csv
import json
import os
import time

STATS_FIELDS = ["method", "iteration", "iteration_label", "total_steps", "total_games",
                "elapsed_s", "best_score", "mean_score_100", "steps_per_sec",
                "end_wall", "end_body", "end_starved", "end_won"]  # how the last 100 games ended (counts)
STOP_FILE = "STOP"


def atomic_write(path, write_fn, retries=20):
    """Write via a temp file + os.replace. Retries briefly because on Windows a reader
    (or OneDrive) holding the target open makes the rename fail. Returns success."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        write_fn(f)
    for i in range(retries):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(0.005 * (i + 1))
    try:
        os.remove(tmp)
    except OSError:
        pass
    return False


def write_json(path, obj):
    data = json.dumps(obj).encode()
    return atomic_write(path, lambda f: f.write(data))


def read_json(path, retries=3):
    for _ in range(retries):
        try:
            with open(path, "rb") as f:
                return json.loads(f.read())
        except FileNotFoundError:
            return None
        except (PermissionError, ValueError):
            time.sleep(0.01)
    return None


class Publisher:
    def __init__(self, method, iteration_label, live_dir="live", results_dir="results"):
        self.method, self.iteration_label = method, iteration_label
        self.live_dir = live_dir
        os.makedirs(live_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        for name in os.listdir(live_dir):  # stale files from an earlier run
            if name.startswith(method + "_"):
                try:
                    os.remove(os.path.join(live_dir, name))
                except OSError:
                    pass
        self.start = time.time()
        self.run_id = f"{int(self.start * 1000)}"
        self.weights_version = 0
        self._last_t, self._last_steps = self.start, 0
        self._csv_file = open(os.path.join(results_dir, f"{method}.csv"), "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(STATS_FIELDS)
        self._csv_file.flush()

    def path(self, name):
        return os.path.join(self.live_dir, name)

    def elapsed(self):
        return time.time() - self.start

    def stats(self, iteration, total_steps, total_games, best_score, mean_score, ends=None, **extra):
        """ends: {"wall": n, "body": n, "starved": n, "won": n} for the same last-100 games as mean_score."""
        now = time.time()
        dt = max(now - self._last_t, 1e-9)
        sps = (total_steps - self._last_steps) / dt
        self._last_t, self._last_steps = now, total_steps
        rec = {"method": self.method, "iteration": iteration,
               "iteration_label": self.iteration_label, "total_steps": total_steps,
               "total_games": total_games, "elapsed_s": round(now - self.start, 2),
               "best_score": best_score, "mean_score_100": round(float(mean_score), 3),
               "steps_per_sec": round(sps, 1),
               **{f"end_{k}": (ends or {}).get(k, "") for k in ("wall", "body", "starved", "won")}}
        self._csv.writerow([rec[k] for k in STATS_FIELDS])
        self._csv_file.flush()
        write_json(self.path(f"{self.method}_stats.json"), {**rec, **extra, "run_id": self.run_id})
        return rec

    def weights(self, write_fn, ext, iteration, **meta):
        """Publish policy weights: write_fn(file) serializes them; meta goes to the sidecar."""
        self.weights_version += 1
        name = f"{self.method}_weights.{ext}"
        if not atomic_write(self.path(name), write_fn):
            return
        write_json(self.path(f"{self.method}_meta.json"), {
            "version": f"{self.run_id}:{self.weights_version}", "iteration": iteration,
            "iteration_label": self.iteration_label, "file": name, "time": time.time(), **meta})

    def stop_requested(self):
        """True if live/STOP was created after this trainer started."""
        try:
            return os.path.getmtime(self.path(STOP_FILE)) >= self.start - 1
        except OSError:
            return False

    def finish(self):
        """Mark the latest stats as final (run finished) and close the CSV."""
        path = self.path(f"{self.method}_stats.json")
        last = read_json(path)
        if last:
            write_json(path, {**last, "final": True})
        self._csv_file.close()
