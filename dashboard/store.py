"""Where the dashboard gets policies and chart history for one run.

  LiveSource  what a run's trainers are publishing right now (its live/ folder)
  SavedRun    the best models a run saved (its models/ folder)
  Watcher     both, plus the run's stats CSVs: everything the Watch view shows
"""
import csv
import io
import os
import threading
from collections import OrderedDict

import numpy as np

import lessons
from policies import MLPPolicy, load_policy
from publisher import read_json

HISTORY_KEYS = ["elapsed_s", "total_steps", "total_games", "best_score",
                "mean_score_100", "steps_per_sec", "iteration",
                "end_wall", "end_body", "end_starved", "end_won"]  # (ends: None in runs recorded before them)
KEEP_VERSIONS = 64  # old versions stay loaded so frozen boards can keep playing them
KEEP_POPULATIONS = 4
META_KEYS = ("version", "iteration", "iteration_label", "vision", "width", "height",
             "fitness", "eval_mean", "rank", "agent")


def slim(meta):
    return {k: meta.get(k) for k in META_KEYS if k in meta}


def history_row(r):
    return [float(r[k]) if r.get(k) not in (None, "") else None for k in HISTORY_KEYS]


def _put(cache, key, value, limit):
    cache[key] = value
    while len(cache) > limit:
        cache.popitem(last=False)


class LiveSource:
    def __init__(self, live_dir, agents):
        self.live_dir, self.agents = live_dir, agents
        self.current = {a: None for a in agents}            # agent -> newest meta
        self.cache = {a: OrderedDict() for a in agents}     # version -> (act, meta)
        self.population = {}                                # agent -> newest population meta
        self.pop_cache = {}                                 # agent -> OrderedDict(version -> genomes, meta)
        self.lock = threading.Lock()
        self.last_error = {}

    def poll(self):
        """Load anything newly published. Returns {"policies": {...}, "population": {...}}."""
        changes = {}
        for a in self.agents:
            meta = read_json(os.path.join(self.live_dir, f"{a}_meta.json"))
            cur = self.current[a]
            if meta and not (cur and cur["version"] == meta["version"]):
                try:
                    act, extra = load_policy(os.path.join(self.live_dir, meta["file"]), meta)
                    meta = slim({**extra, **meta, "agent": a})
                    with self.lock:
                        _put(self.cache[a], meta["version"], (act, meta), KEEP_VERSIONS)
                        self.current[a] = meta
                    changes.setdefault("policies", {})[a] = meta
                except Exception as e:  # half-written or locked file: try again next poll
                    if self.last_error.get(a) != meta["version"]:
                        self.last_error[a] = meta["version"]
                        print(f"[dashboard] could not load {a} weights ({e}); will retry", flush=True)
            pop = self._poll_population(a)  # populations change every generation
            if pop:
                changes.setdefault("population", {})[a] = pop
        return changes

    def _poll_population(self, agent):
        meta = read_json(os.path.join(self.live_dir, f"{agent}_population.json"))
        if not meta or (self.population.get(agent) or {}).get("version") == meta["version"]:
            return None
        try:
            genomes = np.load(os.path.join(self.live_dir, f"{agent}_population.npy"))
        except Exception:
            return None
        if len(genomes) != len(meta["fitness"]):  # npy already from the next generation
            return None
        with self.lock:
            _put(self.pop_cache.setdefault(agent, OrderedDict()), meta["version"], (genomes, meta), KEEP_POPULATIONS)
            self.population[agent] = meta
        return self.population_summary()[agent]

    def population_summary(self):
        return {a: {"version": p["version"], "generation": p["generation"], "size": len(p["fitness"])}
                for a, p in self.population.items()}

    def policies(self):
        return {k: v for k, v in self.current.items() if v}

    def policy(self, agent, version=None):
        with self.lock:
            cache = self.cache.get(agent)
            if not cache:
                return None
            if version in cache:
                return cache[version]
            cur = self.current[agent]
            return cache[cur["version"]] if cur else None

    def genome(self, agent, rank, version=None):
        with self.lock:
            pops = self.pop_cache.get(agent)
            if not pops:
                return None
            genomes, meta = pops.get(version) or pops[self.population[agent]["version"]]
        rank = max(0, min(int(rank), len(genomes) - 1))
        info = {"version": meta["version"], "iteration": meta["generation"], "iteration_label": "generation",
                "rank": rank, "fitness": meta["fitness"][rank], "vision": meta.get("vision", "short"),
                "width": meta.get("width", 10), "height": meta.get("height", 10), "agent": agent}
        return MLPPolicy(genomes[rank]).act, info


class SavedRun:
    """A run's saved best models. Loaded on first use."""

    def __init__(self, run_dir, config):
        self.dir, self.config = run_dir, config
        self._policies = None
        self.lock = threading.Lock()

    def _load(self):
        with self.lock:
            if self._policies is None:
                self._policies = {}
                for a in self.config["agents"]:
                    path = lessons.model_path(self.dir, a)
                    if not os.path.exists(path):
                        continue
                    try:
                        act, meta = load_policy(path)
                    except Exception as e:
                        print(f"[dashboard] could not load {path}: {e}", flush=True)
                        continue
                    meta = {**meta, "version": f"saved:{a['id']}", "agent": a["id"],
                            "iteration": meta.get("generation") or meta.get("episode"),
                            "iteration_label": "generation" if "generation" in meta else "episode"}
                    self._policies[a["id"]] = (act, slim(meta))
        return self._policies

    def policies(self):
        return {k: meta for k, (_, meta) in self._load().items()}

    def policy(self, agent, version=None):
        return self._load().get(agent)


class Watcher:
    """Everything about one run: live weights while it trains, saved models after, stats history."""

    def __init__(self, info):
        self.id, self.dir, self.status = info["id"], info["path"], info["status"]
        self.agents = [a["id"] for a in info["config"]["agents"]]
        self.live = LiveSource(lessons.live_dir(self.id), self.agents)
        self.saved = SavedRun(self.dir, info["config"])
        self.tails = {a: CsvTail(os.path.join(self.dir, "results", f"{a}.csv")) for a in self.agents}
        self.history = {a: [] for a in self.agents}
        self.stats = {}
        self.apply(self.collect())

    @property
    def active(self):
        return self.status in lessons.ACTIVE

    def collect(self):
        """Runs in a worker thread: everything that changed since the last poll."""
        out = self.live.poll() if self.active else {}
        for a, tail in self.tails.items():
            rows, reset = tail.poll()
            if reset:
                out.setdefault("reset", []).append(a)
            if rows:
                out.setdefault("history", {})[a] = [history_row(r) for r in rows]
            if self.active:
                rec = read_json(os.path.join(self.live.live_dir, f"{a}_stats.json"))
                if rec and rec != self.stats.get(a):
                    out.setdefault("stats", {})[a] = rec
        return out

    def apply(self, ch):
        for a in ch.get("reset", []):
            self.history[a] = []
        for a, rows in ch.get("history", {}).items():
            self.history[a].extend(rows)
        self.stats.update(ch.get("stats", {}))

    def policies(self):
        return self.live.policies() if self.active else self.saved.policies()

    def policy(self, agent, version=None):
        return (self.live.policy(agent, version) if self.active else None) or self.saved.policy(agent)

    def snapshot(self):
        return {"history": self.history, "stats": self.stats, "policies": self.policies(),
                "population": self.live.population_summary() if self.active else {}}


def read_history(info):
    """Every stats row of every agent of a run, read from its CSVs now."""
    out = {}
    for a in info["config"]["agents"]:
        rows, _ = CsvTail(os.path.join(info["path"], "results", f"{a['id']}.csv")).poll()
        out[a["id"]] = [history_row(r) for r in rows]
    return out


class CsvTail:
    """Incrementally reads a results CSV; detects truncation by a new run."""

    def __init__(self, path):
        self.path, self.pos, self.partial, self.header = path, 0, b"", None

    def poll(self):
        """Returns (new_rows as dicts, reset_flag)."""
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return [], False
        reset = size < self.pos
        if reset:
            self.pos, self.partial, self.header = 0, b"", None
        if size == self.pos:
            return [], reset
        try:
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                data = f.read()
        except OSError:
            return [], reset
        self.pos += len(data)
        lines = (self.partial + data).split(b"\n")
        self.partial = lines.pop()  # incomplete last line (if any)
        rows = list(csv.reader(io.StringIO(b"\n".join(lines).decode(errors="replace"))))
        if self.header is None and rows:
            self.header, rows = rows[0], rows[1:]
        return [dict(zip(self.header, r)) for r in rows if len(r) == len(self.header)], reset
