"""Server-side state: the run list, the queue, one Watcher per watched run, and SSE subscribers."""
import asyncio
import json
import time

import analysis
import ideas
import lessons
import runconfig
import system
from dashboard.jobs import Jobs
from dashboard.store import HISTORY_KEYS, Watcher

POLL_SECONDS = 0.5
RUNS_EVERY = 4  # re-read the run list every 4 polls (2 s)


def summary(info):
    """What the browser needs to list, label and compare a run."""
    c = info["config"]
    d = runconfig.describe(c)
    started = info.get("started")
    elapsed = None
    if info["status"] == "running" and started:
        elapsed = time.time() - time.mktime(time.strptime(started, "%Y-%m-%d %H:%M:%S"))
    machine = info.get("machine", {})
    return {"id": info["id"], "lesson": info["lesson"], "lesson_title": lessons.lesson_title(info["lesson"]),
            "run_id": info["run_id"], "name": info.get("name", ""), "note": info.get("note", ""),
            "status": info["status"], "created": info.get("created"), "started": started,
            "finished": info.get("finished"), "elapsed_s": elapsed, "budget_min": c["time_budget_min"],
            "vision": {"spec": d["vision_spec"], "label": d["vision"], "inputs": d["inputs"]},
            "reward": d["reward"], "agents": d["agents"], "config": c,
            "evaluation": info.get("evaluation", {}), "final": info.get("final", {}),
            "machine": {"cpus": machine.get("cpus", machine.get("cpu_count")), "on_battery": machine.get("on_battery")},
            "plan": info.get("plan"), "parallel": info.get("parallel", 1),
            "analysis": analysis.brief(info["path"]), "auto_analyze": bool(info.get("auto_analyze")),
            "parent": info.get("parent"), "group": info.get("group"), "course": info.get("course")}


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


class Hub:
    def __init__(self, concurrency=1):
        self.jobs = Jobs(concurrency)
        self.infos, self.runs, self.runs_key = {}, [], None
        self.watchers = {}
        self.subscribers = {}  # asyncio.Queue -> watched run id (or None)
        self.system = system.detect()
        self.refresh_runs()

    # ---- runs ---------------------------------------------------------------------
    def refresh_runs(self):
        """Re-read every run.json; start queued runs if there is room. Returns True if anything changed."""
        infos = lessons.list_runs()
        self.jobs.tick(infos)
        self.infos = {i["id"]: i for i in infos}
        self.runs = [summary(i) for i in infos]
        self.ideas_open = len(ideas.open_ideas())
        key = json.dumps([[(r["id"], r["status"], r["name"], r["note"], bool(r["evaluation"]), r["analysis"])
                           for r in self.runs], self.ideas_open])
        changed, self.runs_key = key != self.runs_key, key
        return changed

    def queue_state(self):
        return {"concurrency": self.jobs.concurrency, "ideas_open": getattr(self, "ideas_open", 0),
                "running": [r["id"] for r in self.runs if r["status"] == "running"],
                "queued": [r["id"] for r in self.runs if r["status"] == "queued"]}

    def watcher(self, run_id):
        info = self.infos.get(run_id)
        if info is None:
            return None
        w = self.watchers.get(run_id)
        if w is None or w.status != info["status"]:  # (re)load when the run starts or ends
            w = self.watchers[run_id] = Watcher(info)
        return w

    def watch_payload(self, run_id):
        w = self.watcher(run_id)
        if w is None:
            return None
        run = next(r for r in self.runs if r["id"] == run_id)
        return {"run": run, "history_keys": HISTORY_KEYS, **w.snapshot()}

    def init_payload(self, run_id):
        return {"runs": self.runs, "queue": self.queue_state(), "system": self.system,
                "watch": self.watch_payload(run_id) if run_id else None}

    # ---- SSE --------------------------------------------------------------------
    def subscribe(self, run_id):
        q = asyncio.Queue()
        self.subscribers[q] = run_id
        return q

    def unsubscribe(self, q):
        self.subscribers.pop(q, None)
        watched = set(self.subscribers.values())
        for rid in [r for r in self.watchers if r not in watched]:
            del self.watchers[rid]  # nobody is looking: free its models and history

    def send(self, event, data, run_id=...):
        msg = sse(event, data)
        for q, rid in list(self.subscribers.items()):
            if run_id is ... or rid == run_id:
                q.put_nowait(msg)

    async def run(self):
        tick = 0
        while True:
            try:
                await self.poll(tick)
            except Exception as e:  # never let one bad read kill the poller
                print(f"[dashboard] poll error: {e!r}", flush=True)
            tick += 1
            await asyncio.sleep(POLL_SECONDS)

    async def poll(self, tick):
        if tick % RUNS_EVERY == 0:
            before = {rid: w.status for rid, w in self.watchers.items()}
            if await asyncio.to_thread(self.refresh_runs):
                self.send("runs", {"runs": self.runs, "queue": self.queue_state()})
            for rid, status in before.items():
                info = self.infos.get(rid)
                if info is None or info["status"] != status:  # finished, stopped or deleted
                    self.watchers.pop(rid, None)
                    payload = await asyncio.to_thread(self.watch_payload, rid) if info else None
                    self.send("watch", payload, run_id=rid)
        for rid in set(self.subscribers.values()):
            w = self.watchers.get(rid)
            if w is not None and w.active:
                ch = await asyncio.to_thread(w.collect)
                if ch:
                    w.apply(ch)
                    self.send("update", ch, run_id=rid)
