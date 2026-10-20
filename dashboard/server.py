"""Snake AI lab: browse, launch, watch and compare training runs.

    python -m dashboard.server              # http://127.0.0.1:8000
    python -m dashboard.server --port 8001 --parallel 2

Runs are started as separate processes (see dashboard/jobs.py) and keep training if this
server stops. Everything is stored in lessons/ (see lessons.py).
"""
import argparse
import asyncio
import os
import random
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402
from fastapi import Body, FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import algorithms  # noqa: E402
import analysis  # noqa: E402
import course  # noqa: E402
import glossary  # noqa: E402
import jev  # noqa: E402
import ideas  # noqa: E402
import lessons  # noqa: E402
import llm  # noqa: E402
import rewards  # noqa: E402
import runconfig  # noqa: E402
import system  # noqa: E402
import vision as visions  # noqa: E402
from dashboard import inspector as inspect_mode  # noqa: E402
from dashboard.hub import Hub, sse  # noqa: E402
from dashboard.replay import play_game  # noqa: E402
from policies import RecordedPolicy  # noqa: E402
from dashboard.store import HISTORY_KEYS, read_history  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"


# What a read-only demo may still do: look things up (GET) and plan a recipe without launching it.
READ_ONLY_ALLOWED = {("POST", "/api/plan")}


def create_app(parallel=1, read_only=False):
    hub = Hub(concurrency=0 if read_only else parallel)  # a demo never starts queued runs
    games = inspect_mode.Inspector()

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(hub.run())
        yield
        task.cancel()

    app = FastAPI(lifespan=lifespan)
    app.state.hub = hub

    if read_only:
        @app.middleware("http")
        async def refuse_changes(request, call_next):
            if request.method not in ("GET", "HEAD") and (request.method, request.url.path) not in READ_ONLY_ALLOWED:
                return JSONResponse({"detail": "This is a read-only demo: nothing can be launched, edited or deleted. "
                                     "Install the lab to train your own snakes."}, status_code=403)
            return await call_next(request)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def find(run_id):
        info = hub.infos.get(run_id)
        if info is None:
            hub.refresh_runs()
            info = hub.infos.get(run_id)
        if info is None:
            raise HTTPException(404, f"no run {run_id}")
        return info

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    # ---- what can be launched, and on what machine ----------------------------------
    @app.get("/api/catalog")
    def catalog():
        return {"visions": [v.info() for v in visions.VISIONS.values()], "rewards": rewards.catalog(),
                "algorithms": algorithms.catalog(), "defaults": runconfig.normalize({}),
                "lessons": [{"name": n, "title": lessons.lesson_title(n)} for n in lessons.lessons()],
                "llm": {**llm.status(), "configured": llm.status()["configured"] and not read_only},
                "glossary": glossary.entries(), "course": course.payload(), "read_only": read_only}

    @app.get("/api/system")
    def system_info():
        hub.system = system.detect()
        return hub.system

    @app.post("/api/plan")
    def plan(body: dict = Body(...)):
        """Normalize + validate a recipe and show how it would use this machine."""
        try:
            config = runconfig.normalize(body.get("config", {}))
        except (ValueError, TypeError, KeyError) as e:
            return {"errors": [str(e)], "warnings": []}
        errors, warnings = runconfig.validate(config)
        p = system.plan(config, hub.system, parallel=int(body.get("parallel") or hub.jobs.concurrency))
        costs = {a["id"]: {**jev.estimate(a["params"]["games"], a["params"]["batch"]), "cap": a["params"]["max_cost"]}
                 for a in config["agents"] if a["algorithm"] == "jev"}  # paid API calls: shown before starting
        return {"config": config, "describe": runconfig.describe(config), "errors": errors,
                "warnings": warnings + p["warnings"], "plan": p, "costs": costs}

    # ---- runs -----------------------------------------------------------------------
    @app.get("/api/runs")
    def runs():
        return {"runs": hub.runs, "queue": hub.queue_state()}

    @app.post("/api/runs")
    def launch(body: dict = Body(...)):
        try:
            config = runconfig.normalize(body.get("config", {}))
        except (ValueError, TypeError, KeyError) as e:
            raise HTTPException(400, str(e))
        errors, _ = runconfig.validate(config)
        if errors:
            raise HTTPException(400, " ".join(errors))
        auto = bool(body.get("auto_analyze")) and llm.status()["configured"]
        origin, seeds = body.get("origin"), int(body.get("seeds") or 1)
        try:
            if origin:  # an edited AI recommendation: also records where it came from
                run_dir = analysis.launch(origin["run"], int(origin["index"]), config=config,
                                          parallel=hub.jobs.concurrency, auto_analyze=auto, seeds=seeds)
            else:
                run_dir = lessons.new_group(config, seeds, status="queued", parallel=hub.jobs.concurrency,
                                            auto_analyze=auto)[0]
        except (ValueError, KeyError, TypeError) as e:
            raise HTTPException(400, str(e))
        hub.refresh_runs()
        return {"id": f"{config['lesson']}/{os.path.basename(run_dir)}", "queue": hub.queue_state()}

    @app.post("/api/course/run")
    def course_run(lesson: str):
        """Queue a course lesson's recipes (those without runs yet) as seed groups."""
        try:
            dirs = course.queue(lesson, parallel=hub.jobs.concurrency)
        except ValueError as e:
            raise HTTPException(400, str(e))
        hub.refresh_runs()
        return {"queued": len(dirs), "queue": hub.queue_state()}

    @app.get("/api/run")
    def run(id: str):
        find(id)
        return next(r for r in hub.runs if r["id"] == id)

    @app.patch("/api/run")
    def edit(id: str, body: dict = Body(...)):
        find(id)
        try:
            new_id = lessons.edit_run(id, name=body.get("name"), note=body.get("note"), lesson=body.get("lesson"))
        except ValueError as e:
            raise HTTPException(400, str(e))
        hub.watchers.pop(id, None)
        hub.refresh_runs()
        return {"id": new_id}

    @app.delete("/api/run")
    def delete(id: str):
        info = find(id)
        try:
            if info["status"] == "queued":
                hub.jobs.stop(info)
            else:
                lessons.delete_run(id)
        except ValueError as e:
            raise HTTPException(400, str(e))
        hub.watchers.pop(id, None)
        hub.refresh_runs()
        return {"deleted": id}

    @app.post("/api/run/stop")
    def stop(id: str):
        hub.jobs.stop(find(id))
        hub.refresh_runs()
        return {"stopping": id}

    @app.post("/api/queue")
    def queue(body: dict = Body(...)):
        hub.jobs.concurrency = max(1, min(int(body.get("concurrency", 1)), hub.system["cpus"]))
        hub.refresh_runs()
        return hub.queue_state()

    # ---- AI analysis (see analysis.py) ------------------------------------------------------
    @app.get("/api/analysis")
    def get_analysis(id: str):
        info = find(id)
        return {"id": id, "llm": llm.status(), "analysis": analysis.payload(info["id"])}

    @app.post("/api/analysis")
    def start_analysis(id: str):
        """Analyse a finished run in a separate process; progress shows up in the run list."""
        info = find(id)
        if not llm.status()["configured"]:
            raise HTTPException(400, f"No LLM configured: {llm.SETUP_HINT}")
        if info["status"] in lessons.ACTIVE:
            raise HTTPException(400, "The run is still training; analyse it once it has finished.")
        current = analysis.read(info["path"])
        if current and current.get("state") == "running":
            raise HTTPException(409, "This run is already being analysed.")
        proc = hub.jobs.analyze(info)
        analysis.write(info["path"], {**(current or {}), "state": "running", "pid": proc.pid,
                                      "model": llm.status()["model"]})
        hub.refresh_runs()
        return {"analyzing": id}

    @app.post("/api/analysis/launch")
    def launch_recommendation(id: str, index: int):
        """One click: queue an AI-recommended run exactly as recommended."""
        info = find(id)
        try:
            run_dir = analysis.launch(info["id"], index, parallel=hub.jobs.concurrency)
        except ValueError as e:
            raise HTTPException(400, str(e))
        hub.refresh_runs()
        new_id = os.path.relpath(run_dir, lessons.LESSONS_DIR).replace(os.sep, "/").replace("/runs/", "/")
        return {"id": new_id, "queue": hub.queue_state()}

    # ---- ideas: improvements that need new code (see ideas.py) ---------------------------------
    @app.get("/api/ideas")
    def get_ideas(review: str | None = None):
        return {"ideas": ideas.load(), "review": ideas.read_review(review), "reviews": ideas.reviews(),
                "llm": llm.status(), "statuses": ideas.STATUSES, "kinds": ideas.KINDS}

    @app.patch("/api/idea")
    def edit_idea(id: str, body: dict = Body(...)):
        try:
            idea = ideas.set_status(id, body.get("status"))
        except KeyError:
            raise HTTPException(404, f"no idea {id}")
        except ValueError as e:
            raise HTTPException(400, str(e))
        hub.refresh_runs()
        return idea

    @app.post("/api/ideas/review")
    def review_ideas():
        """Ask the LLM which ideas to build, in a separate process; poll GET /api/ideas for the result."""
        if not llm.status()["configured"]:
            raise HTTPException(400, f"No LLM configured: {llm.SETUP_HINT}")
        if not ideas.open_ideas():
            raise HTTPException(400, "There are no open ideas to review.")
        current = ideas.read_review()
        if current and current.get("state") == "running":
            raise HTTPException(409, "A review is already running.")
        proc = hub.jobs.review_ideas()
        lessons.write_json(os.path.join(ideas.IDEAS_DIR, "review.json"),
                           {**(current or {}), "state": "running", "pid": proc.pid, "model": llm.status()["model"]})
        return {"reviewing": True}

    @app.get("/api/history")
    def history(id: str):
        """A run's full stats history, read fresh (used by Compare, which has no live stream)."""
        info = find(id)
        return {"id": id, "history_keys": HISTORY_KEYS, "history": read_history(info)}

    @app.get("/api/lesson")
    def lesson(name: str):
        path = os.path.join(lessons.LESSONS_DIR, runconfig.slug(name), "LESSON.md")
        if not os.path.isfile(path):
            raise HTTPException(404, f"no lesson {name}")
        with open(path, encoding="utf-8") as f:
            return {"name": name, "title": lessons.lesson_title(name), "markdown": f.read()}

    # ---- live data + replays ---------------------------------------------------------
    @app.get("/api/events")
    async def events(run: str | None = None):
        q = hub.subscribe(run)

        async def stream():
            try:
                yield sse("init", await asyncio.to_thread(hub.init_payload, run))
                while True:
                    try:
                        yield await asyncio.wait_for(q.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                hub.unsubscribe(q)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/game")
    def game(run: str, agent: str, version: str | None = None, rank: int | None = None):
        """Play one replay game with the requested (or newest) weights. Sync => thread pool."""
        find(run)
        w = hub.watcher(run)
        got = w.live.genome(agent, rank, version) if rank is not None else w.policy(agent, version)
        if got is None:
            return {"waiting": True}
        act, meta = got
        seed = random.randrange(2**31)
        if isinstance(getattr(act, "__self__", None), RecordedPolicy):  # jev: replay one of the games it played
            seed = random.choice(act.__self__.seeds())
            act = act.__self__.for_seed(seed).act
        g = play_game(act, meta.get("width", 10), meta.get("height", 10), seed, meta.get("vision", "short"))
        return {"game": g, "meta": meta}

    @app.get("/api/frame")
    def frame(run: str, agent: str, seed: int, step: int = 0, version: str | None = None):
        """Inspect mode: one move of a replay game (fixed by its seed and model version), with the inputs the
        network saw, its output for each move, and the vision's drawing of what it looks at."""
        info = find(run)
        algo = next((a["algorithm"] for a in info["config"]["agents"] if a["id"] == agent), None)
        got = hub.watcher(run).policy(agent, version) if algo else None
        if got is None:
            raise HTTPException(404, f"no model for {agent} in {run}")
        act, meta = got
        if version and meta["version"] != version:
            raise HTTPException(410, "That model version is no longer loaded. Inspect a newer game.")
        if isinstance(getattr(act, "__self__", None), RecordedPolicy):
            if seed not in act.__self__.games:
                raise HTTPException(404, "jev only played its test games; inspect one of those")
            act = act.__self__.for_seed(seed).act
        g = games.game((run, agent, meta["version"], seed), act, meta, seed)
        return {**g.frame(step), "info": inspect_mode.describe(meta, algo)}

    return app


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--parallel", type=int, default=1, help="runs that may train at the same time")
    p.add_argument("--read-only", action="store_true", help="a shareable demo: browse, watch, inspect and compare, "
                   "but nothing can be launched, edited, deleted or analysed")
    return p.parse_args(argv)


def main():
    cfg = parse_args()
    import torch  # keep replay inference light: one thread
    torch.set_num_threads(1)
    print(f"[dashboard] http://{cfg.host}:{cfg.port}" + (" (read-only demo)" if cfg.read_only else ""), flush=True)
    uvicorn.run(create_app(cfg.parallel, read_only=cfg.read_only), host=cfg.host, port=cfg.port, log_level="warning",
                timeout_graceful_shutdown=2)


if __name__ == "__main__":
    main()
