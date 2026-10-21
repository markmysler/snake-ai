"""Export the read-only demo as a static site: the dashboard plus its server's answers saved as files.

    python tools/export_site.py                 # writes site/ (open it with: python -m http.server -d site)
    python tools/export_site.py --games 5       # more replay games per snake (default 3)

No server is needed afterwards, so the folder can be hosted anywhere that serves files: GitHub Pages
(.github/workflows/pages.yml builds and deploys it on every push), Vercel, Netlify, any web server.
Every GET answer is asked from the real read-only server, in this process, so the site shows exactly what
`python -m dashboard.server --read-only` would. What a server computes on demand is computed here instead:
each snake gets a few replay games (fixed seeds, the same every export), and each of those games is recorded
move by move for Inspect (gzip-compressed: the browser unpacks it). The page knows it is static from
`window.SNAKE_SITE` (set in the exported index.html) and reads these files instead of calling /api/
(see siteGet in dashboard/static/js/util.js).
"""
import argparse
import gzip
import json
import random
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from dashboard import inspector as inspect_mode  # noqa: E402
from dashboard.replay import play_game  # noqa: E402
from dashboard.server import STATIC, create_app  # noqa: E402
from policies import RecordedPolicy  # noqa: E402

SAFE = re.compile(r"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*$")  # ids become paths: nothing that could escape the folder


def write(out, rel, data, compress=False):
    if not SAFE.match(rel):
        raise ValueError(f"unsafe path {rel!r}")
    path = out / "data" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(data, separators=(",", ":"), allow_nan=False).encode()
    path.write_bytes(gzip.compress(raw, mtime=0) if compress else raw)
    return len(raw)


def seeds_for(run_id, agent, n, act):
    """The replay games' seeds: jev only played its test games, so some of those; else fixed per snake."""
    recorded = getattr(act, "__self__", None)
    if isinstance(recorded, RecordedPolicy):
        return recorded.seeds()[:n]
    rng = random.Random(f"{run_id}/{agent}")
    return [rng.randrange(2**31) for _ in range(n)]


def player(act, seed):
    """What plays the game with this seed: the model itself, or a fresh replay of the game jev played."""
    recorded = getattr(act, "__self__", None)
    return recorded.for_seed(seed).act if isinstance(recorded, RecordedPolicy) else act


def export(out, n_games=3, log=print):
    app = create_app(read_only=True)
    hub = app.state.hub
    client = TestClient(app)  # without `with`: the hub's poller never starts

    def get(path, **params):
        r = client.get(path, params=params)
        r.raise_for_status()
        return r.json()

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(STATIC, out / "static")
    html = (STATIC / "index.html").read_text(encoding="utf-8").replace('"/static/', '"static/')
    html = html.replace("</head>", "  <script>window.SNAKE_SITE = true;</script>\n</head>", 1)
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / ".nojekyll").write_text("")  # GitHub Pages: serve the files as they are

    write(out, "catalog.json", get("/api/catalog"))
    runs = get("/api/runs")
    write(out, "runs.json", runs)
    no_machine = {"runs": runs["runs"], "queue": runs["queue"], "system": None}  # this machine is not shown
    write(out, "events.json", {**no_machine, "watch": None})
    ideas = get("/api/ideas")
    write(out, "ideas.json", ideas)
    for review in ideas["reviews"]:
        write(out, f"ideas/{review['name']}.json", get("/api/ideas", review=review["name"]))
    for lesson in get("/api/catalog")["lessons"]:
        write(out, f"lesson/{lesson['name']}.json", get("/api/lesson", name=lesson["name"]))

    total, frames = 0, 0
    for r in runs["runs"]:
        rid = r["id"]
        write(out, f"watch/{rid}.json", {**no_machine, "watch": hub.watch_payload(rid)})
        write(out, f"history/{rid}.json", get("/api/history", id=rid))
        write(out, f"analysis/{rid}.json", get("/api/analysis", id=rid))
        for a in r["config"]["agents"]:
            got = hub.watcher(rid).policy(a["id"], None)
            if got is None:
                continue  # no model saved (e.g. stopped before its first one)
            act, meta = got
            games = []
            for seed in seeds_for(rid, a["id"], n_games, act):
                g = play_game(player(act, seed), meta.get("width", 10), meta.get("height", 10), seed,
                              meta.get("vision", "short"))
                games.append({"game": g, "meta": meta})
                rec = inspect_mode.Game(player(act, seed), meta, seed)
                total += write(out, f"frame/{rid}/{a['id']}/{seed}.json.gz", {
                    "info": inspect_mode.describe(meta, a["algorithm"]),
                    "frames": [rec.frame(s) for s in range(len(rec.actions) + 1)]}, compress=True)
                frames += 1
            write(out, f"game/{rid}/{a['id']}.json", games)
        log(f"[site] {rid}")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    log(f"[site] {len(runs['runs'])} runs, {frames} inspectable games ({total / 1e6:.0f} MB unpacked); "
        f"{size / 1e6:.1f} MB in {out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "site"), help="folder to write (replaced)")
    p.add_argument("--games", type=int, default=3, help="replay games per snake (each one inspectable)")
    args = p.parse_args(argv)
    import torch  # replays are small: one thread is fastest
    torch.set_num_threads(1)
    export(Path(args.out), args.games)


if __name__ == "__main__":
    main()
