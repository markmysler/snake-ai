"""Screenshots for the README: every view in light and dark, plus a GIF of the snakes playing.

    pip install -r requirements-dev.txt && python -m playwright install chromium
    python tools/screenshots.py                       # writes docs/img/*.webp and docs/img/boards.gif
    python tools/screenshots.py --channel msedge      # use an installed Edge/Chrome instead of Playwright's Chromium
    python tools/screenshots.py --watch <run> --inspect <run> --compare <run>,<run>

Starts the dashboard on a free port with nothing training (--parallel 0), reads the lessons in place and
changes nothing. Without arguments it picks the runs itself: the course's lesson 06 against lesson 01 once
the course is trained, otherwise the best runs in the notebook. Re-run it whenever the views change.
"""
import argparse
import io
import os
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import lessons  # noqa: E402

OUT = os.path.join(ROOT, "docs", "img")
VIEWPORT = {"width": 1280, "height": 800}
HIDE = "#machine { visibility: hidden; } #machine-card { display: none; }"  # this machine's cores and RAM: not part of the pictures
PLAYING = r"[...document.querySelectorAll('#games .board .line')].every(e => /^score ([2-9]|\d\d)/.test(e.textContent))"
DONE = ("finished", "stopped")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(port):
    proc = subprocess.Popen([sys.executable, "-m", "dashboard.server", "--port", str(port), "--parallel", "0"], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs", timeout=1)
            return proc
        except OSError:
            time.sleep(0.3)
    proc.kill()
    raise SystemExit("the dashboard did not start")


def pick_runs():
    """(watch, inspect, compare ids): course lesson 06 vs 01 when trained, else the notebook's best runs."""
    runs = [r for r in lessons.list_runs() if r["status"] in DONE and r.get("evaluation")]
    if not runs:
        raise SystemExit("no finished runs to show")
    best = lambda rs: max(rs, key=lambda r: max(v["mean"] for v in r["evaluation"].values()))  # noqa: E731

    def first_of(lesson, recipe):
        rs = [r for r in runs if r["lesson"] == lesson and (r.get("course") or {}).get("recipe") == recipe
              and not r["course"].get("trial")]
        return min(rs, key=lambda r: (r.get("group") or {}).get("index", 0)) if rs else None

    room, base = first_of("c06-room-to-move", "room"), first_of("c01-two-ways-to-learn", "baseline")
    if room and base:
        rays = first_of("c04-seeing-further", "rays 4") or room
        return room["id"], rays["id"], [base["id"], room["id"]]
    two_agents = [r for r in runs if len(r["evaluation"]) >= 2]
    rays = [r for r in runs if r["config"]["vision"]["type"] == "rays"]
    pairs = [lesson for lesson in {r["lesson"] for r in two_agents} if sum(r["lesson"] == lesson for r in two_agents) >= 2]
    pair_lesson = best([r for r in two_agents if r["lesson"] in pairs])["lesson"] if pairs else None
    pair = [r["id"] for r in two_agents if r["lesson"] == pair_lesson][:2] if pair_lesson else [r["id"] for r in runs[-2:]]
    return best(two_agents or runs)["id"], best(rays or runs)["id"], pair


def save_webp(png, name, max_height=None):
    from PIL import Image
    img = Image.open(io.BytesIO(png)).convert("RGB")
    if max_height and img.height > max_height:
        img = img.crop((0, 0, img.width, max_height))
    path = os.path.join(OUT, f"{name}.webp")
    img.save(path, "WEBP", quality=80, method=6)
    return path


def shoot(page, base, theme, ids):
    watch, inspect, compare = ids
    shots = []

    def go(hash_, selector):
        page.goto(f"{base}/#{hash_}")
        page.wait_for_selector(selector, timeout=30000)
        page.add_style_tag(content=HIDE)

    # Runs: the course
    go("runs", ".list-head")
    page.locator(".list-head").first.scroll_into_view_if_needed()
    page.evaluate("window.scrollBy(0, -12)")
    time.sleep(0.8)
    shots.append(save_webp(page.screenshot(), f"runs-{theme}"))

    # Watch: headline numbers and the snakes playing (the AI analysis card has its own picture)
    go(f"watch/{watch}", "#games .board .line")
    if page.locator("#analysis").count():
        page.locator("#analysis").screenshot()  # (scrolls it into view once so it renders)
        shots.append(save_webp(page.locator("#analysis").screenshot(), f"analysis-{theme}", max_height=900))
        page.evaluate("document.querySelector('#analysis').style.display = 'none'")
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_function(PLAYING, timeout=60000)  # a few apples into every game
    games = page.locator("#games").bounding_box()
    shots.append(save_webp(page.screenshot(full_page=True, clip={"x": 0, "y": 0, "width": VIEWPORT["width"],
                                                                 "height": games["y"] + games["height"] + 16}), f"watch-{theme}"))
    page.locator("#progress").scroll_into_view_if_needed()
    time.sleep(0.8)
    shots.append(save_webp(page.locator("#progress").screenshot(), f"learning-{theme}"))

    # Inspect: one move of a game, what the snake sees and what its network thinks
    go(f"watch/{inspect}", "#games .board .line")
    page.wait_for_function("document.querySelector('#games .board .line').textContent.startsWith('score')", timeout=30000)
    page.locator("#games .board").first.get_by_role("button", name="Inspect").click()
    page.wait_for_selector(".inspector .prefs")
    for _ in range(25):  # step a little into the game
        page.keyboard.press("ArrowRight")
        time.sleep(0.06)
    time.sleep(1)
    shots.append(save_webp(page.locator(".inspector").screenshot(), f"inspect-{theme}", max_height=760))

    # Compare: what changed, the curves, the test results and the verdict
    go(f"compare/{','.join(compare)}", "#curves canvas")
    time.sleep(2.5)
    shots.append(save_webp(page.screenshot(full_page=True), f"compare-{theme}", max_height=1700))

    for view, selector in (("best", ".view h1"), ("learn", ".gl-sections"), ("ideas", ".idea-list, .card.analysis")):
        go(view, selector)
        time.sleep(1.2)
        shots.append(save_webp(page.screenshot(), f"{view}-{theme}"))
    return shots


def record_gif(page, base, watch, seconds=7.0, fps=10, width=720):
    """The Watch view's boards playing, as a looping GIF."""
    from PIL import Image
    page.goto(f"{base}/#watch/{watch}")
    page.wait_for_selector("#games .board .line")
    page.get_by_role("button", name="2×").click()
    page.locator("#games .boards").scroll_into_view_if_needed()
    page.wait_for_function(PLAYING, timeout=60000)
    boxes = [b.bounding_box() for b in page.locator("#games .board").all()]
    clip = {"x": min(b["x"] for b in boxes) - 4, "y": min(b["y"] for b in boxes) - 4}
    clip.update(width=max(b["x"] + b["width"] for b in boxes) + 4 - clip["x"],
                height=max(b["y"] + b["height"] for b in boxes) + 4 - clip["y"])
    frames, end = [], time.time() + seconds
    while time.time() < end:
        t = time.time()
        img = Image.open(io.BytesIO(page.screenshot(clip=clip))).convert("RGB")
        frames.append(img.resize((width, round(img.height * width / img.width)), Image.LANCZOS))
        time.sleep(max(0, 1 / fps - (time.time() - t)))
    palette = frames[len(frames) // 2].quantize(colors=64, method=Image.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    path = os.path.join(OUT, "boards.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0, optimize=True)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--channel", help="browser channel, e.g. msedge or chrome (default: Playwright's Chromium)")
    p.add_argument("--watch")
    p.add_argument("--inspect")
    p.add_argument("--compare", help="two or three run ids, comma separated")
    p.add_argument("--no-gif", action="store_true")
    a = p.parse_args()
    from playwright.sync_api import sync_playwright

    watch, inspect, compare = pick_runs()
    ids = (a.watch or watch, a.inspect or inspect, a.compare.split(",") if a.compare else compare)
    print(f"watch {ids[0]} · inspect {ids[1]} · compare {', '.join(ids[2])}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    port = free_port()
    server = start_server(port)
    base = f"http://127.0.0.1:{port}"
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel=a.channel) if a.channel else pw.chromium.launch()
            written = []
            for theme in ("light", "dark"):
                page = browser.new_page(viewport=VIEWPORT, color_scheme=theme)
                written += shoot(page, base, theme, ids)
                page.close()
            if not a.no_gif:
                page = browser.new_page(viewport=VIEWPORT, color_scheme="light")
                written.append(record_gif(page, base, ids[0]))
            browser.close()
    finally:
        server.terminate()
    total = sum(os.path.getsize(f) for f in written)
    for f in written:
        print(f"{os.path.relpath(f, ROOT):32s} {os.path.getsize(f) / 1024:7.0f} KB")
    print(f"{len(written)} files, {total / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
