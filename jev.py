"""jev: TypeSafe's decision model as a Snake player, through TokenRouter's decisions API. Standard library only.

jev doesn't learn from games: it is given a description of what the snake senses and picks a move. So a jev run
has no training phase; it plays the fixed test games (see train_jev.py) and its "training" is the prompt, chosen
with the Instructions setting (PROMPTS below). Every move is one decision; many boards go in one request.

    TOKENROUTER_API_KEY=...       in .env (see llm.py); jev is only on TokenRouter
    SNAKE_AI_JEV_URL=...          optional, the decisions endpoint (default below)
    SNAKE_AI_JEV_MODEL=...        optional, default typesafe/jev-1.13
"""
import json
import math
import os
import time
import urllib.error
import urllib.request

import llm
import vision as visions

URL = "https://api.tokenrouter.com/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
MOVES = ("left", "straight", "right")  # = actions 0, 1, 2
CRITERIA = {"left": "turn left (relative to the direction the snake is heading)",
            "straight": "keep going straight ahead",
            "right": "turn right (relative to the direction the snake is heading)"}
RULES = ("You steer a snake on a square grid. Each turn you choose one move, relative to the direction the snake "
         "is heading: turn left, go straight or turn right. Eating an apple scores a point and makes the snake longer. "
         "The game ends if the head moves into a wall or into the snake's own body, or if the snake goes too long "
         "without eating.")
PROMPTS = {
    "rules": {"label": "Just the rules", "text": RULES},
    "careful": {"label": "Rules + stay safe",
                "text": RULES + " Never choose a move into a deadly square. When several moves are safe, prefer the one "
                        "that keeps the most room to move, so the snake doesn't trap itself; go for the apple when that "
                        "is safe."},
}
COST_PER_DECISION = 5e-6     # USD, measured with 100 boards per request (one board alone costs about 1.5e-5)
MOVES_PER_GAME = 250         # a rough average, for estimates before a run


class JevError(RuntimeError):
    pass


def settings():
    llm.load_env()
    return {"url": os.environ.get("SNAKE_AI_JEV_URL") or URL, "model": os.environ.get("SNAKE_AI_JEV_MODEL") or MODEL,
            "api_key": os.environ.get("TOKENROUTER_API_KEY", "")}


def configured():
    return bool(settings()["api_key"])


def estimate(games, batch=100):
    """Rough cost (USD) and time (minutes) of playing `games` games, before a run. The time assumes jev answers in
    about 1.6 s per request; its API is in alpha and is sometimes ten times slower."""
    decisions = games * MOVES_PER_GAME
    per = COST_PER_DECISION if batch >= 20 else 1.5e-5
    calls = MOVES_PER_GAME * math.ceil(games / max(1, batch))
    return {"usd": round(decisions * per, 3), "minutes": round(calls * 1.6 / 60, 1), "decisions": decisions}


def _apple_words(ahead, behind, left, right):
    parts = [w for w, on in (("ahead", ahead), ("behind", behind)) if on]
    parts += [f"to the {w}" for w, on in (("left", left), ("right", right)) if on]
    return "The apple is " + (" and ".join(parts) if parts else "nowhere to be seen") + "."


def _short_apple(env, obs):
    """short/room give the apple's direction on the screen (left, right, up, down): turn it around the heading."""
    left, right, up, down = (float(v) for v in obs[7:11])
    vx, vy = right - left, down - up
    (ax, ay), (rx, ry) = visions.DIRS[env.direction], visions.DIRS[(env.direction + 1) % 4]
    f, r = vx * ax + vy * ay, vx * rx + vy * ry
    return _apple_words(f > 0, f < 0, r < 0, r > 0)


def describe(spec, env, obs):
    """What the snake senses, in words jev can read. The same information as the vision's inputs, said relative to
    the heading (as the moves are); the picture visions become small character grids."""
    kind = visions.parse(spec)["type"]
    moves = ("straight", "left", "right")
    if kind in ("short", "room"):
        lines = ["Next square: " + ", ".join(f"{m} is {'DEADLY' if obs[i] else 'safe'}" for i, m in enumerate(moves)) + ".",
                 _short_apple(env, obs)]
        if kind == "room":
            lines.append("Free squares the snake could still reach after each move: "
                         + ", ".join(f"{m} {obs[11 + i]:.0%}" for i, m in enumerate(moves)) + ".")
        return "\n".join(lines)
    if kind == "rays":
        depth = visions.parse(spec)["depth"]
        seen = []
        for i, d in enumerate(visions.RAY_NAMES):
            things = [f"{what} {round(depth + 1 - obs[3 * i + j] * depth)} square(s) away"
                      for j, what in enumerate(("wall", "own body", "apple")) if obs[3 * i + j] > 0]
            seen.append(f"{d}: " + (", ".join(things) if things else f"nothing within {depth} squares"))
        return ("Looking along 8 lines from the head (ahead = the straight move's side):\n" + "\n".join(seen)
                + "\n" + _apple_words(*(bool(v) for v in obs[24:28])))
    names = visions.input_names(spec, env.width, env.height)
    if names:  # a vision added later: its inputs by name
        about = visions.VISIONS[kind].description
        return (f"(What the numbers mean: {about})\n"
                + "; ".join(f"{n}: {v:g}" for n, v in zip(names, (round(float(x), 2) for x in obs))))
    pic = visions.picture_info(spec, env.width, env.height)
    side, n = pic["side"], pic["side"] ** 2
    blocked, body, apple = (obs[i * n:(i + 1) * n] for i in range(3))
    rows = []
    for y in range(side):
        row = ""
        for x in range(side):
            i = y * side + x
            row += ("H" if (x, y) == (side // 2, side // 2) else "A" if apple[i] else "#" if blocked[i]
                    else "o" if body[i] else ".")
        rows.append(row)
    text = ("The snake's view, centred on its head (H) and turned so it is heading up: # = wall or body that blocks, "
            "o = body that will have moved away, A = apple, . = free.\n" + "\n".join(rows))
    extra = obs[3 * n:]
    if len(extra):
        text += "\n" + "; ".join(f"{name}: {'yes' if v else 'no'}" for name, v in zip(pic["extra"], extra))
    return text


def decide(boards, prompt="rules", timeouts=(15, 30, 60, 120)):
    """boards: {id: description}. Returns ({id: (move index, [p_left, p_straight, p_right])}, cost in USD).
    The decisions API is in alpha and its answer times have a long tail (seconds, sometimes a minute), so a request
    that hangs is sent again with a longer timeout rather than waited out."""
    s = settings()
    if not s["api_key"]:
        raise JevError("jev needs TOKENROUTER_API_KEY in .env")
    questions = {bid: {"type": "choice", "criteria": CRITERIA,
                       "instructions": f"{PROMPTS[prompt]['text']}\n\nWhat this snake senses now:\n{text}\n\nChoose its move."}
                 for bid, text in boards.items()}
    body = json.dumps({"model": s["model"], "state": {"game": "snake"}, "questions": questions}).encode()
    req = urllib.request.Request(s["url"], data=body, headers={"Authorization": f"Bearer {s['api_key']}",
                                                               "Content-Type": "application/json"})
    retries = len(timeouts) - 1
    for attempt, timeout in enumerate(timeouts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                data = json.loads(res.read())
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(3 * (attempt + 1))
                continue
            raise JevError(f"jev request failed ({e.code}): {detail}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < retries:
                if "timed out" not in str(e):  # a network error: give it a moment; a timeout: just ask again
                    time.sleep(3 * (attempt + 1))
                continue
            raise JevError(f"Could not reach jev: {e}") from None
    out = {}
    for bid in boards:
        a = (data.get("answers") or {}).get(bid) or {}
        probs = a.get("probabilities") or {}
        p = [float(probs.get(m, 0.0)) for m in MOVES]
        choice = a.get("choice")
        out[bid] = (MOVES.index(choice) if choice in MOVES else max(range(3), key=lambda i: p[i]), p)
    return out, float((data.get("usage") or {}).get("cost") or 0.0)
