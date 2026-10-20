"""Play one viewing-only game and encode it compactly for the browser.

Cells are flat indices (y * width + x). The browser rebuilds every frame from:
  init   snake cells at the start, head first
  heads  head cell after each move
  food   initial food cell; eats = [[frame, new food cell or -1], ...]
"""
from snake_env import SnakeEnv

MAX_REPLAY_STEPS = 20_000


def play_game(act, width=10, height=10, seed=None, vision="short"):
    env = SnakeEnv(width, height, vision=vision)
    state = env.reset(seed=seed)  # the same way evaluate.py starts a test game

    def cell(c):
        return c[1] * width + c[0]

    init, food = [cell(c) for c in env.snake], cell(env.food)
    heads, eats = [], []
    info = {"score": 0, "collision": False, "truncated": False, "won": False}
    done = False
    while not done and env.steps < MAX_REPLAY_STEPS:
        state, _, done, info = env.step(act(state))
        if info["collision"]:
            break
        heads.append(cell(env.snake[0]))
        if info["score"] > len(eats):
            eats.append([len(heads), cell(env.food) if env.food is not None else -1])
    end = ("crash" if info["collision"] else "stall" if info["truncated"]
           else "win" if info["won"] else "limit")
    return {"w": width, "h": height, "init": init, "food": food, "heads": heads, "seed": seed,
            "eats": eats, "score": info["score"], "steps": env.steps, "end": end}
