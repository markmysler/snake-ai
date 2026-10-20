"""Evaluate saved models on fixed-seed games (greedy policy, same seeds for every model).

    python evaluate.py models/ga_best.npy
    python evaluate.py models/*.npy models/*.pt --games 100
"""
import argparse

import numpy as np

from policies import load_policy
import vision as visions
from snake_env import ENDS, SnakeEnv, end_code, end_counts

EVAL_SEED = 2_000_000  # disjoint from training and DQN-checkpoint seeds


def evaluate(path, games=100, seed=EVAL_SEED, width=None, height=None):
    act, meta = load_policy(path)
    width, height = width or meta["width"], height or meta["height"]
    if meta["vision"]["type"] != "short" and (width, height) != (meta["width"], meta["height"]):
        raise SystemExit(f"{path}: this vision only works on the grid it was trained on")
    env = SnakeEnv(width, height, vision=meta["vision"])
    scores, steps, ends = [], [], []
    for i in range(games):
        state, done = env.reset(seed=seed + i), False
        while not done:
            state, _, done, info = env.step(act(state))
        scores.append(info["score"])
        steps.append(info["steps"])
        ends.append(end_code(info))
    return {"model": path, "vision": visions.to_string(meta["vision"]), "games": games, "mean": float(np.mean(scores)),
            "median": float(np.median(scores)), "max": int(np.max(scores)),
            "mean_steps": float(np.mean(steps)), "median_steps": float(np.median(steps)),
            "steps_per_apple": float(np.sum(steps) / max(np.sum(scores), 1)),
            "outcomes": end_counts(ends),
            "per_game": {"score": [int(s) for s in scores], "steps": [int(s) for s in steps], "end": "".join(ends)}}


def format_table(rows, columns):
    """Plain-text table; columns = [(header, key, format)]."""
    cells = [[h for h, _, _ in columns]] + [[f.format(r[k]) for _, k, f in columns] for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(columns))]
    lines = ["  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(row, widths)))
             for row in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


EVAL_COLUMNS = [("model", "model", "{}"), ("vision", "vision", "{}"), ("games", "games", "{}"),
                ("mean", "mean", "{:.2f}"), ("median", "median", "{:.1f}"), ("max", "max", "{}"),
                ("mean steps", "mean_steps", "{:.0f}"), ("steps/apple", "steps_per_apple", "{:.0f}"),
                ("wall/body/starved/won", "ends", "{}")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("models", nargs="+", help=".npy (GA genome) or .pt (DQN checkpoint)")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=EVAL_SEED, help="game i uses seed + i")
    p.add_argument("--width", type=int, default=None, help="default: grid the model was trained on")
    p.add_argument("--height", type=int, default=None)
    args = p.parse_args()
    rows = [evaluate(m, args.games, args.seed, args.width, args.height) for m in args.models]
    for r in rows:
        r["ends"] = "/".join(str(r["outcomes"][k]) for k in ENDS.values())
    print(format_table(rows, EVAL_COLUMNS))


if __name__ == "__main__":
    main()
