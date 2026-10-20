"""jev player: TypeSafe's decision model plays the fixed test games, many boards per request (see jev.py).

    python train_jev.py --vision short --games 20 --prompt careful

jev doesn't learn, so there is no training phase: the run *is* the test. It plays test games 1..N, the same
fixed-seed games every other model is tested on (evaluate.py), all at once: each request asks for the next move
on every unfinished board. Its moves and how sure it was of each are saved in models/<agent>_best.json, so the
dashboard replays and inspects these games without calling jev again.

The learning curve is the mean score of the games finished so far. --max-cost stops the run before it spends
more than that (USD); a time budget stops it too. Games still running then are left out of the result.
"""
import argparse
import json
import math
import os
import time

import numpy as np

import jev
import rewards
import vision as visions
from evaluate import EVAL_SEED
from publisher import Publisher, write_json
from snake_env import SnakeEnv, end_code, end_counts


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--agent", default="jev", help="name used for the files this player writes")
    p.add_argument("--vision", default="short", help="what the snake senses, e.g. rays:depth=4")
    p.add_argument("--reward", default="score", help="recorded only: jev is not trained on a reward")
    p.add_argument("--games", type=int, default=20, help="test games to play (the usual test is 100)")
    p.add_argument("--batch", type=int, default=100, help="boards per request")
    p.add_argument("--prompt", choices=tuple(jev.PROMPTS), default="rules", help="the instructions jev gets")
    p.add_argument("--max-cost", type=float, default=0.5, help="stop before spending more than this (USD)")
    p.add_argument("--threads", type=int, default=1, help="(accepted for the planner; requests are sequential)")
    p.add_argument("--time-budget", type=float, default=0, help="minutes, 0 = none")
    p.add_argument("--width", type=int, default=10)
    p.add_argument("--height", type=int, default=10)
    p.add_argument("--seed", type=int, default=0, help="(unused: the test games have fixed seeds)")
    p.add_argument("--live-dir", default="live")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--models-dir", default="models")
    return p.parse_args()


def evaluation(done):
    """The same summary evaluate.py makes, from the finished games."""
    scores = [g["score"] for g in done]
    steps = [g["steps"] for g in done]
    ends = "".join(g["end"] for g in done)
    return {"games": len(done), "mean": float(np.mean(scores)), "median": float(np.median(scores)),
            "max": int(np.max(scores)), "mean_steps": float(np.mean(steps)), "median_steps": float(np.median(steps)),
            "steps_per_apple": float(np.sum(steps) / max(np.sum(scores), 1)), "outcomes": end_counts(ends),
            "per_game": {"score": scores, "steps": steps, "end": ends}}


def main():
    args = parse_args()
    vision, reward = visions.parse(args.vision), rewards.parse(args.reward)
    pub = Publisher(args.agent, "move", args.live_dir, args.results_dir)
    os.makedirs(args.models_dir, exist_ok=True)
    s = jev.settings()
    print(f"[{args.agent}] {s['model']} · vision={visions.to_string(vision)} · prompt={args.prompt} · "
          f"{args.games} games · {args.batch} boards per request · cap ${args.max_cost:g}", flush=True)

    games = []
    for i in range(args.games):
        env = SnakeEnv(args.width, args.height, vision=vision)
        games.append({"seed": EVAL_SEED + i, "env": env, "state": env.reset(seed=EVAL_SEED + i),
                      "actions": [], "probs": [], "done": False})
    done, cost, moves, best, stopped = [], 0.0, 0, 0, None
    meta = {"kind": "jev", "model": s["model"], "prompt": args.prompt, "vision": vision, "width": args.width,
            "height": args.height, "reward": reward}

    def save():
        recorded = [{k: g[k] for k in ("seed", "actions", "probs", "score", "steps", "end")} for g in done]
        out = {**meta, "games": recorded, "evaluation": evaluation(done) if done else None, "cost_usd": round(cost, 5),
               "unfinished": sum(not g["done"] for g in games), "stopped": stopped}
        write_json(os.path.join(args.models_dir, f"{args.agent}_best.json"), out)
        info = {k: v for k, v in out.items() if k != "games"}
        pub.weights(lambda f: f.write(json.dumps(out).encode()), "json", moves, **info)

    try:
        while True:
            active = [g for g in games if not g["done"]]
            if not active:
                break
            if cost + len(active) * jev.COST_PER_DECISION * 3 > args.max_cost:
                stopped = f"spending cap ${args.max_cost:g} reached"
                break
            if args.time_budget and pub.elapsed() >= args.time_budget * 60:
                stopped = "time budget reached"
                break
            if pub.stop_requested():
                stopped = "stopped"
                break
            for start in range(0, len(active), max(1, args.batch)):
                chunk = active[start:start + args.batch]
                answers, c = jev.decide({str(g["seed"]): jev.describe(vision, g["env"], g["state"]) for g in chunk},
                                        prompt=args.prompt)
                cost += c
                for g in chunk:
                    action, probs = answers[str(g["seed"])]
                    g["actions"].append(action)
                    g["probs"].append([round(p, 3) for p in probs])
                    g["state"], _, finished, info = g["env"].step(action)
                    moves += 1
                    if finished:
                        g.update(done=True, score=info["score"], steps=info["steps"], end=end_code(info))
                        done.append(g)
                        best = max(best, info["score"])
            newly = [g for g in done if not g.get("saved")]
            if newly:
                for g in newly:
                    g["saved"] = True
                save()
            mean = np.mean([g["score"] for g in done]) if done else 0.0
            pub.stats(moves, moves, len(done), best, mean, ends=end_counts(g["end"] for g in done),
                      cost_usd=round(cost, 4), unfinished=len(active),
                      playing=[g["env"].score for g in games if not g["done"]])  # apples so far in unfinished games
            if len(done) and math.isfinite(mean):
                print(f"[{args.agent}] {moves:6d} moves | {len(done):3d}/{len(games)} games | mean {mean:5.2f} | "
                      f"best {best:3d} | ${cost:.4f} | {pub.elapsed():6.1f}s", flush=True)
    except jev.JevError as e:
        stopped = str(e)
        print(f"[{args.agent}] {e}", flush=True)
    except KeyboardInterrupt:
        stopped = "interrupted"
    finally:
        if done:
            save()
        pub.finish()
    print(f"[{args.agent}] done: {len(done)} of {len(games)} games, ${cost:.4f}" + (f" ({stopped})" if stopped else ""),
          flush=True)


if __name__ == "__main__":
    main()
