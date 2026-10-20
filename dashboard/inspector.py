"""Inspect mode: one move of a replay game, with what the snake saw and what its network thought.

A replay game is fixed by its seed and the policy version, so any move can be rebuilt on the server:
the game is played once to record its moves, then a cursor steps an environment through them.
Stepping forward one move at a time (the usual case) costs one environment step.
"""
import threading
from collections import OrderedDict

import numpy as np

import vision as visions
from dashboard.replay import MAX_REPLAY_STEPS
from policies import scores_of
from snake_env import SnakeEnv

KEEP_GAMES = 16


class Game:
    """One recorded game and a cursor environment positioned at some move."""

    def __init__(self, act, meta, seed):
        self.act, self.meta, self.seed = act, meta, seed
        self.width, self.height, self.vision = meta.get("width", 10), meta.get("height", 10), meta.get("vision", "short")
        env = SnakeEnv(self.width, self.height, seed=seed, vision=self.vision)
        state, done, self.actions, info = env.reset(), False, [], None
        while not done and env.steps < MAX_REPLAY_STEPS:
            a = act(state)
            self.actions.append(a)
            state, _, done, info = env.step(a)
        self.end = None if info is None else ("wall" if info.get("crash") == "wall" else "body" if info["collision"]
                                               else "won" if info["won"] else "starved" if info["truncated"] else "limit")
        self.env, self.step = None, 0
        self.lock = threading.Lock()

    def seek(self, step):
        """The environment after `step` moves (0 = the start)."""
        if self.env is None or step < self.step:
            self.env = SnakeEnv(self.width, self.height, seed=self.seed, vision=self.vision)
            self.state, self.step = self.env.reset(), 0
        while self.step < step:
            self.state, _, done, _ = self.env.step(self.actions[self.step])
            self.step += 1
        return self.env

    def frame(self, step):
        step = max(0, min(int(step), len(self.actions)))
        with self.lock:
            env = self.seek(step)
            obs = np.asarray(self.state, dtype=np.float32)
            last = step == len(self.actions)
            out = None if last else [round(float(v), 4) for v in scores_of(self.act)(obs)]
            return {"step": step, "moves": len(self.actions), "end": self.end if last else None,
                    "snake": [list(c) for c in env.snake], "food": list(env.food) if env.food is not None else None,
                    "direction": env.direction, "score": env.score,
                    "action": None if last else self.actions[step], "out": out,
                    "inputs": [round(float(v), 3) for v in obs],
                    "explain": visions.explain(self.vision, env, obs)}


class Inspector:
    """Recorded games by (run, agent, version, seed), most recently used kept."""

    def __init__(self):
        self.games = OrderedDict()
        self.lock = threading.Lock()

    def game(self, key, act, meta, seed):
        with self.lock:
            g = self.games.get(key)
            if g is not None:
                self.games.move_to_end(key)
                return g
        g = Game(act, meta, seed)
        with self.lock:
            self.games[key] = g
            while len(self.games) > KEEP_GAMES:
                self.games.popitem(last=False)
        return g


def describe(meta, algorithm):
    """What the inspector panel needs once per game: input names or picture layout, and what the outputs mean."""
    w, h, v = meta.get("width", 10), meta.get("height", 10), meta.get("vision", "short")
    return {"vision": visions.to_string(v), "names": visions.input_names(v, w, h), "picture": visions.picture_info(v, w, h),
            "outputs": "q" if algorithm == "dqn" else "scores", "width": w, "height": h}
