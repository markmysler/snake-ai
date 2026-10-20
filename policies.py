"""Numpy MLP policy used by the GA, and load_policy() for any saved model file."""
import json
import os

import numpy as np

from snake_env import N_ACTIONS
import vision as visions

GA_HIDDEN = 16


def mlp_param_count(n_in, hidden=GA_HIDDEN, n_out=N_ACTIONS):
    return n_in * hidden + hidden + hidden * n_out + n_out


def mlp_inputs(n_params, hidden=GA_HIDDEN, n_out=N_ACTIONS):
    """Inverse of mlp_param_count: how many inputs a genome of this size was built for."""
    return (n_params - hidden - hidden * n_out - n_out) // hidden


class MLPPolicy:
    """state -> tanh(hidden) -> 3 action scores, weights unpacked from one flat vector."""

    def __init__(self, flat, hidden=GA_HIDDEN, n_out=N_ACTIONS):
        flat = np.asarray(flat, dtype=np.float32)
        n_in = mlp_inputs(flat.size, hidden, n_out)
        assert flat.size == mlp_param_count(n_in, hidden, n_out), "genome has wrong size"
        i = 0
        self.w1 = flat[i:i + n_in * hidden].reshape(n_in, hidden); i += n_in * hidden
        self.b1 = flat[i:i + hidden]; i += hidden
        self.w2 = flat[i:i + hidden * n_out].reshape(hidden, n_out); i += hidden * n_out
        self.b2 = flat[i:i + n_out]
        self.n_in = n_in

    def scores(self, state):
        """The 3 output values (left, straight, right); the highest one is the move."""
        return np.tanh(state @ self.w1 + self.b1) @ self.w2 + self.b2

    def act(self, state):
        return int(np.argmax(self.scores(state)))


def scores_of(act):
    """The output values behind a policy's act function (MLPPolicy.act, TorchPolicy.act or a replay's act)."""
    return act.__self__.scores


class RecordedPolicy:
    """Games a remote model (jev) played, kept with every move and how sure it was of it. It can't play new games:
    the dashboard replays these, by seed (see for_seed)."""

    def __init__(self, data):
        self.data = data
        self.games = {int(g["seed"]): g for g in data.get("games", [])}

    def seeds(self):
        return sorted(self.games)

    def for_seed(self, seed):
        return Replay(self.games[int(seed)])

    def act(self, state):
        raise RuntimeError("a recorded player can only replay its own games (use for_seed)")


class Replay:
    """One recorded game, move by move: act() returns the next recorded move; scores() its log-probabilities."""

    def __init__(self, game):
        self.game, self.i = game, 0

    def scores(self, state, step=None):
        p = self.game["probs"][self.i if step is None else step]
        return np.log(np.maximum(np.asarray(p, dtype=np.float64), 1e-6))

    def act(self, state):
        a = self.game["actions"][min(self.i, len(self.game["actions"]) - 1)]
        self.i += 1
        return int(a)


def load_policy(path, meta=None):
    """Load a saved model. Returns (act_fn, meta); meta always has vision, width, height.
    .npy = GA genome (+ .json sidecar, or pass `meta`), .pt = DQN checkpoint, .json = recorded jev games
    (act is then RecordedPolicy.act, which replays them by seed: see RecordedPolicy)."""
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        policy = RecordedPolicy(data)
        meta = {**(meta or {}), **{k: v for k, v in data.items() if k != "games"}, "recorded": True}
        meta["vision"] = visions.parse(meta.get("vision", "short"))
        meta.setdefault("width", 10)
        meta.setdefault("height", 10)
        return policy.act, meta
    if path.endswith(".npy"):
        meta_path = os.path.splitext(path)[0] + ".json"
        meta = dict(meta or {})
        if not meta and os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        policy = MLPPolicy(np.load(path), hidden=meta.get("hidden", GA_HIDDEN))
        act = policy.act
    elif path.endswith(".pt"):
        from dqn_model import TorchPolicy, load_qnet  # torch only when needed
        net, saved = load_qnet(path)
        meta = {**(meta or {}), **saved}
        act = TorchPolicy(net).act
    else:
        raise ValueError(f"unknown model type: {path}")
    meta["vision"] = visions.parse(meta.get("vision", "short"))  # older models: no vision saved
    meta.setdefault("width", 10)
    meta.setdefault("height", 10)
    expected = visions.state_size(meta["vision"], meta["width"], meta["height"])
    n_in = policy.n_in if path.endswith(".npy") else net.n_in
    if n_in != expected:
        raise ValueError(f"{path}: model takes {n_in} inputs but vision "
                         f"{visions.to_string(meta['vision'])} gives {expected}")
    return act, meta
