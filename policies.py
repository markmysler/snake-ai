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

    def act(self, state):
        return int(np.argmax(np.tanh(state @ self.w1 + self.b1) @ self.w2 + self.b2))


def load_policy(path, meta=None):
    """Load a saved model. Returns (act_fn, meta); meta always has vision, width, height.
    .npy = GA genome (+ .json sidecar, or pass `meta`), .pt = DQN checkpoint."""
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
