"""PyTorch Q-network shared by the DQN trainer, the dashboard and evaluate.py."""
import numpy as np
import torch
from torch import nn

from snake_env import N_ACTIONS

DQN_HIDDEN = (128, 128)


class QNet(nn.Module):
    def __init__(self, n_in, hidden=DQN_HIDDEN, n_out=N_ACTIONS):
        super().__init__()
        self.n_in, self.hidden = n_in, tuple(hidden)
        layers, prev = [], n_in
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def save_checkpoint(f, net, **meta):
    """Save weights + metadata. `f` is a path or a binary file object."""
    state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    torch.save({"state_dict": state, "hidden": list(net.hidden), "n_in": net.n_in, **meta}, f)


def load_qnet(f):
    """Load a checkpoint saved by save_checkpoint. Returns (net in eval mode, meta)."""
    ckpt = torch.load(f, map_location="cpu", weights_only=True)
    state = ckpt.pop("state_dict")
    n_in = ckpt.pop("n_in", None) or state["net.0.weight"].shape[1]  # older checkpoints
    net = QNet(n_in, hidden=ckpt.pop("hidden"))
    net.load_state_dict(state)
    net.eval()
    return net, ckpt


class TorchPolicy:
    """Greedy policy around a CPU copy of a QNet."""

    def __init__(self, net):
        self.net = net

    @torch.no_grad()
    def act(self, state):
        q = self.net(torch.from_numpy(np.asarray(state, dtype=np.float32)))
        return int(q.argmax())
