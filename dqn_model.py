"""PyTorch Q-network shared by the DQN trainer, the dashboard and evaluate.py."""
import numpy as np
import torch
from torch import nn

from snake_env import N_ACTIONS

DQN_HIDDEN = (128, 128)


class QNet(nn.Module):
    """state -> hidden layers -> one Q-value per move. With `picture` (vision.picture_info: channels, side), the
    first channels x side x side inputs are read as images by a small CNN first; the rest (e.g. the window's apple
    flags) join its output before the hidden layers."""

    def __init__(self, n_in, hidden=DQN_HIDDEN, n_out=N_ACTIONS, picture=None):
        super().__init__()
        self.n_in, self.hidden, self.picture = n_in, tuple(hidden), picture
        prev = n_in
        if picture:
            c, s = len(picture["channels"]), picture["side"]
            self.shape, self.pic_size = (c, s, s), c * s * s
            stride = 2 if s >= 9 else 1  # big pictures (the whole board) are halved twice
            self.conv = nn.Sequential(nn.Conv2d(c, 16, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(16, 32, 3, stride=stride, padding=1), nn.ReLU(),
                                      nn.Conv2d(32, 32, 3, stride=stride, padding=1), nn.ReLU(), nn.Flatten())
            with torch.no_grad():
                prev = self.conv(torch.zeros(1, *self.shape)).shape[1] + n_in - self.pic_size
        layers = []
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        if self.picture:
            x = torch.cat([self.conv(x[:, :self.pic_size].reshape(-1, *self.shape)), x[:, self.pic_size:]], 1)
        out = self.net(x)
        return out.squeeze(0) if single else out


def save_checkpoint(f, net, **meta):
    """Save weights + metadata. `f` is a path or a binary file object."""
    state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    torch.save({"state_dict": state, "hidden": list(net.hidden), "n_in": net.n_in, "picture": net.picture, **meta}, f)


def load_qnet(f):
    """Load a checkpoint saved by save_checkpoint. Returns (net in eval mode, meta)."""
    ckpt = torch.load(f, map_location="cpu", weights_only=True)
    state = ckpt.pop("state_dict")
    n_in = ckpt.pop("n_in", None) or state["net.0.weight"].shape[1]  # older checkpoints
    net = QNet(n_in, hidden=ckpt.pop("hidden"), picture=ckpt.pop("picture", None))
    net.load_state_dict(state)
    net.eval()
    return net, ckpt


class TorchPolicy:
    """Greedy policy around a CPU copy of a QNet."""

    def __init__(self, net):
        self.net = net

    @torch.no_grad()
    def scores(self, state):
        """Q-values for left, straight, right: the expected future reward of each move."""
        return self.net(torch.from_numpy(np.asarray(state, dtype=np.float32))).numpy()

    def act(self, state):
        return int(np.argmax(self.scores(state)))
