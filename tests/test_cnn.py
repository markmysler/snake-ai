"""DQN with a small CNN for the picture visions (window, full)."""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import runconfig  # noqa: E402
import vision as visions  # noqa: E402
from algorithms import ALGORITHMS  # noqa: E402
from dqn_model import QNet, TorchPolicy, load_qnet, save_checkpoint  # noqa: E402
from policies import load_policy  # noqa: E402
from snake_env import SnakeEnv  # noqa: E402


@pytest.mark.parametrize("spec", ["window:radius=2", "window:radius=4", "full"])
def test_conv_net_reads_the_pictures(tmp_path, spec):
    env = SnakeEnv(10, 10, vision=spec, seed=1)
    obs = env.reset()
    net = QNet(len(obs), picture=visions.picture_info(spec))
    assert net(torch.from_numpy(obs)).shape == (3,)
    assert net(torch.from_numpy(np.stack([obs, obs]))).shape == (2, 3)
    mlp = QNet(len(obs))
    assert sum(p.numel() for p in net.parameters()) < sum(p.numel() for p in mlp.parameters()) or spec != "full"
    path = tmp_path / "m.pt"
    save_checkpoint(str(path), net, vision=visions.parse(spec), width=10, height=10)
    loaded, meta = load_qnet(str(path))
    assert loaded.picture == net.picture
    assert torch.allclose(loaded(torch.from_numpy(obs)), net(torch.from_numpy(obs)))
    act, _ = load_policy(str(path))
    assert act(obs) == TorchPolicy(net).act(obs)


def test_net_setting_is_a_choice_and_validated():
    dqn = ALGORITHMS["dqn"]
    assert dqn.clean({"net": "conv"})["net"] == "conv" and dqn.clean({"net": "nope"})["net"] == "mlp"
    bad = runconfig.normalize({"vision": "rays", "agents": [{"algorithm": "dqn", "params": {"net": "conv"}}]})
    assert any("CNN" in e for e in runconfig.validate(bad)[0])
    ok = runconfig.normalize({"vision": "full", "agents": [{"algorithm": "dqn", "params": {"net": "conv"}}]})
    assert not runconfig.validate(ok)[0]
    cmd = runconfig.trainer_command("python", ok, ok["agents"][0], 1, ok["agents"][0]["params"], {"live": "l", "results": "r", "models": "m"})
    assert cmd[cmd.index("--net") + 1] == "conv"


def test_trainer_runs_with_a_cnn(tmp_path):
    subprocess.run([sys.executable, "train_dqn.py", "--vision", "window:radius=2", "--net", "conv", "--max-steps", "4000",
                    "--eval-every", "1000", "--live-dir", str(tmp_path / "live"), "--results-dir", str(tmp_path / "results"),
                    "--models-dir", str(tmp_path / "models")], cwd=ROOT, check=True, capture_output=True, timeout=300)
    net, meta = load_qnet(str(tmp_path / "models" / "dqn_best.pt"))
    assert net.picture["side"] == 5


def test_a_cnn_gets_more_threads():
    import system
    machine = {"cpus": 22, "ram_available_mb": 8000, "ram_total_mb": 16000}
    plan = lambda agents: {k: v["cores"] for k, v in system.plan(  # noqa: E731
        runconfig.normalize({"vision": "full", "agents": agents}), machine, parallel=2)["agents"].items()}
    assert plan(["dqn"]) == {"dqn": 1}
    assert plan([{"algorithm": "dqn", "params": {"net": "conv"}}]) == {"dqn": system.CONV_THREADS}
    assert plan(["ga", {"algorithm": "dqn", "params": {"net": "conv"}}])["dqn"] == system.CONV_THREADS
