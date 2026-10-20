"""DQN trainer (PyTorch): target network, replay buffer, epsilon-greedy, N vectorized envs.

    python train_dqn.py --reward shaped --envs 8 --episodes 2000 --vision rays:depth=4

"Best" weights are chosen by a periodic greedy evaluation on fixed seeds (not counted
in the training step/game totals) and saved to models/<agent>_best.pt.
"""
import argparse
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from dqn_model import QNet, save_checkpoint
from publisher import Publisher, atomic_write
import rewards as reward_specs
import vision as visions
from snake_env import N_ACTIONS, SnakeEnv, end_code, end_counts

EVAL_SEED = 1_000_000


class ReplayBuffer:
    """States are stored as bytes (vision values are in [0, 1], kept to 1/255 precision):
    4x less memory, which matters for the 1083-number full-vision states."""

    def __init__(self, capacity, dim):
        self.s = np.zeros((capacity, dim), np.uint8)
        self.s2 = np.zeros((capacity, dim), np.uint8)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)
        self.capacity, self.pos, self.size = capacity, 0, 0

    def push(self, s, a, r, s2, d):
        idx = (self.pos + np.arange(len(a))) % self.capacity
        self.s[idx], self.s2[idx] = np.rint(s * 255), np.rint(s2 * 255)
        self.a[idx], self.r[idx], self.d[idx] = a, r, d
        self.pos = (self.pos + len(a)) % self.capacity
        self.size = min(self.size + len(a), self.capacity)

    def sample(self, n, rng):
        i = rng.integers(0, self.size, n)
        s, s2 = (torch.from_numpy(x[i]).float().div_(255) for x in (self.s, self.s2))
        return s, torch.from_numpy(self.a[i]), torch.from_numpy(self.r[i]), s2, torch.from_numpy(self.d[i])


def train_step(online, target, opt, batch, gamma):
    s, a, r, s2, d = batch
    q = online(s).gather(1, a[:, None]).squeeze(1)
    with torch.no_grad():  # Double DQN: online net picks, target net scores
        a2 = online(s2).argmax(1, keepdim=True)
        y = r + gamma * (1.0 - d) * target(s2).gather(1, a2).squeeze(1)
    loss = F.smooth_l1_loss(q, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    opt.step()
    return loss.item()


@torch.no_grad()
def greedy_eval(net, n_games, width, height, vision):
    """Play n_games greedily in parallel on fixed seeds; returns the list of scores."""
    envs = [SnakeEnv(width, height, seed=EVAL_SEED + i, vision=vision) for i in range(n_games)]
    states = [e.reset() for e in envs]
    scores, active = [0] * n_games, list(range(n_games))
    while active:
        acts = net(torch.from_numpy(np.stack([states[i] for i in active]))).argmax(1).tolist()
        still = []
        for i, a in zip(active, acts):
            states[i], _, done, info = envs[i].step(a)
            if done:
                scores[i] = info["score"]
            else:
                still.append(i)
        active = still
    return scores


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--agent", default="dqn", help="name used for the files this trainer writes")
    p.add_argument("--reward", default="score", help="preset (score, shaped) or features, e.g. apple=10,death=10")
    p.add_argument("--vision", default="short", help="what the snake sees, e.g. full, rays:depth=4")
    p.add_argument("--envs", type=int, default=8, help="vectorized environments")
    p.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    p.add_argument("--episodes", type=int, default=0, help="stop after N games (0 = no limit)")
    p.add_argument("--max-steps", type=int, default=0, help="stop after N env steps (0 = no limit)")
    p.add_argument("--time-budget", type=float, default=0, help="minutes, 0 = none")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--buffer", type=int, default=100_000)
    p.add_argument("--learn-start", type=int, default=2_000)
    p.add_argument("--target-sync", type=int, default=500, help="updates between target syncs")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.02)
    p.add_argument("--eps-decay-steps", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=20_000, help="env steps between greedy evals")
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--net", choices=("mlp", "conv"), default="mlp", help="conv: a small CNN reads the window / full pictures")
    p.add_argument("--publish-every", type=float, default=1.0, help="seconds between stats records")
    p.add_argument("--width", type=int, default=10)
    p.add_argument("--height", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--live-dir", default="live")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--models-dir", default="models")
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    method = args.agent
    vision, reward = visions.parse(args.vision), reward_specs.parse(args.reward)
    n = args.envs
    envs = [SnakeEnv(args.width, args.height, reward, seed=args.seed * 1000 + i, vision=vision)
            for i in range(n)]
    states = np.stack([e.reset() for e in envs])
    n_in = envs[0].state_size
    picture = visions.picture_info(vision, args.width, args.height) if args.net == "conv" else None
    if args.net == "conv" and not picture:
        raise SystemExit(f"--net conv needs a picture vision (window or full), not {visions.to_string(vision)}")
    online, target = QNet(n_in, picture=picture), QNet(n_in, picture=picture)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=args.lr)
    buf = ReplayBuffer(args.buffer, n_in)
    pub = Publisher(method, "episode", args.live_dir, args.results_dir)
    os.makedirs(args.models_dir, exist_ok=True)
    best_path = os.path.join(args.models_dir, f"{method}_best.pt")
    grid = {"width": args.width, "height": args.height, "reward": reward, "vision": vision}

    total_steps = total_games = best_score = updates = 0
    recent, loss = deque(maxlen=100), float("nan")
    recent_ends = deque(maxlen=100)
    best_eval, next_eval, last_pub = -1.0, args.eval_every, time.time()
    print(f"[{method}] vision={visions.to_string(vision)} reward={reward_specs.to_string(reward)} inputs={n_in} envs={n} threads={args.threads} "
          f"hidden={online.hidden} net={args.net}", flush=True)

    try:
        while True:
            eps = max(args.eps_end, args.eps_start - (args.eps_start - args.eps_end)
                      * total_steps / args.eps_decay_steps)
            with torch.no_grad():
                actions = online(torch.from_numpy(states)).argmax(1).numpy()
            explore = rng.random(n) < eps
            actions[explore] = rng.integers(0, N_ACTIONS, int(explore.sum()))

            next_states, after = np.empty_like(states), np.empty_like(states)
            rews, terminal = np.empty(n, np.float32), np.zeros(n, np.float32)
            for i, env in enumerate(envs):
                s2, r, done, info = env.step(int(actions[i]))
                next_states[i], rews[i] = s2, r
                if done:
                    terminal[i] = not info["truncated"]  # stall cut-offs still bootstrap
                    total_games += 1
                    recent.append(info["score"])
                    recent_ends.append(end_code(info))
                    best_score = max(best_score, info["score"])
                    s2 = env.reset()
                after[i] = s2
            buf.push(states, actions, rews, next_states, terminal)
            states = after
            total_steps += n

            if buf.size >= args.learn_start:
                loss = train_step(online, target, opt, buf.sample(args.batch, rng), args.gamma)
                updates += 1
                if updates % args.target_sync == 0:
                    target.load_state_dict(online.state_dict())

            if total_steps >= next_eval:
                next_eval += args.eval_every
                online.eval()
                scores = greedy_eval(online, args.eval_games, args.width, args.height, vision)
                online.train()
                mean_eval = float(np.mean(scores))
                if mean_eval > best_eval:
                    best_eval = mean_eval
                    info = {"episode": total_games, "eval_mean": mean_eval,
                            "eval_max": int(max(scores)), "total_steps": total_steps, **grid}
                    atomic_write(best_path, lambda f: save_checkpoint(f, online, **info))
                    pub.weights(lambda f: save_checkpoint(f, online, **info), "pt", total_games, **info)
                    print(f"[{method}] new best greedy eval {mean_eval:.2f} (max {max(scores)}) "
                          f"at episode {total_games}", flush=True)

            now = time.time()
            if now - last_pub >= args.publish_every:
                last_pub = now
                rec = pub.stats(total_games, total_steps, total_games, best_score,
                                np.mean(recent) if recent else 0.0, ends=end_counts(recent_ends), epsilon=round(eps, 4),
                                loss=round(loss, 5), best_eval_mean=best_eval, updates=updates)
                print(f"[{method}] ep {total_games:6d} | best {best_score:3d} | mean100 "
                      f"{rec['mean_score_100']:6.2f} | eps {eps:.3f} | loss {loss:.4f} | "
                      f"{rec['steps_per_sec']:7,.0f} steps/s | {rec['elapsed_s']:6.1f}s", flush=True)
                if pub.stop_requested():
                    break

            if ((args.episodes and total_games >= args.episodes)
                    or (args.max_steps and total_steps >= args.max_steps)
                    or (args.time_budget and pub.elapsed() >= args.time_budget * 60)):
                break
    except KeyboardInterrupt:
        print(f"[{method}] interrupted", flush=True)
    pub.stats(total_games, total_steps, total_games, best_score,
              np.mean(recent) if recent else 0.0, ends=end_counts(recent_ends), best_eval_mean=best_eval, updates=updates)
    pub.finish()
    print(f"[{method}] done. {total_games} episodes, best score {best_score}, "
          f"best greedy eval {best_eval:.2f}", flush=True)


if __name__ == "__main__":
    main()
