"""A run's full recipe: everything needed to repeat it exactly.

    {"lesson": "03-rays", "name": "rays 4", "note": "...",
     "vision": {"type": "rays", "depth": 4},           # see vision.py
     "reward": {"apple": 10, "death": 10},             # see rewards.py (shared by all agents)
     "agents": [{"id": "ga", "algorithm": "ga", "params": {...}},      # see algorithms.py
                {"id": "dqn", "algorithm": "dqn", "params": {...}, "reward": optional override}],
     "time_budget_min": 10, "width": 10, "height": 10, "seed": 0}

The recipe is stored in every run's run.json; "Run again" in the dashboard reuses it.
"""
import re

import rewards
import vision as visions
from algorithms import ALGORITHMS

DEFAULTS = {"lesson": "scratch", "name": "", "note": "", "vision": "short", "reward": "score",
            "agents": ["ga", "dqn"], "time_budget_min": 10.0, "width": 10, "height": 10, "seed": 0}
MAX_BUDGET_MIN = 24 * 60


def slug(text):
    return re.sub(r"[^a-z0-9_-]+", "-", str(text).strip().lower()).strip("-")[:60]


def normalize(cfg):
    """Fill defaults and canonicalize every part. Raises ValueError on unknown names."""
    c = {**DEFAULTS, **{k: v for k, v in cfg.items() if v is not None}}
    own = [a for a in cfg.get("agents") or [] if isinstance(a, dict) and a.get("reward") is not None]
    if "reward" in cfg and cfg["reward"] is None and own and len(own) == len(cfg["agents"]):
        c["reward"] = "per-agent"  # a recorded per-agent recipe (reward: null) stays per agent
    c["lesson"] = slug(c["lesson"]) or "scratch"
    c["name"], c["note"] = str(c["name"]).strip()[:120], str(c["note"]).strip()[:2000]
    c["width"] = min(max(int(c["width"]), 6), 30)
    c["height"] = min(max(int(c["height"]), 6), 30)
    c["vision"] = visions.parse(c["vision"])
    c["reward"] = rewards.parse(c["reward"]) if c["reward"] not in (None, "per-agent") else None
    c["time_budget_min"] = min(max(float(c["time_budget_min"]), 0.0), MAX_BUDGET_MIN)
    c["seed"] = int(c["seed"])
    agents, seen = [], set()
    for a in c["agents"]:
        a = {"algorithm": a} if isinstance(a, str) else dict(a)
        if a.get("algorithm") not in ALGORITHMS:
            raise ValueError(f"unknown algorithm {a.get('algorithm')!r}; choose from {sorted(ALGORITHMS)}")
        algo = ALGORITHMS[a["algorithm"]]
        base = slug(a.get("id") or algo.name)
        agent_id, n = base, 2
        while agent_id in seen:
            agent_id, n = f"{base}-{n}", n + 1
        seen.add(agent_id)
        out = {"id": agent_id, "algorithm": algo.name, "params": algo.clean(a.get("params"))}
        if a.get("reward") is not None:
            out["reward"] = rewards.parse(a["reward"])
        if a.get("cores"):
            out["cores"] = max(1, int(a["cores"]))
        agents.append(out)
    c["agents"] = agents
    return c


def agent_reward(c, agent):
    return agent.get("reward") or c["reward"] or rewards.parse("score")


def validate(c):
    """(errors, warnings) for a normalized config, in plain English."""
    errors, warnings = [], []
    if not c["agents"]:
        errors.append("Pick at least one training algorithm.")
    seen = set()
    for a in c["agents"]:
        key = rewards.to_string(agent_reward(c, a))
        if key not in seen:
            seen.add(key)
            e, w = rewards.validate(agent_reward(c, a))
            errors += e
            warnings += w
    if c["time_budget_min"] <= 0 and not all(a["algorithm"] == "jev" for a in c["agents"]):  # (jev stops by itself)
        warnings.append("No time budget: the run trains until you stop it.")
    if any(a["algorithm"] == "jev" for a in c["agents"]):
        import jev
        if not jev.configured():
            errors.append("jev runs on TokenRouter: add TOKENROUTER_API_KEY to .env (see .env.example).")
        if c["time_budget_min"]:
            warnings.append("jev plays its test games until they end; a time limit may cut some short (they are then "
                            "left out). Use no time limit for a jev-only run.")
    if any(a["params"].get("net") == "conv" for a in c["agents"]) and not visions.picture_info(c["vision"]):
        errors.append("The CNN network reads pictures: pick the window or whole-board vision, or use the MLP.")
    inputs = visions.state_size(c["vision"], c["width"], c["height"])
    if inputs > 2000:
        warnings.append(f"This vision gives {inputs:,} inputs: training will be slow (see lesson 02).")
    return errors, warnings


def trainer_command(python, c, agent, cores, params, dirs):
    """Command line that starts one agent's trainer. dirs = {live, results, models}."""
    algo = ALGORITHMS[agent["algorithm"]]
    cmd = [python, algo.script, "--agent", agent["id"], "--vision", visions.to_string(c["vision"]),
           "--reward", rewards.to_string(agent_reward(c, agent)), "--seed", str(c["seed"]),
           "--width", str(c["width"]), "--height", str(c["height"]),
           "--live-dir", dirs["live"], "--results-dir", dirs["results"], "--models-dir", dirs["models"],
           f"--{algo.resource['flag']}", str(cores)]
    if c["time_budget_min"]:
        cmd += ["--time-budget", str(c["time_budget_min"])]
    for name, value in params.items():
        cmd += ["--" + name.replace("_", "-"), str(value)]
    return cmd


def legacy(config):
    """Recipe for runs recorded before recipes existed (lessons 01-02): GA + DQN(score) + DQN(shaped)."""
    return normalize({**{k: v for k, v in config.items() if k in DEFAULTS}, "reward": "per-agent",
                      "agents": [{"id": "ga", "algorithm": "ga", "reward": "ga_classic"},
                                 {"id": "dqn_score", "algorithm": "dqn", "reward": "score"},
                                 {"id": "dqn_shaped", "algorithm": "dqn", "reward": "shaped"}]})


def describe(c):
    """Short human labels for the dashboard and LESSON.md."""
    reward = rewards.describe(c["reward"]) if c["reward"] else "per agent"
    return {"vision": visions.label(c["vision"]), "vision_spec": visions.to_string(c["vision"]),
            "inputs": visions.state_size(c["vision"], c["width"], c["height"]), "reward": reward,
            "agents": [{"id": a["id"], "algorithm": a["algorithm"], "label": ALGORITHMS[a["algorithm"]].label,
                        "reward": rewards.describe(agent_reward(c, a))} for a in c["agents"]]}
