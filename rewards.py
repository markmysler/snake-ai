"""Reward systems built from switchable features (the checkboxes in the dashboard's New run form).

A reward spec maps feature -> weight, e.g. {"apple": 10, "death": 10, "hunger": 0.01}, or as a
string "apple=10,death=10,hunger=0.01", or a preset name ("score", "shaped", "ga_classic").
The GA uses the same rewards: a genome's fitness is its average total reward per game.

To add a feature: add it to FEATURES (with any incompatibilities) and handle it in Rewarder.
"""

FEATURES = {
    "apple": {"label": "Apple eaten", "default": 10.0, "when": "per apple",
              "description": "Added every time the snake eats. The main goal."},
    "death": {"label": "Death penalty", "default": 10.0, "when": "on crash", "sign": -1,
              "incompatible": ["death_wall", "death_body"],
              "description": "Subtracted when the snake hits a wall or itself."},
    "death_wall": {"label": "Wall crash penalty", "default": 10.0, "when": "on hitting a wall", "sign": -1,
                   "incompatible": ["death"],
                   "description": "Subtracted only when the snake hits a wall. Use with the body crash penalty "
                                  "to weigh the two kinds of crash differently."},
    "death_body": {"label": "Body crash penalty", "default": 10.0, "when": "on hitting itself", "sign": -1,
                   "incompatible": ["death"],
                   "description": "Subtracted only when the snake runs into its own body, the usual way a "
                                  "long snake dies."},
    "starve": {"label": "Starvation penalty", "default": 10.0, "when": "on starving", "sign": -1,
               "description": "Subtracted when the game ends because the snake went 100 x length "
                              "moves without eating (looping forever)."},
    "win": {"label": "Board-full bonus", "default": 100.0, "when": "on filling the board",
            "description": "Added when the snake fills every square."},
    "distance": {"label": "Distance hint", "default": 0.1, "when": "per move", "incompatible": ["potential"],
                 "description": "+w for moving closer to the apple, -w for moving away. Simple, but it can "
                                "change which strategy is best (e.g. rush straight at the apple)."},
    "potential": {"label": "Potential-based hint", "default": 0.1, "when": "per move", "incompatible": ["distance"],
                  "description": "gamma x Phi(after) - Phi(before) with Phi = -w x distance to the apple. "
                                 "Proven not to change the best strategy; it only speeds up learning."},
    "hunger": {"label": "Hunger cost", "default": 0.01, "when": "per move", "incompatible": ["survival"], "sign": -1,
               "description": "Subtracted every move, so eating sooner is always better."},
    "survival": {"label": "Survival bonus", "default": 0.1, "when": "per move", "incompatible": ["hunger"],
                 "description": "Added every move the snake stays alive. Pays for wandering and looping "
                                "(this was part of the GA's fitness in lessons 01-02)."},
}
PRESETS = {
    "score": {"apple": 10.0, "death": 10.0},
    "shaped": {"apple": 10.0, "death": 10.0, "distance": 0.1},
    "ga_classic": {"apple": 100.0, "survival": 0.1},  # fitness = apples x 100 + moves x 0.1
    # body crashes cost more; starving costs as much as the worst crash, so looping is no escape
    "split_death": {"apple": 10.0, "death_wall": 10.0, "death_body": 30.0, "starve": 30.0},
}


def sign(name):
    """-1 for penalties (the weight is subtracted), +1 for rewards."""
    return FEATURES[name].get("sign", 1)
POTENTIAL_GAMMA = 0.95  # matches the DQN's default discount


def parse(spec):
    """Preset name, 'a=1,b=2' string or dict -> {feature: weight} (validated names, float weights)."""
    if spec is None:
        spec = "score"
    if isinstance(spec, str):
        if spec in PRESETS:
            return dict(PRESETS[spec])
        spec = dict(kv.split("=", 1) for kv in spec.split(",") if kv)
    out = {}
    for name, weight in spec.items():
        if name not in FEATURES:
            raise ValueError(f"unknown reward feature {name!r}; choose from {sorted(FEATURES)}")
        out[name] = float(weight)
    return out


def to_string(spec):
    return ",".join(f"{k}={v:g}" for k, v in parse(spec).items())


def validate(spec):
    """Returns (errors, warnings) as lists of plain-English strings."""
    spec = parse(spec)
    errors, warnings = [], []
    for name in spec:
        for other in FEATURES[name].get("incompatible", []):
            if other in spec and name < other:
                errors.append(f"'{FEATURES[name]['label']}' and '{FEATURES[other]['label']}' "
                              "can't be used together.")
    for name, weight in spec.items():
        if weight < 0:
            errors.append(f"'{FEATURES[name]['label']}' needs a positive weight (the sign is built in).")
    if spec.get("apple", 0) <= 0:
        warnings.append("Without an apple reward nothing tells the snake to eat.")
    if "survival" in spec and "starve" not in spec:
        warnings.append("A survival bonus without a starvation penalty pays the snake to loop forever.")
    crash = max(spec.get(k, 0.0) for k in ("death", "death_wall", "death_body"))
    if crash > spec.get("apple", 0.0) and spec.get("starve", 0.0) < crash and "hunger" not in spec:
        warnings.append("Crashing costs more than an apple is worth, but starving costs less and time is free: "
                        "the snake may learn to loop until it starves rather than risk a crash. Raise the "
                        "starvation penalty or add a hunger cost.")
    return errors, warnings


def describe(spec):
    spec = parse(spec)
    for name, preset in PRESETS.items():
        if spec == preset:
            return name
    return ", ".join(f"{name} {'-' if sign(name) < 0 else '+'}{w:g}" for name, w in spec.items())


def catalog():
    return {"features": [{"name": k, **v, "incompatible": v.get("incompatible", []), "sign": sign(k)}
                         for k, v in FEATURES.items()],
            "presets": PRESETS}


class Rewarder:
    """Turns what happened in one move into a reward."""

    def __init__(self, spec):
        self.w = parse(spec)

    def __call__(self, ate, died, starved, won, dist_before, dist_after):
        """died: False, or how the snake crashed ("wall" / "body"; True counts as a crash of unknown kind)."""
        w, r = self.w, 0.0
        if died:
            r -= w.get("death", 0.0)
            if died == "wall":
                r -= w.get("death_wall", 0.0)
            elif died == "body":
                r -= w.get("death_body", 0.0)
        if ate:
            r += w.get("apple", 0.0)
        if won:
            r += w.get("win", 0.0)
        if starved:
            r -= w.get("starve", 0.0)
        r += w.get("survival", 0.0) - w.get("hunger", 0.0)
        if not died and not ate and "distance" in w:
            r += w["distance"] if dist_after < dist_before else -w["distance"]
        if "potential" in w and not died:
            after = 0.0 if dist_after is None else -w["potential"] * dist_after  # no apple left: Phi = 0
            r += POTENTIAL_GAMMA * after + w["potential"] * dist_before
        return r
