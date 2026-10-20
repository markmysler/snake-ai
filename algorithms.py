"""Training algorithms: what the dashboard can launch and how.

Each algorithm is a trainer script that accepts the common flags (--agent, --vision, --reward,
--seed, --width, --height, --time-budget, --live-dir, --results-dir, --models-dir) plus its own
settings listed in `params` (each one is a --flag, e.g. mutation_rate -> --mutation-rate).
`resource` names the flag that controls its CPU use, and `memory_mb` estimates its RAM.

To add an algorithm: write the trainer (copy train_dqn.py's structure: it must publish with
publisher.Publisher and save models/<agent>_best.<ext>), then register(Algorithm(...)) below.
"""
import jev


def num(name, label, default, lo, hi, kind="float", help=""):
    return {"name": name, "label": label, "type": kind, "default": default, "min": lo, "max": hi, "help": help}


def choice(name, label, default, options, help=""):
    """A setting with named options ({value: label}), e.g. which network architecture to use."""
    return {"name": name, "label": label, "type": "choice", "default": default, "options": options, "help": help}


class Algorithm:
    def __init__(self, name, label, description, script, model_ext, params, resource, memory_mb):
        self.name, self.label, self.description = name, label, description
        self.script, self.model_ext, self.params = script, model_ext, params
        self.resource = resource      # {"flag": ..., "label": ..., "role": "pool"|"threads"}
        self.memory_mb = memory_mb    # (state_size, params, cores) -> estimated MB

    def defaults(self):
        return {p["name"]: p["default"] for p in self.params}

    def clean(self, params):
        """Fill defaults, coerce types and clamp to the allowed range; unknown keys are dropped."""
        out = {}
        for p in self.params:
            v = (params or {}).get(p["name"], p["default"])
            if p["type"] == "choice":
                out[p["name"]] = str(v) if str(v) in p["options"] else p["default"]
                continue
            v = int(round(float(v))) if p["type"] == "int" else float(v)
            out[p["name"]] = min(max(v, p["min"]), p["max"])
        return out

    def info(self):
        return {"name": self.name, "label": self.label, "description": self.description,
                "params": self.params, "resource": self.resource}


ALGORITHMS = {}


def register(algo):
    ALGORITHMS[algo.name] = algo


def ga_memory(state_size, params, workers):
    genome = state_size * 16 + 16 + 16 * 3 + 3
    population_mb = params["population"] * genome * 4 * 6 / 1e6  # population, children, copies
    return 90 + population_mb + workers * (45 + genome * 4 / 1e6)


def dqn_memory(state_size, params, threads):
    buffer_mb = 2 * params["buffer"] * state_size / 1e6  # states stored as bytes (s and s')
    return 320 + buffer_mb + 40 * threads


register(Algorithm(
    "ga", "Genetic algorithm",
    "Evolution, no gradients: a population of networks plays, the fittest survive and have "
    "children (mixed weights + random mutations). Fitness = average total reward per game.",
    "train_ga.py", "npy",
    [num("population", "Population", 200, 10, 5000, "int"),
     num("games", "Games per genome", 3, 1, 50, "int", "more games = fairer fitness, slower generations"),
     num("elite_frac", "Elite fraction", 0.10, 0.0, 0.9, help="share copied unchanged to the next generation"),
     num("tournament", "Tournament size", 3, 2, 50, "int", "bigger = stronger selection pressure"),
     num("mutation_rate", "Mutation rate", 0.10, 0.0, 1.0, help="chance each weight is changed"),
     num("mutation_sigma", "Mutation size", 0.20, 0.001, 5.0, help="std-dev of each change")],
    {"flag": "workers", "label": "worker processes", "role": "pool"}, ga_memory))

register(Algorithm(
    "dqn", "Deep Q-learning (DQN)",
    "Learns to predict the future reward of each move from stored experience (replay buffer), "
    "with a target network and epsilon-greedy exploration, then plays the best predicted move.",
    "train_dqn.py", "pt",
    [num("lr", "Learning rate", 1e-3, 1e-6, 1.0),
     num("gamma", "Discount (gamma)", 0.95, 0.0, 0.9999, help="how much future rewards count"),
     num("batch", "Batch size", 256, 8, 4096, "int"),
     num("buffer", "Replay buffer", 100_000, 1_000, 5_000_000, "int", "transitions remembered"),
     num("eps_decay_steps", "Exploration decay (steps)", 100_000, 1, 10_000_000, "int"),
     num("target_sync", "Target sync (updates)", 500, 1, 100_000, "int"),
     num("envs", "Parallel games", 8, 1, 256, "int", "games stepped together in one process"),
     num("eval_every", "Evaluate every (steps)", 20_000, 1_000, 10_000_000, "int"),
     choice("net", "Network", "mlp", {"mlp": "MLP (fully connected)", "conv": "CNN (convolutional)"},
            "CNN reads the window / whole-board pictures as images; needs one of those visions")],
    {"flag": "threads", "label": "torch threads", "role": "threads"}, dqn_memory))


def jev_memory(state_size, params, threads):
    return 120  # it plays through an API: just the games in memory


register(Algorithm(
    "jev", "jev (TypeSafe decision model)",
    "No training: a ready-made decision model reads what the snake senses, as text, and picks each move. It runs on "
    "TokenRouter (needs TOKENROUTER_API_KEY) and costs money per move, about $0.03 for 20 games. Its run plays the "
    "fixed test games directly; what you can tune is its instructions.",
    "train_jev.py", "json",
    [num("games", "Test games", 20, 1, 100, "int", "fixed test games it plays (the other algorithms are tested on 100)"),
     choice("prompt", "Instructions", "rules", {name: p["label"] for name, p in jev.PROMPTS.items()},
            "the prompt is jev's only 'training': try both"),
     num("batch", "Boards per request", 100, 1, 100, "int", "games asked about in one request: more = faster and cheaper"),
     num("max_cost", "Spending cap (USD)", 0.5, 0.01, 20.0, help="the run stops before spending more than this")],
    {"flag": "threads", "label": "request threads", "role": "threads"}, jev_memory))


def catalog():
    return [a.info() for a in ALGORITHMS.values()]
