"""The glossary: what every algorithm, setting, vision and reward feature does, plus the words the
dashboard and the AI analysis use. Shown in the dashboard's Learn tab and next to every option.

Entries are built from the registries (algorithms.py, vision.py, rewards.py), so anything added
there appears here automatically with its own description. EXTRA adds a longer explanation and
"what happens if you raise / lower it" for the ones that have been written up; CONCEPTS are the
general terms. Each entry:

    {"id", "term", "kind": algorithm|setting|vision|reward|preset|concept, "group", "short", "long",
     "up", "down", "default", "min", "max", "aliases", "related"}

`aliases` are the words that point to an entry when they appear in a recommendation (matched
whole-word, case-insensitive), so keep them specific.
"""
import algorithms
import rewards
import vision as visions

# ---- longer explanations of registry entries (key = entry id) ---------------------------------
EXTRA = {
    # algorithms
    "ga": {"long": "Start with a population of random networks. Each one plays a few games and gets a fitness score "
                   "(its average total reward). The best few are copied unchanged (elites); the rest of the next "
                   "generation are children of two parents picked by tournament, with their weights mixed "
                   "(crossover) and nudged at random (mutation). No gradients and no memory of single moves: "
                   "it only learns which whole networks did better. It is cheap per game and spreads over many "
                   "CPU cores, so it plays a huge number of games.",
           "aliases": ["GA", "genetic algorithm", "neuroevolution", "evolution"],
           "related": ["c.generation", "c.fitness", "c.selection", "ga.population", "ga.mutation_rate"]},
    "dqn": {"long": "Deep Q-learning trains one network to predict, for each move, the total future reward it "
                    "expects (its Q-value). It plays, stores every move in a replay buffer, and repeatedly learns "
                    "from random batches of past moves. It explores by sometimes moving at random (epsilon), less "
                    "and less over time. This lab uses Double DQN: a second, slowly updated target network "
                    "keeps the predictions from chasing themselves.",
            "aliases": ["DQN", "deep Q-learning", "Q-learning", "double DQN"],
            "related": ["c.q-value", "c.epsilon", "dqn.gamma", "dqn.buffer", "c.target-network"]},
    # GA settings
    "ga.population": {"long": "How many networks are tried in each generation.",
                      "up": "More variety to choose from and steadier progress, but each generation takes longer.",
                      "down": "Faster generations, but less variety; the population can get stuck on one idea.",
                      "aliases": ["population", "population size"]},
    "ga.games": {"long": "Each network's fitness is its average over this many games, each with a different apple "
                         "sequence.",
                 "up": "Fairer fitness (less luck), so selection picks truly better networks; slower generations.",
                 "down": "Faster, but a network can win by being lucky in one game.",
                 "aliases": ["games per genome"], "related": ["c.fitness", "c.noise"]},
    "ga.elite_frac": {"long": "The share of the best networks copied unchanged into the next generation, so the best "
                              "solution found so far is never lost.",
                      "up": "Safer but slower to explore: more of each generation is a copy.",
                      "down": "More exploration, but a lucky good network can be lost.",
                      "aliases": ["elite", "elites", "elite fraction", "elitism"]},
    "ga.tournament": {"long": "To pick a parent, this many random networks are compared and the fittest wins.",
                      "up": "Stronger selection pressure: the best networks have most children, which is fast but "
                            "can lose variety.",
                      "down": "Weaker pressure: more variety survives, progress is slower.",
                      "aliases": ["tournament", "tournament size", "selection pressure"], "related": ["c.selection"]},
    "ga.mutation_rate": {"long": "The chance that each weight of a child network is changed at random.",
                         "up": "More exploration, but good networks get damaged more often.",
                         "down": "Children stay close to their parents: fine-tuning, but slow to find new ideas.",
                         "aliases": ["mutation rate", "mutation_rate", "mutation"], "related": ["ga.mutation_sigma"]},
    "ga.mutation_sigma": {"long": "How big each random change is (the standard deviation of the noise added to a weight).",
                          "up": "Bigger jumps: escapes dead ends, but rarely improves an already good network.",
                          "down": "Small careful steps: good late in training, slow early on.",
                          "aliases": ["mutation size", "mutation_sigma", "sigma"]},
    # DQN settings
    "dqn.lr": {"long": "Learning rate: how big a step the network takes each time it learns from a batch.",
               "up": "Learns faster at first, but can become unstable and forget.",
               "down": "Steadier but slower learning.",
               "aliases": ["learning rate", "lr"]},
    "dqn.gamma": {"long": "Discount factor: how much a reward k moves in the future counts now (gamma^k). It sets "
                          "how far ahead the snake plans: roughly 1 / (1 − gamma) moves, so 0.95 ≈ 20 moves and "
                          "0.99 ≈ 100 moves.",
                  "up": "Plans further ahead (avoids trapping itself), but the predictions get noisier and learning "
                        "slower; per-move rewards like survival add up to much more.",
                  "down": "Short-sighted: grabs the nearest apple, ignores consequences further away.",
                  "aliases": ["gamma", "discount", "discount factor", "discounting", "horizon"],
                  "related": ["c.q-value", "reward.survival"]},
    "dqn.batch": {"long": "How many stored moves the network learns from at once.",
                  "up": "Smoother, more reliable updates; each one takes longer.",
                  "down": "Faster updates, noisier learning.",
                  "aliases": ["batch size"]},
    "dqn.buffer": {"long": "The replay buffer keeps the most recent moves (state, action, reward, next state). "
                           "Learning from random old moves instead of only the latest ones keeps training stable.",
                   "up": "Remembers more varied experience; uses more RAM (the main memory cost of a run).",
                   "down": "Less memory; learns mostly from recent play, which can make it forget older lessons.",
                   "aliases": ["replay buffer", "buffer"], "related": ["c.replay"]},
    "dqn.eps_decay_steps": {"long": "Epsilon, the chance of a random move, falls from 1 to its minimum over this many "
                                    "env steps.",
                            "up": "Explores longer: finds more strategies, but the score rises later.",
                            "down": "Starts exploiting what it knows sooner; may settle on a mediocre strategy.",
                            "aliases": ["eps_decay_steps", "epsilon decay", "exploration schedule"],
                            "related": ["c.epsilon"]},
    "dqn.target_sync": {"long": "Every this many learning steps, the target network is set to a copy of the main "
                                "network.",
                        "up": "More stable targets, slower to take in what it has learned.",
                        "down": "Faster but can oscillate or diverge.",
                        "aliases": ["target_sync", "target sync", "target update"], "related": ["c.target-network"]},
    "dqn.envs": {"long": "How many games DQN plays side by side to collect experience (one network, several boards).",
                 "up": "More varied experience per second.",
                 "down": "Less experience per second, but each learning step sees fresher data.",
                 "aliases": ["parallel games"]},
    "dqn.eval_every": {"long": "Every this many env steps, DQN plays test games without random moves; the best "
                               "network so far is saved as the run's model.",
                       "up": "Less time spent testing, but the saved model is picked from fewer checkpoints.",
                       "down": "Finds a good checkpoint more precisely, costs a bit of training time.",
                       "aliases": ["eval_every", "evaluation interval", "checkpoint"]},
    # visions
    "vision.short": {"long": "11 numbers: is there danger one square straight / left / right, which way the snake is "
                             "heading, and whether the apple is ahead, behind, left or right. Tiny networks learn it "
                             "fast, but the snake can't see its own body two squares away, so long snakes trap "
                             "themselves.",
                     "aliases": ["short vision", "adjacent", "11 inputs"], "related": ["c.inputs"]},
    "vision.rays": {"long": "Looks along 8 lines (straight, diagonals, sides, behind) from the head and reports how "
                            "close the nearest wall, body part and apple are on each line, up to depth squares away.",
                    "aliases": ["rays", "ray vision", "ray depth"], "related": ["vision.rays.depth"]},
    "vision.rays.depth": {"long": "How many squares each ray looks.",
                          "up": "Sees obstacles earlier; on a 10×10 board depth 9 covers the whole board.",
                          "down": "Cheaper but short-sighted.",
                          "aliases": ["depth"]},
    "vision.room": {"long": "The 11 adjacent hints plus 3 numbers: for the moves straight, left and right, the share of the "
                            "board's free squares the snake could still reach after that move (a flood fill that stops at "
                            "walls and the body). A move into a small pocket scores low while it is still safe, so the "
                            "network can learn not to box itself in, the usual way a long snake dies.",
                    "aliases": ["room vision", "room to move", "flood fill", "reachable space", "flood-fill"],
                    "related": ["vision.short", "reward.death_body", "c.game-end"]},
    "vision.window": {"long": "A small picture of the squares around the head (3 layers: blocked, body, apple), "
                              "rotated so the snake always faces up. The network has to learn to read the picture.",
                      "aliases": ["window vision"], "related": ["vision.window.radius", "c.egocentric"]},
    "vision.window.radius": {"long": "How many squares the window reaches from the head in each direction.",
                             "up": "Sees more, but inputs grow fast (3·(2r+1)²): slower learning, especially for the GA.",
                             "down": "Fewer inputs, faster learning, less foresight.",
                             "aliases": ["radius"]},
    "vision.full": {"long": "The whole board as a picture centred on the head and rotated so the snake faces up. All "
                            "the information there is, but over a thousand inputs: DQN learns slowly per second and "
                            "the GA struggles (lesson 02).",
                    "aliases": ["full vision", "full board", "whole board"], "related": ["c.inputs", "c.egocentric"]},
    # rewards
    "reward.apple": {"long": "The main goal. Everything else is a hint or a penalty around it.",
                     "aliases": ["apple reward"]},
    "reward.death": {"long": "Makes crashing costly, so the snake learns to avoid walls and itself.",
                     "aliases": ["death penalty"]},
    "reward.death_wall": {"long": "Charged only for hitting a wall. Together with the body crash penalty it replaces the single "
                                  "death penalty, so the two kinds of crash can cost different amounts (lesson 05 showed that "
                                  "one shared penalty can trade body crashes for wall crashes).",
                          "aliases": ["wall crash penalty", "death_wall"], "related": ["reward.death_body", "c.game-end"]},
    "reward.death_body": {"long": "Charged only for running into its own body, the usual way a long snake dies. Make it larger "
                                  "than the wall penalty to push the snake to keep room for itself.",
                          "aliases": ["body crash penalty", "death_body"], "related": ["reward.death_wall", "vision.room"]},
    "preset.split_death": {"long": "Apple +10, wall crash −10, body crash −30, starving −30: crashes into itself cost three "
                                   "times as much as wall crashes, and starving costs as much as the worst crash, otherwise "
                                   "looping until it starves becomes the safe way out (a GA finds it within a minute).",
                           "aliases": ["split_death"]},
    "reward.starve": {"long": "A game also ends after 100 × length moves without eating. This penalty discourages "
                              "looping forever.",
                      "aliases": ["starve", "starvation", "starving"]},
    "reward.win": {"long": "A bonus for filling the whole board. Rarely reached on purpose, but it marks the true end goal.",
                   "aliases": ["board-full bonus", "win bonus"]},
    "reward.distance": {"long": "A small reward for every move toward the apple and a penalty for moving away. Speeds "
                                "up early learning, but the snake may take risky shortcuts it is paid for.",
                        "aliases": ["distance hint", "distance reward", "distance shaping"],
                        "related": ["c.shaping", "reward.potential"]},
    "reward.potential": {"long": "Potential-based shaping: pays γ·Φ(after) − Φ(before) with Φ = −distance to the apple. "
                                 "It guides like the distance hint but provably doesn't change which strategy is best "
                                 "(moving away and back again earns nothing overall).",
                         "aliases": ["potential", "potential-based", "potential-based shaping"],
                         "related": ["c.shaping", "reward.distance"]},
    "reward.hunger": {"long": "A small cost for every move, so a fast route to the apple is better than a slow one. "
                              "The opposite of the survival bonus.",
                      "aliases": ["hunger", "hunger penalty", "step penalty", "time penalty"],
                      "related": ["reward.survival"]},
    "reward.survival": {"long": "A small reward for every move alive. It helps the GA tell early random snakes apart "
                                "(they all eat nothing, but some survive longer). Too big compared with the apple and "
                                "the snake learns to wander instead of eat.",
                        "aliases": ["survival", "survival bonus", "survival reward"],
                        "related": ["reward.hunger", "c.fitness"]},
    "preset.score": {"long": "Only what counts in the game: +apple, −death. The cleanest goal, but early on every "
                             "snake scores the same, which leaves the GA nothing to select.",
                     "aliases": ["score reward"]},
    "preset.shaped": {"long": "The score reward plus the distance hint.", "aliases": ["shaped reward"]},
    "preset.ga_classic": {"long": "Apple × 100 plus survival 0.1: the GA fitness of lessons 01 and 02.",
                          "aliases": ["ga_classic"]},
}

# ---- general terms ------------------------------------------------------------------------------
CONCEPTS = [
    ("c.env-step", "Env step", "One move of the snake in one game.",
     "The unit of experience. Comparing scores per env step shows how much experience a method needed.",
     ["env step", "env steps", "environment step"], ["c.sample-efficiency"]),
    ("c.episode", "Episode / game", "One game, from the first move to a crash or starvation.",
     "DQN counts its progress in episodes.", ["episode", "episodes"], []),
    ("c.generation", "Generation", "One round of the GA: every network plays, then the next population is bred.",
     "The GA counts its progress in generations.", ["generation", "generations"], ["ga"]),
    ("c.fitness", "Fitness", "How the GA scores a network: its average total reward over its games.",
     "Only the order matters (selection is rank-based), so what changes the GA is which rewards are included and "
     "their ratio, not their size.", ["fitness"], ["c.selection", "ga.games"]),
    ("c.selection", "Selection pressure", "How strongly the GA favours its best networks when picking parents.",
     "Strong pressure converges fast but can get stuck; weak pressure explores more.",
     ["selection", "rank-based selection", "rank-based"], ["ga.tournament", "ga.elite_frac"]),
    ("c.q-value", "Q-value", "DQN's prediction of the total future reward of making a move in a state.",
     "The snake picks the move with the highest Q-value. Future rewards are discounted by gamma.",
     ["Q-value", "Q-values", "Q value"], ["dqn", "dqn.gamma"]),
    ("c.epsilon", "Epsilon (exploration)", "The chance that DQN makes a random move instead of its best guess.",
     "Starts at 1 (all random) and decays, so it first explores and later uses what it learned. Training scores "
     "include these random moves; test games don't.", ["epsilon", "exploration", "epsilon-greedy"],
     ["dqn.eps_decay_steps"]),
    ("c.replay", "Experience replay", "Learning from random batches of stored past moves instead of only the latest one.",
     "Breaks the link between consecutive moves, which makes neural network training much more stable.",
     ["experience replay"], ["dqn.buffer"]),
    ("c.target-network", "Target network", "A slowly updated copy of DQN's network used to compute learning targets.",
     "Without it the network would chase its own changing predictions.", ["target network"], ["dqn.target_sync"]),
    ("c.test-score", "Test score", "The saved best model playing 100 fixed-seed games, without random moves.",
     "The fair number to compare runs on: same 100 games for every model. Shown as mean / median / max apples.",
     ["test score", "test mean", "evaluation", "test games"], ["c.train-mean"]),
    ("c.train-mean", "Training mean (last 100)", "The average apples of the last 100 games played while training.",
     "Includes DQN's random exploration moves and the GA's whole population, so it is usually below the test score "
     "(or above it for a lucky GA).", ["training mean", "mean of last 100", "mean_last_100"], ["c.test-score"]),
    ("c.best-game", "Best game", "The highest score in any single training game.",
     "It shows what the snake can do on a good day, but it is one lucky game: compare runs on the test score.", ["best game"], ["c.test-score"]),
    ("c.sample-efficiency", "Sample efficiency", "How much score a method gets per env step of experience.",
     "DQN learns from every move many times, so it needs far fewer steps; the GA makes up for it with raw speed.",
     ["sample efficiency", "sample-efficient", "sample efficient"], ["c.env-step", "c.wall-clock"]),
    ("c.wall-clock", "Wall-clock time", "Real time on this machine. Every algorithm in a run gets the same.",
     "Fair for practical speed, but depends on the machine (cores, battery, other runs).", ["wall-clock", "wall clock"],
     ["c.sample-efficiency"]),
    ("c.plateau", "Plateau", "The learning curve flattens: more training stops helping.",
     "Usually a sign that something else limits the score (what the snake can see, how far ahead it plans), "
     "not training time.", ["plateau", "plateaued", "plateaus"], []),
    ("c.noise", "Run-to-run noise", "Two runs with the same recipe and different seeds score differently by chance.",
     "One run is one seed. Treat small differences between runs as maybe noise; repeat with another seed to check.",
     ["noise", "single seed", "single-seed"], ["c.seed"]),
    ("c.seed", "Seed", "The number that fixes all randomness: same seed + same recipe = the same run.",
     "Change only the seed to measure run-to-run noise.", ["seed"], ["c.noise", "c.seed-spread"]),
    ("c.seed-spread", "Seed spread", "How much one recipe's test score varies when only the seed changes.",
     "Set Seeds above 1 in New run to train the same recipe several times (a seed group). Each run's test score is "
     "a dot; the range from lowest to highest is the spread. Two recipes only differ for sure when their ranges "
     "don't overlap; overlapping ranges mean the difference could be luck.",
     ["seed spread", "seed group", "spread", "error bar", "seeds"], ["c.noise", "c.seed", "c.test-score"]),
    ("c.game-end", "How test games ended", "Each test game ends one of four ways: hitting a wall, hitting its own body, "
     "starving (100 × length moves without eating) or filling the board (won).",
     "Shows what limits a snake. Mostly body crashes: it eats well but traps itself as it grows. Mostly starving: it "
     "has learned to wander instead of eat (often a sign the reward pays for surviving). Steps per apple tells the "
     "same story: around 10 is efficient, hundreds means wandering.",
     ["how games ended", "starved share", "steps per apple", "steps per game", "game end"],
     ["reward.starve", "reward.survival", "c.test-score"]),
    ("c.shaping", "Reward shaping", "Extra small rewards that hint at progress (like moving toward the apple).",
     "Makes learning faster, but a badly designed hint can be exploited: the snake maximises the hint instead of the "
     "goal.", ["shaping", "reward shaping"], ["reward.distance", "reward.potential"]),
    ("c.inputs", "Inputs", "How many numbers the network sees each move (set by the vision).",
     "More inputs = more information but more weights to learn: slower learning, especially for the GA.",
     ["inputs"], ["vision.short", "vision.full"]),
    ("c.egocentric", "Egocentric view", "The picture is centred on the head and rotated so the snake always faces up.",
     "The network then learns one rule for \"wall ahead\" instead of four (one per direction).",
     ["egocentric"], ["vision.window", "vision.full"]),
    ("c.hidden", "Network / hidden layer", "The policy is a small neural network: inputs → hidden layers → 3 move scores.",
     "GA networks have 16 hidden units; DQN's have two layers of 128.", ["neural network", "hidden layer", "hidden units"], []),
]

KIND_ORDER = ["algorithm", "setting", "vision", "reward", "preset", "concept"]
# Registry names that are everyday words: never used to spot a term in free text ("too short", "the score")
GENERIC = {"short", "full", "score", "games", "apple", "death", "win", "batch", "envs", "window", "shaped"}


def _entry(id, term, kind, short, group=None, **more):
    e = {"id": id, "term": term, "kind": kind, "group": group, "short": short, "long": "", "up": None, "down": None,
         "aliases": [], "related": []}
    e.update({k: v for k, v in more.items() if v is not None})
    e.update(EXTRA.get(id, {}))
    if e["short"].strip().lower() == term.strip().lower() and e["long"]:  # no help text: lead with the explanation
        first, _, rest = e["long"].partition(". ")
        e["short"], e["long"] = first.rstrip(".") + ".", rest
    e["short"] = e["short"][:1].upper() + e["short"][1:]
    return e


def entries():
    out = []
    for algo in algorithms.ALGORITHMS.values():
        out.append(_entry(algo.name, algo.label, "algorithm", algo.description))
        for p in algo.params:
            out.append(_entry(f"{algo.name}.{p['name']}", p["label"], "setting", p.get("help") or p["label"], group=algo.name,
                              default=p["default"], min=p["min"], max=p["max"], name=p["name"]))
    for v in visions.VISIONS.values():
        out.append(_entry(f"vision.{v.name}", v.label, "vision", v.description, name=v.name))
        for p in v.params:
            out.append(_entry(f"vision.{v.name}.{p['name']}", p["label"], "setting", p.get("help") or p["label"],
                              group=f"vision.{v.name}", default=p["default"], min=p["min"], max=p["max"], name=p["name"]))
    for f in rewards.catalog()["features"]:
        out.append(_entry(f"reward.{f['name']}", f["label"], "reward", f["description"], default=f["default"],
                          when=f["when"], name=f["name"],
                          related=[f"reward.{o}" for o in f["incompatible"]] or None,
                          incompatible=[f"reward.{o}" for o in f["incompatible"]]))
    for name, preset in rewards.PRESETS.items():
        out.append(_entry(f"preset.{name}", f"Preset “{name}”", "preset", rewards.describe(preset), name=name))
    for id, term, short, long, aliases, related in CONCEPTS:
        out.append(_entry(id, term, "concept", short, long=long, aliases=aliases, related=related))
    for e in out:  # registry names always point to their entry (e.g. "gamma", "rays", "survival")
        if e.get("name") and e["name"] not in e["aliases"] and len(e["name"]) > 2 and e["name"] not in GENERIC:
            e["aliases"] = [e["name"], *e["aliases"]]
    ids = {e["id"] for e in out}
    for e in out:
        e["related"] = [r for r in e["related"] if r in ids]
    return out


if __name__ == "__main__":
    for e in entries():
        print(f"{e['kind']:9s} {e['id']:28s} {e['term']}: {e['short'][:70]}")
