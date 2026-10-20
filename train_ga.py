"""Genetic-algorithm trainer: evolves flat-vector MLP policies, numpy-only inference.

    python train_ga.py --workers 8 --generations 50 --vision rays:depth=4 --reward score

Fitness = a genome's average total reward per game under --reward (rewards.py). The default
preset "ga_classic" (apples x 100 + moves x 0.1) is the fitness used in lessons 01-02.
"""
import argparse
import os
import signal
from collections import deque
from multiprocessing import Pool

import numpy as np

import rewards
import vision as visions
from policies import GA_HIDDEN, MLPPolicy, mlp_param_count
from publisher import Publisher, atomic_write, write_json
from snake_env import SnakeEnv, end_code, end_counts

_env = None  # one env per worker process


def _init_worker(width, height, vision, reward):
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent handles Ctrl+C
    global _env
    _env = SnakeEnv(width, height, reward=reward, vision=vision)


def _play(task):
    """Play one genome's games. Returns (scores, steps, total rewards, how each game ended)."""
    genome, seeds = task
    act = MLPPolicy(genome).act
    scores, steps, returns, ends = [], [], [], []
    for seed in seeds:
        state, done, total = _env.reset(seed=seed), False, 0.0
        while not done:
            state, r, done, info = _env.step(act(state))
            total += r
        scores.append(info["score"])
        steps.append(info["steps"])
        returns.append(total)
        ends.append(end_code(info))
    return scores, steps, returns, ends


def game_seeds(base_seed, gen, idx, n):
    return [int(s) for s in np.random.SeedSequence([base_seed, gen, idx]).generate_state(n)]


def next_generation(pop, fit, rng, elite_frac, k, rate, sigma):
    """Elites survive unchanged; the rest come from tournament + uniform crossover + mutation."""
    n, n_params = pop.shape
    order = np.argsort(-fit)
    n_elite = max(1, int(round(n * elite_frac)))
    n_child = n - n_elite

    def tournament():
        idx = rng.integers(0, n, size=(n_child, k))
        return idx[np.arange(n_child), np.argmax(fit[idx], axis=1)]

    p1, p2 = tournament(), tournament()
    children = np.where(rng.random((n_child, n_params)) < 0.5, pop[p1], pop[p2])
    mutate = rng.random(children.shape) < rate
    children = children + mutate * rng.normal(0.0, sigma, children.shape)
    return np.concatenate([pop[order[:n_elite]], children]).astype(np.float32)


def save_npy(path, arr):
    return atomic_write(path, lambda f: np.save(f, arr))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--agent", default="ga", help="name used for the files this trainer writes")
    p.add_argument("--vision", default="short", help="what the snake sees, e.g. full, rays:depth=4")
    p.add_argument("--reward", default="ga_classic", help="preset or features, e.g. apple=10,death=10")
    p.add_argument("--population", type=int, default=200)
    p.add_argument("--games", type=int, default=3, help="games per genome per generation")
    p.add_argument("--elite-frac", type=float, default=0.10)
    p.add_argument("--tournament", type=int, default=3)
    p.add_argument("--mutation-rate", type=float, default=0.1)
    p.add_argument("--mutation-sigma", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--generations", type=int, default=0, help="0 = until stopped")
    p.add_argument("--time-budget", type=float, default=0, help="minutes, 0 = none")
    p.add_argument("--width", type=int, default=10)
    p.add_argument("--height", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--live-dir", default="live")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--models-dir", default="models")
    return p.parse_args()


def main():
    args = parse_args()
    vision, reward = visions.parse(args.vision), rewards.parse(args.reward)
    rng = np.random.default_rng(args.seed)
    n_params = mlp_param_count(visions.state_size(vision, args.width, args.height))
    pop = rng.normal(0.0, 0.5, (args.population, n_params)).astype(np.float32)
    pub = Publisher(args.agent, "generation", args.live_dir, args.results_dir)
    os.makedirs(args.models_dir, exist_ok=True)
    grid = {"width": args.width, "height": args.height, "vision": vision, "reward": reward}
    total_steps = total_games = best_score = 0
    best_fit, recent, recent_ends = -np.inf, deque(maxlen=100), deque(maxlen=100)
    print(f"[{args.agent}] vision={visions.to_string(vision)} reward={rewards.to_string(reward)} "
          f"population={args.population} params={n_params} workers={args.workers}", flush=True)

    pool = Pool(args.workers, initializer=_init_worker,
                initargs=(args.width, args.height, vision, reward))
    try:
        gen = 0
        while True:
            gen += 1
            tasks = [(pop[i], game_seeds(args.seed, gen, i, args.games)) for i in range(len(pop))]
            chunk = max(1, len(tasks) // (args.workers * 4))
            results = pool.map(_play, tasks, chunksize=chunk)

            fit = np.array([float(np.mean(ret)) for _, _, ret, _ in results])
            for sc, st, _, en in results:
                total_steps += sum(st)
                total_games += len(sc)
                recent.extend(sc)
                recent_ends.extend(en)
                best_score = max(best_score, max(sc))
            order = np.argsort(-fit)

            if fit[order[0]] > best_fit:
                best_fit = float(fit[order[0]])
                genome = pop[order[0]].copy()
                info = {"generation": gen, "fitness": best_fit, "hidden": GA_HIDDEN,
                        "mean_score": float(np.mean(results[order[0]][0])), **grid}
                save_npy(os.path.join(args.models_dir, f"{args.agent}_best.npy"), genome)
                write_json(os.path.join(args.models_dir, f"{args.agent}_best.json"), info)
                pub.weights(lambda f: np.save(f, genome), "npy", gen, **info)

            mean_scores = [float(np.mean(results[i][0])) for i in order]
            if save_npy(pub.path(f"{args.agent}_population.npy"), pop[order]):
                write_json(pub.path(f"{args.agent}_population.json"), {
                    "version": f"{pub.run_id}:{gen}", "generation": gen,
                    "fitness": [round(float(fit[i]), 2) for i in order],
                    "mean_score": mean_scores, **grid})

            rec = pub.stats(gen, total_steps, total_games, best_score, np.mean(recent), ends=end_counts(recent_ends),
                            best_fitness=round(best_fit, 2), gen_best_fitness=round(float(fit[order[0]]), 2),
                            gen_mean_score=round(float(np.mean(mean_scores)), 3))
            print(f"[{args.agent}] gen {gen:4d} | fit best {fit[order[0]]:8.2f} all-time {best_fit:8.2f} | "
                  f"best score {best_score:3d} | mean100 {rec['mean_score_100']:6.2f} | "
                  f"{rec['steps_per_sec']:9,.0f} steps/s | {rec['elapsed_s']:7.1f}s", flush=True)

            out_of_time = args.time_budget and pub.elapsed() >= args.time_budget * 60
            if (args.generations and gen >= args.generations) or out_of_time or pub.stop_requested():
                break
            pop = next_generation(pop, fit, rng, args.elite_frac, args.tournament,
                                  args.mutation_rate, args.mutation_sigma)
        pool.close()
    except KeyboardInterrupt:
        print(f"[{args.agent}] interrupted", flush=True)
        pool.terminate()
    finally:
        pool.join()
        pub.finish()
    print(f"[{args.agent}] done. best fitness {best_fit:.2f}, best score {best_score}", flush=True)


if __name__ == "__main__":
    main()
