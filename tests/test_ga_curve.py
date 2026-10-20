"""The GA's learning curve follows the networks it keeps (elites on fresh games), not its mutated children."""
import csv
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from dashboard.store import HISTORY_KEYS, history_row  # noqa: E402
from train_ga import elite_count, kept_results, next_generation  # noqa: E402


def test_elites_come_first_in_the_next_generation():
    rng = np.random.default_rng(0)
    pop = rng.normal(size=(20, 5)).astype(np.float32)
    fit = rng.normal(size=20)
    nxt = next_generation(pop, fit, rng, 0.1, 3, 0.1, 0.2)
    n = elite_count(20, 0.1)
    assert np.array_equal(nxt[:n], pop[np.argsort(-fit)[:n]])


def test_curve_uses_only_the_elites_after_the_first_generation():
    results = [([i], [10], [0.0], ["w"]) for i in range(10)]
    assert kept_results(results, 1, 2) == results  # nothing kept yet: the starting population
    assert kept_results(results, 2, 2) == results[:2]
    assert kept_results(results, 3, 2) == results[:2]


def test_three_generations_record_kept_and_population_curves(tmp_path):
    subprocess.run([sys.executable, "train_ga.py", "--generations", "3", "--population", "20", "--workers", "1",
                    "--elite-frac", "0.25", "--live-dir", str(tmp_path / "live"), "--results-dir", str(tmp_path / "results"),
                    "--models-dir", str(tmp_path / "models")], cwd=ROOT, check=True, capture_output=True, timeout=300)
    with open(tmp_path / "results" / "ga.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    for r in rows:
        pop_ends = [int(r[f"pop_end_{k}"]) for k in ("wall", "body", "starved", "won")]
        assert sum(pop_ends) == 20 * 3  # the whole generation: 20 genomes x 3 games
        ends = [int(r[f"end_{k}"]) for k in ("wall", "body", "starved", "won")]
        assert 0 < sum(ends) <= 100
    # generation 1 counts all 60 games; generations 2-3 add only the 5 elites' 15 games each
    assert [sum(int(r[f"end_{k}"]) for k in ("wall", "body", "starved", "won")) for r in rows] == [60, 75, 90]
    assert all(history_row(r)[11] is not None for r in rows)


def test_old_csv_rows_still_load():
    old = {k: "1" for k in HISTORY_KEYS if not k.startswith("pop_")}
    row = history_row(old)
    assert len(row) == len(HISTORY_KEYS) and row[11:] == [None] * 5
