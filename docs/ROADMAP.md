# Roadmap

Plans for what comes next. Each plan says why it matters for learners, how it fits the existing code, what to build, how to
check it, and what needs deciding.

Effort: **S** about a day, **M** a few days, **L** a week or more (much of it training time).

## Before publishing

Built (October 2026):
- **The GA's training curve follows the networks it keeps** (was plan 3a). The main curve and endings come from the elites
  carried into each generation, playing its fresh games; the whole generation is recorded as `pop_mean_score` and `pop_end_*`
  and drawn dotted, with a *Kept networks / Whole generation* switch on the endings chart. Old runs are flagged and drawn
  dashed. Glossary: *Kept networks*, *Winner's curse*.
- **Inspect mode, "what the snake sees"** (was plan 1). Every vision has an `explain` (drawing primitives), and input names
  or a picture layout; `GET /api/frame` rebuilds any move of a replay from its seed and model version (`dashboard/inspector.py`).
  Decided: move preferences are shown as a softmax with the raw values beside them; inspect lives in Watch only (Compare
  links to Watch). Frames are fetched one move at a time for every vision (1–5 KB each), rather than sending small visions'
  frames in one payload: simpler, and stepping is instant locally.
- **The course tooling** (was plan 2): `lessons/course.json`, `course.py` (`list`, `run`, `check`, `write`; also
  `python lessons.py course ...`), the generated `LESSON.md` blocks with previous / next links and *Try this*, the course
  first in Runs (Train this lesson, Compare, exercises), and a guard so a queued run is only ever started once. Decided: the
  old lessons 01–09 stay as the **Notebook**, listed after the course; several are named as pilots in the course lessons.
- **README visuals** (was plan 5): `tools/screenshots.py` shoots every view in light and dark (`docs/img/*.webp`, shown
  with `<picture>`) and a GIF of the boards; the README opens with the GIF and a *First 15 minutes* walkthrough.
- **Continuous integration**: `.github/workflows/ci.yml` runs the tests on Ubuntu, Windows and macOS with Python 3.10 and
  3.13, and builds the Docker image and runs the tests in it.

Still to do:
1. **Train the course** (about 8 hours one at a time, 4 with `--parallel 2`; mains power, not battery):
   `python course.py run all --wait --parallel 2`. With an API key set, each seed group gets an AI analysis as a draft.
2. **Write "What we learned"** for each of the nine lessons from the results and the drafts, then `python course.py check`
   until it reports nothing.
3. **Re-shoot the README images** from the trained course: `python tools/screenshots.py` (it then shows course lesson 06
   against 01 in Compare).
4. Push to the empty GitHub repository and check that the CI run is green.

---

## Next: equal-compute comparisons (ideas 2 and 13) · M

**Problem.** In one run the GA gets about 11 cores and DQN gets 1. A 10-minute race partly measures hardware.

**Design.**
- A recipe field `budget`, which is one of:
  - `{"type": "time", "minutes": 10}` (today's behaviour, the default);
  - `{"type": "steps", "steps": 2_000_000}`;
  - `{"type": "equal_cores", "cores": 4, "minutes": 10}`: every agent gets the same cores.
- Trainers get `--max-steps`.
- The planner gives each agent the same cores in `equal_cores` mode.
- Charts get an x-axis option "core-minutes" (elapsed × cores of that agent).
- Compare shows "score per core-minute".

**Files.**
- `runconfig.py`: the budget, normalisation and validation.
- `system.plan`.
- `train_*.py`.
- `experiment.py`: the stop rule.
- `linechart.js`: `X_MODES`.
- `compare.js`.
- New run form: a "Budget" control.

**Checks.**
- A steps budget stops both trainers within one stats interval of N.
- With equal cores, the plan gives every agent equal cores.
- Old recipes normalise to `time`.

## Next: play it yourself · M

**Why.** Playing with only the 11 short-vision hints shows, within a minute, why short vision traps itself. It turns a
reported fact into felt experience.

**Design.**
- A **Play** view.
- Pick a vision and a seed, then play with the arrow keys (mapped to the relative actions), turn-based or real time.
- A **"See only what the snake sees"** toggle hides the board and draws only that vision's `explain` primitives
  (from Inspect mode).
- After a game, show "Your score vs the GA and DQN on this same seed". The test seeds are fixed, so the comparison is
  exact.

**Engine.**
- The Python environment stays the only implementation (no JavaScript copy to drift).
- `POST /api/play/start` takes a vision and seed and returns the state and primitives.
- `POST /api/play/step` takes an action and returns the next state.
- Sessions are held in memory with a cap and expire after 10 minutes.
- Local latency is fine for turn-based play and for real time at 1× speed (8 moves per second).

**Files.** `dashboard/play.py` (sessions), routes in `server.py`, `views/play.js`, a tab.

**Checks.**
- Replaying a session's actions in the environment gives the same score.
- Expired sessions return 410.
- Playwright plays a scripted game.

**Decide.** Keep human scores only in the browser, or allow saving them to a lesson as a "human" agent for comparison?

---

## Nice to have after publishing

### jev as a third player · M–L

**What.** An `algorithm` entry `jev` that plays through TokenRouter's decisions API (`typesafe/jev-1.13`):
- the request carries the `state` (from the vision's `explain`, see Inspect mode) and one `choice` question whose criteria are
  the three moves;
- its "training" is editing the instructions and criteria;
- it is evaluated on the same fixed test games.

**Cost and speed.** About $0.000016 and about 1 s per decision. A 100-game evaluation is about 30,000 decisions (about
$0.50). Done one at a time that would take more than 8 hours, so batch boards: `questions` is a map, so many boards fit
in one call (the size limit needs measuring).

**Design.**
- `train_jev.py` with a game budget instead of a time budget.
- A cost estimate and confirmation in New run.
- Evaluation on 20 games, flagged as fewer than the usual 100.

**Decide.** Spending caps, and whether to evaluate on fewer games.

### Convolutional network for full-board vision (idea 14) · M

**What.** DQN with a small CNN over the picture-based visions (window, full) instead of an MLP over flattened pixels.

**Design.**
- Visions declare a `shape` (channels, side).
- `dqn_model.QNet` gets a conv branch, chosen with `--net conv` (a new DQN setting, default `mlp`).
- The GA stays on MLPs: a CNN has too many weights for evolution at this scale.

**Check.** A lesson comparing full + MLP with full + CNN at equal steps (needs equal-compute comparisons).

### Read-only demo mode · S–M

**What.** `python -m dashboard.server --read-only`: the server refuses launch, delete, edit and analysis; the browser
hides those buttons. It allows a public demo of the curated course, for example a Docker image on a small server or a
Hugging Face Space.

A fully static site (no Python) would need the policies ported to the browser (ONNX or JS), which is out of scope.

**Checks.** Every changing endpoint returns 403 in read-only mode, and a UI test finds no launch, delete or edit buttons.
