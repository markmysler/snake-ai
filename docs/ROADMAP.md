# Roadmap

Plans for finishing the lab before publishing it, then for what comes after. Each plan says why it matters for
learners, how it fits the existing code, what to build, how to check it, and what needs deciding.

Effort: **S** about a day, **M** a few days, **L** a week or more (much of it training time).

**Suggested order:** 3a → 1 → 2 → 5 → publish → 4 → 3b → nice-to-haves. The curriculum (2) should be trained after the
GA curve fix (3a), so its charts are right. The README visuals (5) should be shot after the overlay (1) exists.

## Before publishing

- **Continuous integration.** GitHub Actions running `pytest` on Ubuntu, Windows and macOS with Python 3.10 and 3.13,
  plus a Docker build. The tests already pass on Windows and in Linux containers (3.10 and 3.12).
- **Lessons.** Plan 2: turn the lessons into a curated course.

---

## 1. "What the snake sees" overlay · M–L

**Why.** Learners only see a snake moving. Drawing what the network gets as input, and how sure it is of each move,
turns the black box into something you can reason about. For example: "the rays never saw that wall", or "the
straight move scored 0.02 room, so it turned".

**Design.**
- **Inspect mode** on any board. Pause, then step forward or back one move. A side panel shows:
  - the inputs, drawn on the board;
  - the network's preference for each move, as three bars with the chosen move marked;
  - the raw input numbers, as a table view.
- **Each vision explains itself.** `Vision` gets an optional `explain(env, obs)` that returns drawing primitives in
  board coordinates: `cells` (shaded squares), `lines` (from, to, colour), `marks` (point and label) and `text`.
  The browser only knows how to draw primitives, so a new vision becomes inspectable by writing `explain`:
  - short: three danger arrows and an arrow toward the apple;
  - rays: 8 lines, with markers where they meet a wall, the body or the apple;
  - window: the square it covers;
  - room: each move's reachable pocket shaded, with its share;
  - full: a note that it sees the whole board.
- **Network outputs.**
  - GA: `MLPPolicy.scores(state)`, the 3 output values shown as a softmax.
  - DQN: `QNet(state)`, the Q-values.
  - Normalised to "preference" for display, with the raw values in the tooltip.
- **Data size.** Replays already carry their seed and are deterministic. Small visions (≤ 30 inputs) send inputs and
  outputs with every move. For window and full, the browser asks `GET /api/frame?run&agent&version&seed&step` for one
  move at a time. That replays the game on the server, cached per (game, policy).

**Files.**
- `vision.py`: an `explain` for each vision.
- `policies.py` and `dqn_model.py`: scores and Q-values.
- `dashboard/replay.py`: record per-move inputs and outputs.
- `dashboard/server.py`: `/api/frame`.
- `dashboard/static/js/components/board.js`: inspect mode, primitive renderer, side panel.

**Checks.**
- Replaying a game from its seed reproduces every move.
- `explain` returns valid, in-board primitives for every vision, on random positions.
- Small-vision payloads stay under about 200 KB per game.
- Playwright: step through a game and compare the panel with the server's numbers.

**Decide.**
- Bar style for move preferences: softmax, or ranks only?
- Should inspect mode also exist in Compare (side-by-side boards)?

## 2. A curated curriculum · L (mostly training time)

**Why.** Today the lessons are a lab notebook: 01–02 are written up, and 03–09 are AI-analysed one-change experiments
whose human conclusions are still empty. A learner needs a path where each step answers one question, uses 3 seeds, has a written conclusion and ends
with an exercise.

**Proposed course.** All lessons use a 10×10 board, 10 minutes per run and 3 seeds. Existing runs that fit are reused
as pilots, but every lesson is re-run as a seed group.

| # | Question | The one change | Builds on |
|---|---|---|---|
| 01 | How do evolution and Q-learning learn the same game? | baseline: short vision, `score` reward | lesson 01 |
| 02 | You get what you reward | survival bonus vs none | 03, 07 |
| 03 | Can hints speed learning up safely? | none vs distance vs potential shaping (DQN) | new |
| 04 | Does seeing further help? | short vs rays (depth 4, 9) | 04 |
| 05 | Is more information always better? | rays vs full board | 02 |
| 06 | Knowing how much room is left | short vs room | 1.5-min pilot runs (DQN 35–36) |
| 07 | Luck or skill? | 1 recipe × 5 seeds, then a "fake improvement" | new |
| 08 | How far ahead to plan | DQN gamma 0.95 vs 0.99 | 06 |
| 09 | Tuning evolution | games per genome, mutation rate | 09 |

**Each lesson's `LESSON.md`:**
- Question, Change and Prediction;
- Results (generated, including seed spread and how games ended);
- What we learned: drafted by the AI analysis of the group, then edited by a person;
- **Try this**: 1–2 exercises with a one-click recipe;
- glossary links.

**Tooling.**
- `course.json`: the ordered list of lessons, each with its title, question and recipes.
- `python lessons.py course run [n]` queues a lesson's seed groups, so the course can be rebuilt overnight.
- Runs tab: course lessons first, then "Playground" (the user's own lessons). Each `LESSON.md` gets "Previous / Next lesson" links.

**Compute.** 9 lessons × about 2 recipes × 3 seeds × 10 min ≈ 9 hours. Run it overnight, 2 at a time with "runs at a
time" set to 2. Each lesson's runs share the same setting, so wall-clock comparisons stay fair within a lesson.

**Checks.** Every course lesson has a "What we learned" written by a person, no `_Fill this in_` left, and passes "one
setting differs" in Compare.

**Decide.**
- Keep 03–09 as "notebook history" next to the course, or replace them?

## 3. Honest comparisons

### 3a. Fix the GA training curve · S–M

**Problem.** The GA's "mean of last 100 games" and the "how training games ended" chart come from the last ~33
networks of each generation. That is always the mutated **children**, never the elites. So the GA looks worse and
starves more than the networks it keeps (run 09/143223: 50% starved in training, 0% in testing).

**Fix.**
- The main GA numbers come from the **elites carried into the next generation, playing their fresh games**.
  - These are the first `n_elite` networks of each new population.
  - They were selected on last generation's boards and now play new ones, so their scores are an unbiased estimate of
    "how good is what we kept".
  - Scoring elites on the same games they were selected on would overstate them (the winner's curse). That is worth a
    glossary entry.
- The whole population stays visible as a secondary series: CSV columns `pop_mean_score` and `pop_end_*`, drawn as a
  thin dashed line labelled "this generation's experiments".

**Files.**
- `train_ga.py`: two rolling windows.
- `publisher.py` and `dashboard/store.py`: columns appended, so old rows still load.
- `linechart.js`: optional secondary series.
- `endchart.js`: an elites / population toggle.
- `analysis.py`: tell the model which metric a run has.

**Old runs.** Their CSVs lack the new columns, so the chart says "this run's curve shows the population's
experiments" and draws them dashed.

**Checks.**
- In a 3-generation test, the elite mean is computed only from `results[:n_elite]`.
- Old CSVs still load.
- A run in which elites never change shows a steady elite curve.

### 3b. Equal-compute comparisons (ideas 2 and 13) · M

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

## 4. Play it yourself · M (after 1)

**Why.** Playing with only the 11 short-vision hints shows, within a minute, why short vision traps itself. It turns a
reported fact into felt experience.

**Design.**
- A **Play** view.
- Pick a vision and a seed, then play with the arrow keys (mapped to the relative actions), turn-based or real time.
- A **"See only what the snake sees"** toggle hides the board and draws only that vision's `explain` primitives
  (from plan 1).
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

## 5. README visuals and a first-15-minutes walkthrough · S

**Why.** Visitors decide from the first screen. The README is complete but text-only.

**Build.**
- `tools/screenshots.py`: a Playwright script that starts the dashboard on the curated lessons and captures Runs,
  Watch, Compare (with the verdict), Best, Learn and Ideas, in light and dark, into `docs/img/`. It is re-runnable, so
  the images stay current.
- One short GIF of boards playing: frames captured by Playwright and assembled with Pillow. Pillow becomes a dev-only
  dependency.
- The README opens with the hero image and a **First 15 minutes** walkthrough:
  1. start the lab;
  2. open lesson 01 and watch;
  3. press Run again, changing vision to room, for 3 minutes;
  4. compare the runs and read the verdict;
  5. optionally, analyse with AI.

**Checks.** Images total under 2 MB. The walkthrough is followed on a fresh clone in Docker.

---

## Nice to have after publishing

### jev as a third player · M–L

**What.** An `algorithm` entry `jev` that plays through TokenRouter's decisions API (`typesafe/jev-1.13`):
- the request carries the `state` (from the vision's `explain`, plan 1) and one `choice` question whose criteria are
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

**Check.** A lesson comparing full + MLP with full + CNN at equal steps (needs 3b).

### Read-only demo mode · S–M

**What.** `python -m dashboard.server --read-only`: the server refuses launch, delete, edit and analysis; the browser
hides those buttons. It allows a public demo of the curated course, for example a Docker image on a small server or a
Hugging Face Space.

A fully static site (no Python) would need the policies ported to the browser (ONNX or JS), which is out of scope.

**Checks.** Every changing endpoint returns 403 in read-only mode, and a UI test finds no launch, delete or edit buttons.
