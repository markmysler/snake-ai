// Watch: one run. Headline numbers per agent, its snakes playing, its learning curves.
import { Board } from "../components/board.js";
import { EndChart, hasEndings, hasPopEndings } from "../components/endchart.js";
import { info, termsRow } from "../components/glossary.js";
import { groupMembers, outcomeBar, outcomeLegend, spreadOf, stripPlot } from "../components/spread.js";
import { hasPop, isOldGaCurve, LineChart, METRICS, X_MODES } from "../components/linechart.js";
import { Inspector } from "../components/inspector.js";
import { addBoards, playbackControls, removeBoards } from "../components/playback.js";
import { Population } from "../components/population.js";
import { agentLabel, api, fmtCompact, fmtDate, fmtDuration, fmtMB, fmtNum, h, markdown, q, runTitle, seriesColor,
         seriesVar, statusPill, toast } from "../util.js";

const view = { x: "time", metric: "mean", logX: false, endPop: {} }; // remembered while the page is open
const KEPT_NOTE = "The GA's curve follows the networks it keeps (its elites) playing each new generation's fresh games.";
const OLD_GA_NOTE = "Recorded before the GA curve fix: the dashed GA line is its last mutated children, not the networks it "
  + "kept, so it looks worse than the test score.";

export function mount(main, runId, app) {
  if (!runId) {  // default: the newest training run, else the newest run
    const pick = [...app.runs].reverse().find((r) => r.status === "running") || app.runs.at(-1);
    if (pick) { app.go(`#watch/${pick.id}`); return { ownsConnection: true }; }
    main.append(h("div", { class: "card empty" }, h("strong", {}, "Nothing to watch yet"),
      "Start a run and its snakes will appear here while they learn.",
      h("button", { class: "primary", onclick: () => app.go("#new") }, "New run")));
    return {};
  }
  const root = h("div", { class: "view" });
  main.append(root);
  let data = null, chart = null, boards = [], population = null, endCharts = [], inspector = null;
  const ai = { key: null, data: null };  // the full analysis, fetched when the run's analysis state changes
  app.connect(runId);

  function picker() {
    const groups = {};
    for (const r of app.runs) (groups[r.lesson] ||= []).push(r);
    return h("select", { "aria-label": "Run to watch", onchange: (e) => app.go(`#watch/${e.target.value}`) },
      Object.entries(groups).map(([lesson, runs]) => h("optgroup", { label: lesson },
        [...runs].reverse().map((r) => h("option", { value: r.id, selected: r.id === runId },
          `${runTitle(r)} · ${r.status}`)))));
  }

  function header(run) {
    const elapsed = run.status === "running" && run.started ? (Date.now() - Date.parse(run.started.replace(" ", "T"))) / 1000
      : Math.max(0, ...Object.values(run.final).map((f) => f.elapsed_s));
    const pct = run.budget_min ? Math.min(100, (elapsed / 60 / run.budget_min) * 100) : 0;
    const inCompare = app.compareIds.includes(run.id);
    return h("div", { class: "card" },
      h("div", { class: "page-head" },
        h("div", {}, h("div", { class: "btn-row" }, h("h1", {}, runTitle(run)), statusPill(run.status)),
          h("p", { class: "lead" }, `${run.lesson_title} · ${run.id}`)),
        h("div", { class: "btn-row" }, picker(),
          run.status === "running" ? h("button", { onclick: () => api.post(`/api/run/stop?id=${q(run.id)}`)
            .then(() => toast("Stopping… the run will be evaluated and saved.")) }, "Stop & save") : null,
          h("button", { onclick: () => app.go(`#new/${run.id}`) }, "Run again"),
          h("button", { "aria-pressed": String(inCompare), disabled: !inCompare && app.compareIds.length >= app.maxCompare,
            onclick: () => { app.toggleCompare(run.id); render(); } }, inCompare ? "In comparison" : "Add to comparison"))),
      h("div", { class: "chips" },
        h("span", { class: "chip" }, "vision ", h("b", {}, run.vision.spec), ` · ${run.vision.inputs.toLocaleString()} inputs`),
        h("span", { class: "chip" }, "reward ", h("b", {}, run.reward)),
        h("span", { class: "chip" }, h("b", {}, run.budget_min ? `${run.budget_min} min` : "no time limit")),
        h("span", { class: "chip" }, `grid ${run.config.width}×${run.config.height} · seed ${run.config.seed}`),
        run.machine.cpus ? h("span", { class: "chip" }, `${run.machine.cpus} CPUs${run.machine.on_battery ? " · on battery" : ""}`) : null,
        run.auto_analyze && ["queued", "running"].includes(run.status) ? h("span", { class: "chip" }, "AI analysis when it ends") : null,
        run.parent?.run ? h("a", { class: "chip", href: `#watch/${run.parent.run}`, title: "This run was recommended by the AI analysis of another run" },
          "recommended by ", h("b", {}, run.parent.run))
          : run.parent ? h("span", { class: "chip", title: run.parent.title }, "recommended by a run that was deleted") : null),
      run.note ? h("p", { class: "ink2" }, run.note) : null,
      run.budget_min && run.status !== "queued" ? h("div", { class: "meter-row" }, h("span", {}, "Time used"),
        h("div", { class: "meter" }, h("i", { style: { width: `${pct}%` } })),
        h("span", {}, `${fmtDuration(elapsed)} / ${fmtDuration(run.budget_min * 60)}`)) : null);
  }

  // One stat tile per agent: the evaluation once there is one, otherwise the live training mean.
  function kpis(run) {
    const perAgent = run.reward === "per agent";
    const anyOutcomes = run.agents.some((a) => run.evaluation[a.id]?.outcomes);
    return h("div", { class: "kpis-wrap" }, h("div", { class: "kpis" }, run.agents.map((a, i) => {
      const last = data.history[a.id]?.at(-1), ev = run.evaluation[a.id], live = data.stats[a.id];
      const hero = ev ? ev.mean : last ? last[4] : null;
      return h("div", { class: "stat", "--series": seriesVar(i) },
        h("div", { class: "label" }, h("span", { class: "swatch" }), agentLabel(a, perAgent)),
        h("div", { class: "hero" }, hero === null ? "–" : fmtNum(hero, 1),
          h("small", {}, ev ? "apples · test (100 games)" : a.algorithm === "ga" && hasPop(data.history[a.id])
            ? "apples · kept networks, last 100 games" : "apples · mean of last 100")),
        h("div", { class: "sub" },
          ev ? h("span", {}, "median ", h("b", {}, fmtNum(ev.median, 1)), " · max ", h("b", {}, ev.max)) : null,
          last ? h("span", {}, "best game ", h("b", {}, last[3])) : null,
          last ? h("span", {}, `${live?.iteration_label || (a.algorithm === "ga" ? "generation" : "episode")} `,
            h("b", {}, fmtCompact(last[6]))) : null,
          last ? h("span", {}, h("b", {}, fmtCompact(last[5])), " steps/s") : null),
        ev?.outcomes ? h("div", { class: "sub" }, h("span", {}, "test games: ", h("b", {}, fmtNum(ev.mean_steps, 0)), " moves each · ",
          h("b", {}, fmtNum(ev.steps_per_apple, 0)), " per apple")) : null,
        outcomeBar(ev),
        !last && run.status !== "finished" ? h("div", { class: "sub" }, "warming up…") : null);
    })), anyOutcomes ? h("div", { class: "btn-row small muted" }, "How the 100 test games ended:", outcomeLegend(), info("c.game-end")) : null);
  }

  // A run trained as one of a seed group: where its test score sits among the same recipe's other seeds.
  function seedCard(run) {
    if (!run.group) return null;
    const members = groupMembers(run, app.runs);
    const perAgent = run.reward === "per agent";
    const rows = run.agents.map((a, i) => ({ label: agentLabel(a, perAgent), color: seriesVar(i), spread: spreadOf(members, a.id), highlight: run.id }));
    const pending = members.filter((m) => ["queued", "running"].includes(m.status)).length;
    return h("section", { class: "card", id: "seeds" },
      h("div", { class: "section-head" }, h("div", {}, h("h2", {}, "Seed spread ", info("c.seed-spread")),
        h("p", {}, `This is seed ${run.group.index + 1} of ${run.group.size}: the same recipe trained with seeds ${run.group.base_seed}–`
          + `${run.group.base_seed + run.group.size - 1}. Each dot is one seed's test score; the bar is the range; the line is the mean.`
          + (pending ? ` ${pending} still to finish.` : ""))),
        h("div", { class: "btn-row" }, members.map((m) => h("a", { class: "chip" + (m.id === run.id ? " current" : ""), href: `#watch/${m.id}` },
          `seed ${m.config.seed}`, " ", statusPill(m.status))))),
      stripPlot(rows));
  }

  function gamesCard(run) {
    removeBoards(boards);
    boards = run.agents.map((a, i) => new Board({ run: run.id, agent: a.id, title: agentLabel(a, run.reward === "per agent"),
      color: seriesVar(i), flashNewBest: run.status === "running", onInspect: (b) => inspect(b),
      waitingText: a.algorithm === "ga" ? "Waiting for the first generation…" : "Waiting for the first evaluation…" }));
    addBoards(boards);
    return h("section", { class: "card", id: "games" },
      h("div", { class: "section-head" }, h("div", {}, h("h2", {}, "The best snakes playing"),
        h("p", {}, (run.status === "running" ? "Each board plays its agent's best model so far and switches to newer ones "
          + "between games. Watching never slows training down." : "The saved best model of each agent.")
          + " Inspect steps through a game move by move, showing what the snake sees and what its network thinks.")),
        playbackControls()),
      h("div", { class: "boards" }, boards.map((b) => b.el)));
  }

  function inspect(board) {
    inspector?.dispose();
    const g = board.game;
    inspector = new Inspector({ run: data.run.id, agent: board.o.agent, title: board.o.title, color: board.o.color, seed: g.seed,
      version: board.meta.version, step: Math.min(board.frame, g.steps - 1),
      onClose: () => { inspector.el.remove(); inspector = null; } });
    root.querySelector(".inspector")?.remove();
    root.querySelector("#games").after(inspector.el);
    inspector.el.focus({ preventScroll: true });
    inspector.el.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  function progressCard(run) {
    const seg = (options, key) => h("div", { class: "segmented", role: "group" }, Object.entries(options).map(([k, o]) =>
      h("button", { "aria-pressed": String(view[key] === k), onclick: () => { view[key] = k; root.querySelector("#progress")
        .replaceWith(progressCard(run)); } }, o.short || o.title.split(",")[0])));
    const canvas = h("canvas");
    const popShown = view.metric === "mean" && run.agents.some((a) => a.algorithm === "ga" && hasPop(data.history[a.id]));
    const oldGa = run.agents.some((a) => isOldGaCurve(a, data.history[a.id]));
    const legend = h("div", { class: "legend" }, run.agents.map((a, i) =>
      h("span", { "--series": seriesVar(i) }, h("i", { class: "key" + (isOldGaCurve(a, data.history[a.id]) ? " dash" : "") }),
        agentLabel(a, run.reward === "per agent"))),
      popShown ? h("span", { "--series": seriesVar(run.agents.findIndex((a) => a.algorithm === "ga")) }, h("i", { class: "key dots" }),
        "GA's whole generation (mostly mutated children)") : null);
    const card = h("section", { class: "card", id: "progress" },
      h("div", { class: "section-head" }, h("h2", {}, "How it learned"),
        h("p", {}, view.x === "steps" ? "Env steps compare how much experience each method needed (sample efficiency)."
          : view.x === "time" ? "Wall-clock compares practical speed on this machine." : "Games played so far.")),
      h("div", { class: "controls" }, h("div", { class: "ctl", role: "group", "aria-label": "Show" }, "Show", seg({ mean: { title: "Mean score" }, best: { title: "Best score" } }, "metric")),
        h("div", { class: "ctl", role: "group", "aria-label": "Over" }, "Over", seg({ time: { short: "Time" }, steps: { short: "Env steps" }, games: { short: "Games" } }, "x")),
        h("label", {}, h("input", { type: "checkbox", checked: view.logX, onchange: (e) => { view.logX = e.target.checked; drawChart(); } }), "log x")),
      legend, h("div", { class: "chart-box" }, canvas),
      h("p", { class: "small muted" }, METRICS[view.metric].title + " against " + X_MODES[view.x].title.toLowerCase()
        + ". The exact numbers are in the table below.", popShown ? ` ${KEPT_NOTE} ` : " ", info("c.kept-curve")),
      oldGa && view.metric === "mean" ? h("p", { class: "small msg warn" }, OLD_GA_NOTE) : null);
    chart?.destroy();
    chart = new LineChart(canvas);
    queueMicrotask(drawChart);
    return card;
  }

  // How the last 100 training games ended, over training: one small stacked chart per agent.
  function endingsCard(run) {
    endCharts.forEach((c) => c.chart.destroy());
    endCharts = [];
    const agents = run.agents.filter((a) => hasEndings(data.history[a.id]));
    if (!agents.length) {
      return run.status === "running" || !Object.keys(data.history).length ? null
        : h("details", { class: "card", id: "endings" }, h("summary", {}, "How training games ended ",
          h("span", { class: "muted" }, "not recorded for this run (runs from now on record it)")));
    }
    const facets = agents.map((a) => {
      const canvas = h("canvas"), pop = hasPopEndings(data.history[a.id]);
      const toggle = pop ? h("div", { class: "segmented", role: "group", "aria-label": `Games of ${a.id}` },
        [[false, "Kept networks"], [true, "Whole generation"]].map(([v, label]) => h("button", { "aria-pressed": String(!!view.endPop[a.id] === v),
          onclick: () => { view.endPop[a.id] = v; root.querySelector("#endings").replaceWith(endingsCard(run)); } }, label))) : null;
      return { a, canvas, el: h("div", { class: "facet" }, h("div", { class: "facet-head" }, h("h3", {}, agentLabel(a, run.reward === "per agent")), toggle),
        h("div", { class: "chart-box" }, canvas)) };
    });
    queueMicrotask(() => {
      for (const f of facets) {
        const c = new EndChart(f.canvas);
        c.set(data.history[f.a.id], view.x, !!view.endPop[f.a.id]);
        endCharts.push({ chart: c, agent: f.a.id });
      }
    });
    return h("details", { class: "card", id: "endings", open: root.querySelector("#endings")?.open ?? true },
      h("summary", {}, "How training games ended ", h("span", { class: "muted" }, "share of the last 100 training games, over time")),
      h("div", { class: "btn-row small muted" }, outcomeLegend(), info("c.game-end")),
      h("div", { class: "facets" }, facets.map((f) => f.el)),
      h("p", { class: "small muted" }, "Training games include DQN's random exploration moves, and the GA's are played on new boards "
        + "every generation, so they end differently from the saved model's test games (the bars at the top). For the GA, "
        + "“Kept networks” are the elites it carries forward; “Whole generation” adds its mutated children, which crash and "
        + "starve far more. Watch for a shift: e.g. starving rising when a reward pays for wandering, or body crashes taking "
        + "over as the snake grows."),
      agents.some((a) => isOldGaCurve(a, data.history[a.id])) ? h("p", { class: "small msg warn" },
        "Recorded before the GA curve fix: the GA's chart shows its last mutated children, not the networks it kept.") : null);
  }

  function drawChart() {
    for (const c of endCharts) c.chart.set(data?.history[c.agent], view.x, !!view.endPop[c.agent]);
    if (!chart || !data) return;
    chart.set(data.run.agents.map((a, i) => ({ label: agentLabel(a), short: agentLabel(a), color: seriesColor(i),
      pop: a.algorithm === "ga", dash: isOldGaCurve(a, data.history[a.id]), rows: data.history[a.id] || [] })),
      { x: view.x, metric: view.metric, logX: view.logX });
  }

  function detailsCard(run) {
    const rows = run.agents.map((a) => {
      const last = data.history[a.id]?.at(-1), ev = run.evaluation[a.id];
      return h("tr", {}, h("th", {}, agentLabel(a, run.reward === "per agent")),
        h("td", { class: "r" }, last ? fmtCompact(last[6]) : "–"), h("td", { class: "r" }, last ? fmtCompact(last[1]) : "–"),
        h("td", { class: "r" }, last ? fmtCompact(last[2]) : "–"), h("td", { class: "r" }, last ? last[3] : "–"),
        h("td", { class: "r" }, last ? fmtNum(last[4], 2) : "–"),
        h("td", { class: "r" }, last && last[11] !== null && last[11] !== undefined ? fmtNum(last[11], 2) : "–"),
        h("td", { class: "r" }, ev ? `${fmtNum(ev.mean, 2)} / ${fmtNum(ev.median, 1)} / ${ev.max}` : "–"),
        h("td", { class: "r" }, ev ? fmtNum(ev.mean_steps, 0) : "–"),
        h("td", { class: "r" }, ev?.outcomes ? ["wall", "body", "starved", "won"].map((k) => ev.outcomes[k]).join(" / ") : "–"));
    });
    return h("details", { class: "card", id: "numbers", open: root.querySelector("#numbers")?.open },
      h("summary", {}, "Numbers ", h("span", { class: "muted" }, "table view of everything above")),
      h("div", { class: "table-scroll" }, h("table", {},
        h("thead", {}, h("tr", {}, ["Agent", "Generation / episode", "Env steps", "Games", "Best game", "Mean (last 100)", "GA whole generation",
          "Test mean / median / max", "Test moves per game", "Test ends: wall / body / starved / won"].map((t, i) => h("th", { class: i ? "r" : "" }, t)))),
        h("tbody", {}, rows))));
  }

  // ---- AI analysis: what happened, why, and runs to try next (see analysis.py) ----
  const aiKey = (run) => run.analysis ? `${run.analysis.state}|${run.analysis.created}|${run.analysis.failed}` : "none";
  async function loadAnalysis(run, force) {
    const key = aiKey(run);
    if (!force && ai.key === key) return;
    ai.key = key;
    ai.data = run.analysis?.state === "done" ? (await api.get(`/api/analysis?id=${q(run.id)}`).catch(() => ({}))).analysis : null;
    if (data?.run.id === run.id) root.querySelector("#analysis")?.replaceWith(analysisCard(data.run));
  }

  async function analyze(run) {
    try {
      await api.post(`/api/analysis?id=${q(run.id)}`);
      await app.refreshRuns();
    } catch (e) { toast(e.message); }
  }

  async function launchRec(run, i, button) {
    button.disabled = true;
    try {
      const res = await api.post(`/api/analysis/launch?id=${q(run.id)}&index=${i}`);
      toast(res.queue.running.length >= res.queue.concurrency && !res.queue.running.includes(res.id)
        ? "Added to the queue." : "Starting… the first results appear within a few seconds.");
      await app.refreshRuns();
      app.go(`#watch/${res.id}`);
    } catch (e) { toast(e.message); button.disabled = false; }
  }

  function recCard(run, rec, i) {
    const blocked = rec.errors?.length > 0;
    return h("article", { class: "rec" },
      h("div", { class: "rec-head" }, h("span", { class: "n" }, i + 1), h("h3", {}, rec.title),
        i === recommendedIndex ? h("span", { class: "pill finished", title: "Offered as “Run recommended” in Runs" }, "recommended") : null),
      h("p", { class: "ink2" }, rec.why),
      h("div", { class: "chips" }, rec.changes.length ? rec.changes.map((c) => h("span", { class: "chip" }, h("b", {}, c)))
        : h("span", { class: "chip" }, "same recipe (tests run-to-run noise)")),
      rec.prediction ? h("p", { class: "small" }, h("b", {}, "Prediction: "), rec.prediction) : null,
      termsRow([rec.changes.join(" ; "), rec.title, rec.why, rec.prediction], "Learn the terms"),
      h("p", { class: "small muted" }, "Recorded in ", h("b", {}, rec.lesson_target), rec.new_lesson ? " (a new lesson, written for you)" : ""),
      blocked ? h("div", { class: "msg error" }, rec.errors.join(" ")) : null,
      (rec.warnings || []).map((w) => h("div", { class: "msg warn" }, w)),
      h("div", { class: "btn-row" },
        h("button", { class: "primary", disabled: blocked, title: "Queue this run now, exactly as recommended",
          onclick: (e) => launchRec(run, i, e.currentTarget) }, rec.launched?.length ? "Launch again" : "Launch"),
        h("button", { disabled: blocked, title: "Open it in New run to change something first", onclick: () => {
          app.draft = { config: rec.config, origin: { run: run.id, index: i }, title: rec.title, seeds: run.group?.size || 1 };
          app.go("#new/draft");
        } }, "Edit first")),
      rec.launched?.length ? h("div", { class: "chips" }, rec.launched.map((id) =>
        h("a", { class: "chip", href: `#watch/${id}` }, "launched as ", h("b", {}, id)))) : null);
  }

  let recommendedIndex = null;
  function ideaCard(idea) {
    return h("article", { class: "rec idea-card" },
      h("div", { class: "rec-head" }, h("h3", {}, idea.title)),
      h("div", { class: "chips" }, h("span", { class: "chip" }, idea.kind), h("span", { class: "chip" }, `${idea.effort} effort`)),
      idea.what ? h("p", { class: "small" }, idea.what) : null,
      idea.why ? h("p", { class: "small ink2" }, h("b", {}, "Why: "), idea.why) : null,
      idea.note ? h("p", { class: "small muted" }, idea.note) : null,
      h("a", { class: "small", href: `#ideas/${idea.id}` }, `${idea.id} in Ideas →`));
  }

  function analysisCard(run) {
    if (["queued", "running"].includes(run.status)) return null;
    const llm = app.catalog?.llm || {};
    const st = run.analysis?.state, a = st === "done" && ai.key === aiKey(run) ? ai.data : null;
    if (aiKey(run) !== ai.key) queueMicrotask(() => loadAnalysis(run));
    const head = (sub, ...right) => h("div", { class: "section-head" },
      h("div", {}, h("h2", {}, "AI analysis"), sub ? h("p", {}, sub) : null), h("div", { class: "btn-row" }, right));
    const card = h("section", { class: "card analysis", id: "analysis" });
    const add = (...els) => card.append(...els.flat().filter(Boolean));  // (append would print "null")
    if (st === "running") {
      add(head(null), h("p", { class: "ink2 working" }, h("i", { class: "dot" }),
        `${run.analysis.model || "The model"} is reading this run's settings, learning curves and test games… usually under a minute.`));
    } else if (!llm.configured) {
      add(head("Get an explanation of this run and suggested next experiments."),
        h("p", { class: "small muted" }, "Not set up: add TOKENROUTER_BASE_URL and TOKENROUTER_API_KEY to a .env file "
          + "(see .env.example), then restart the dashboard."));
    } else if (!a) {
      add(head(st === "error" ? null : "An LLM explains what happened and why, writes it into the lesson, "
          + "and suggests next runs you can launch with one click.",
        h("button", { class: st === "error" ? "" : "primary", onclick: () => analyze(run) }, st === "error" ? "Try again" : "Analyze this run")),
        st === "error" ? h("div", { class: "msg error" }, run.analysis.error || "The analysis failed.") : null,
        st === "done" ? h("p", { class: "muted" }, "Loading…") : null);
    } else {
      const r = a.result;
      recommendedIndex = a.recommended ?? a.recommendations.findIndex((x) => !x.errors?.length);
      add(head(`by ${a.model} · ${fmtDate(a.created)} · also saved in the lesson's LESSON.md`,
          h("button", { class: "ghost small", onclick: () => analyze(run), title: "Ask again (replaces this analysis)" }, "Re-analyze")),
        a.last_error ? h("div", { class: "msg warn" }, `Analysing it again failed (${a.last_error}). This is the earlier analysis.`) : null,
        h("p", { class: "headline" }, r.headline),
        h("div", { class: "callout" }, h("b", {}, "What we learned"), markdown(r.learned),
          termsRow([r.learned, r.headline], "Terms")),
        r.prediction_check ? h("p", { class: "small ink2" }, h("b", {}, "Prediction check: "), r.prediction_check) : null,
        h("details", { class: "more", open: root.querySelector("#analysis details.more")?.open },
          h("summary", {}, "What happened and why"), h("h3", {}, "What happened"), markdown(r.summary),
          h("h3", {}, "Why"), markdown(r.diagnosis),
          termsRow([r.summary, r.diagnosis], "Terms used above")),
        a.recommendations.length ? h("div", {}, h("h3", { class: "try-next" }, "Next runs ",
            h("span", { class: "muted small" }, "recipe changes the lab can run now")),
          h("div", { class: "recs" }, a.recommendations.map((rec, i) => recCard(run, rec, i)))) : null,
        a.ideas?.length ? h("div", {}, h("h3", { class: "try-next" }, "Ideas for the lab ",
            h("span", { class: "muted small" }, "need new code · saved to the Ideas tab for a later review")),
          h("div", { class: "recs" }, a.ideas.map(ideaCard))) : null,
        h("p", { class: "small muted" }, "AI analyses can be wrong. Check the claims against the charts and numbers on this page."));
    }
    return card;
  }

  function populationCard(run) {
    const ga = run.agents.findIndex((a) => a.algorithm === "ga");
    if (run.status !== "running" || ga < 0) return null;
    population = new Population(run.id, run.agents[ga].id, seriesVar(ga));
    population.set(data.population[run.agents[ga].id]);
    addBoards(population.tiles);
    return h("details", { class: "card" }, h("summary", {}, "GA population ", h("span", { class: "muted" },
      "16 genomes from the latest generation, playing at once")), population.el);
  }

  function recipeCard(run) {
    const p = run.plan;
    return h("details", { class: "card" }, h("summary", {}, "Recipe and resources ",
      h("span", { class: "muted" }, "everything needed to repeat this run")),
      p ? h("p", { class: "small ink2" }, Object.entries(p.agents).map(([id, a]) =>
        `${id}: ${a.cores} ${a.flag === "workers" ? "worker processes" : "torch thread(s)"}${a.mem_mb ? `, ~${fmtMB(a.mem_mb)}` : ""}`).join(" · ")) : null,
      h("pre", { class: "markdown", style: { whiteSpace: "pre-wrap", fontSize: "12px" } }, JSON.stringify(run.config, null, 2)));
  }

  function render() {
    if (!data) { root.replaceChildren(h("div", { class: "card empty" }, "Loading…")); return; }
    const run = data.run;
    removeBoards(population?.tiles || []);
    population = null;
    root.replaceChildren(...[header(run), kpis(run), seedCard(run), analysisCard(run), gamesCard(run), inspector?.el, progressCard(run), endingsCard(run), populationCard(run),
      detailsCard(run), recipeCard(run)].filter(Boolean));
  }

  function refreshNumbers() {  // light re-render of the parts that change while training
    if (!data) return;
    root.querySelector(".kpis-wrap")?.replaceWith(kpis(data.run));
    root.querySelector("#numbers")?.replaceWith(detailsCard(data.run));
    if (endCharts.length < data.run.agents.filter((a) => hasEndings(data.history[a.id])).length) {  // first endings arrived
      const card = endingsCard(data.run), old = root.querySelector("#endings");
      if (old) old.replaceWith(card); else if (card) root.querySelector("#progress")?.after(card);
    }
    drawChart();
  }

  render();
  let throttle = null;
  const ticker = setInterval(() => {  // keep the time-used bar moving while training
    if (data?.run.status === "running") root.firstChild?.replaceWith(header(data.run));
  }, 2000);
  return {
    ownsConnection: true,
    onWatch(payload) {
      if (!payload) { toast("This run no longer exists."); app.go("#runs"); return; }
      data = payload; render();
    },
    onUpdate(ch) {
      if (!data) return;
      for (const a of ch.reset || []) data.history[a] = [];
      for (const [a, rows] of Object.entries(ch.history || {})) (data.history[a] ||= []).push(...rows);
      Object.assign(data.stats, ch.stats || {});
      for (const [a, meta] of Object.entries(ch.population || {})) {
        data.population[a] = meta;
        if (population && population.tiles[0].o.agent === a) population.set(meta);
      }
      if (!throttle) throttle = setTimeout(() => { throttle = null; refreshNumbers(); }, 1000);
    },
    onRuns() {
      const run = app.run(runId);
      if (data && run) {
        const aiChanged = aiKey(run) !== aiKey(data.run) || run.status !== data.run.status;
        data.run = run; root.firstChild?.replaceWith(header(run));
        if (aiChanged) {
          const card = analysisCard(run), old = root.querySelector("#analysis");
          if (old && card) old.replaceWith(card); else if (old) old.remove(); else if (card) root.querySelector(".kpis-wrap")?.after(card);
          root.querySelector("#seeds")?.replaceWith(seedCard(run));
        }
      }
    },
    unmount() { removeBoards(boards, population?.tiles || []); chart?.destroy(); endCharts.forEach((c) => c.chart.destroy()); inspector?.dispose();
      clearTimeout(throttle); clearInterval(ticker); },
  };
}
