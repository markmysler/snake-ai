// Compare: up to 3 runs. Color = run (fixed per slot). One small chart per algorithm with a
// shared y-scale, a "what changed" table that highlights differences, and test results.
import { Board } from "../components/board.js";
import { LineChart, maxOf, METRICS, X_MODES } from "../components/linechart.js";
import { info } from "../components/glossary.js";
import { addBoards, playbackControls, removeBoards } from "../components/playback.js";
import { groupMembers, outcomeBar, outcomeLegend, spreadOf, stripPlot, verdict } from "../components/spread.js";
import { agentLabel, api, fmtCompact, fmtDuration, fmtNum, h, q, runTitle, seriesColor, seriesVar,
         statusPill } from "../util.js";

const view = { x: "time", metric: "mean", logX: false, onlyDiff: true };
const ALGO_LABEL = { ga: "Genetic algorithm", dqn: "Deep Q-learning (DQN)" };

export function mount(main, param, app) {
  if (param) app.setCompare(param.split(","));
  const root = h("div", { class: "view" });
  main.append(root);
  const histories = {};
  let charts = [], boards = [], timer = null;

  const runs = () => app.compareIds.map((id) => app.run(id)).filter(Boolean);
  const color = (i) => seriesColor(i);

  async function load() {
    await Promise.all(runs().filter((r) => !histories[r.id] || r.status === "running").map(async (r) => {
      try { histories[r.id] = (await api.get(`/api/history?id=${q(r.id)}`)).history; } catch (e) { histories[r.id] = {}; }
    }));
  }

  function picker(rs) {
    const others = app.runs.filter((r) => !app.compareIds.includes(r.id));
    const add = h("select", { "aria-label": "Add a run", disabled: rs.length >= app.maxCompare, onchange: (e) => {
      app.setCompare([...app.compareIds, e.target.value]); history.replaceState(null, "", `#compare/${app.compareIds.join(",")}`); render(true);
    } }, h("option", { value: "" }, rs.length >= app.maxCompare ? `Up to ${app.maxCompare} runs` : "+ Add a run…"),
      [...others].reverse().map((r) => h("option", { value: r.id }, `${r.lesson} · ${runTitle(r)} (${r.vision.spec})`)));
    return h("section", { class: "card" },
      h("div", { class: "section-head" }, h("h2", {}, "Runs being compared"), h("p", {}, "Colors stay with each run on every chart and board.")),
      h("div", { class: "btn-row" }, rs.map((r, i) => h("span", { class: "chip run-chip", "--series": seriesVar(i) },
        h("span", { class: "swatch" }), h("b", {}, runTitle(r)), h("span", {}, r.lesson), statusPill(r.status),
        h("button", { class: "ghost", "aria-label": `Remove ${runTitle(r)}`,
          onclick: () => { app.setCompare(app.compareIds.filter((x) => x !== r.id)); history.replaceState(null, "", `#compare/${app.compareIds.join(",")}`); render(true); } }, "✕"))),
        add));
  }

  // ---- what changed ----
  function settingRows(rs) {
    const algos = [...new Set(rs.flatMap((r) => r.config.agents.map((a) => a.algorithm)))];
    // kind: "info" = context only (not a choice), "recipe" = what the experimenter changed, "machine" = affects wall-clock only
    const rows = [
      ["Lesson", (r) => r.lesson, "info"], ["Vision", (r) => r.vision.spec], ["Inputs", (r) => r.vision.inputs.toLocaleString(), "info"],
      ["Reward", (r) => r.reward === "per agent" ? r.agents.map((a) => `${a.id}: ${a.reward}`).join(", ") : r.reward],
      ["Agents", (r) => r.agents.map((a) => agentLabel(a)).join(" + ")],
    ];
    for (const algo of algos) {
      const spec = app.catalog.algorithms.find((x) => x.name === algo);
      for (const p of spec?.params || []) {
        rows.push([`${algo.toUpperCase()} · ${p.label}`, (r) => {
          const a = r.config.agents.find((x) => x.algorithm === algo);
          return a ? String(a.params[p.name]) : "–";
        }]);
      }
    }
    rows.push(["Training time", (r) => r.budget_min ? `${r.budget_min} min` : "no limit"],
      ["Board", (r) => `${r.config.width}×${r.config.height}`], ["Seed", (r) => r.config.seed],
      ["CPUs", (r) => r.machine.cpus ?? "?", "machine"],
      ["Power", (r) => r.machine.on_battery ? "battery" : r.machine.on_battery === false ? "mains" : "?", "machine"],
      ["Runs at once", (r) => r.parallel, "machine"]);
    return rows.map(([label, get, kind = "recipe"]) => {
      const vals = rs.map((r) => String(get(r)));
      return { label, vals, kind, diff: new Set(vals).size > 1 };
    });
  }

  function changedCard(rs) {
    const rows = settingRows(rs);
    const shown = view.onlyDiff ? rows.filter((r) => r.diff) : rows;
    const nDiff = rows.filter((r) => r.diff && r.kind === "recipe").length;
    const machineDiff = rows.some((r) => r.diff && r.kind === "machine");
    return h("section", { class: "card" },
      h("div", { class: "section-head" }, h("div", {}, h("h2", {}, "What changed"),
        h("p", {}, (nDiff === 0 ? "The recipes are identical (a repeat: differences come from chance)."
          : nDiff === 1 ? "Exactly one setting differs: a clean experiment."
          : `${nDiff} settings differ, so a difference in results can't be pinned on one of them.`)
          + (machineDiff ? " The machine setup differed too, so compare wall-clock with care." : ""))),
        h("label", { class: "controls" }, h("input", { type: "checkbox", checked: view.onlyDiff,
          onchange: (e) => { view.onlyDiff = e.target.checked; root.querySelector("#changed").replaceWith(changedCard(rs)); } }),
          "Only differences")),
      h("div", { class: "table-scroll" }, h("table", {},
        h("thead", {}, h("tr", {}, h("th", {}, "Setting"), rs.map((r, i) => h("th", { "--series": seriesVar(i) },
          h("span", { class: "swatch" }), " ", runTitle(r))))),
        h("tbody", {}, shown.length ? shown.map((row) => h("tr", { class: row.diff && row.kind !== "info" ? "diff" : "" },
          h("th", {}, row.label), row.vals.map((v) => h("td", {}, v))))
          : h("tr", {}, h("td", { colspan: rs.length + 1, class: "muted" }, "No differences."))))));
  }

  // ---- curves: one chart per algorithm, color = run, shared y-scale ----
  function curvesCard(rs) {
    charts.forEach((c) => c.destroy());
    charts = [];
    const algos = [...new Set(rs.flatMap((r) => r.agents.map((a) => a.algorithm)))];
    const seg = (options, key) => h("div", { class: "segmented", role: "group" }, Object.entries(options).map(([k, label]) =>
      h("button", { "aria-pressed": String(view[key] === k), onclick: () => { view[key] = k; root.querySelector("#curves").replaceWith(curvesCard(rs)); } }, label)));
    const facets = algos.map((algo) => {
      const canvas = h("canvas");
      const series = rs.flatMap((r, i) => r.agents.filter((a) => a.algorithm === algo).map((a, j) => ({
        label: `${runTitle(r)} · ${agentLabel(a, r.reward === "per agent")}`, short: j ? `${runTitle(r).slice(0, 10)} ${a.id}` : runTitle(r).slice(0, 14),
        color: color(i), dash: j > 0, rows: histories[r.id]?.[a.id] || [] })));
      return { el: h("div", { class: "facet" }, h("h3", {}, ALGO_LABEL[algo] || algo), h("div", { class: "chart-box" }, canvas)), canvas, series };
    });
    const yMax = Math.ceil(maxOf(facets.flatMap((f) => f.series.map((s) => s.rows)), view.metric) * 1.05) || undefined;
    queueMicrotask(() => {
      for (const f of facets) {
        const c = new LineChart(f.canvas);
        c.set(f.series, { x: view.x, metric: view.metric, logX: view.logX, yMax });
        charts.push(c);
      }
    });
    return h("section", { class: "card", id: "curves" },
      h("div", { class: "section-head" }, h("h2", {}, "How they learned"),
        h("p", {}, "Same y-scale on every chart. Env steps = fairest for sample efficiency; wall-clock = fairest for speed.")),
      h("div", { class: "controls" },
        h("div", { class: "ctl", role: "group", "aria-label": "Show" }, "Show", seg({ mean: "Mean score", best: "Best score" }, "metric")),
        h("div", { class: "ctl", role: "group", "aria-label": "Over" }, "Over", seg({ time: "Time", steps: "Env steps", games: "Games" }, "x")),
        h("label", {}, h("input", { type: "checkbox", checked: view.logX, onchange: (e) => { view.logX = e.target.checked; root.querySelector("#curves").replaceWith(curvesCard(rs)); } }), "log x")),
      h("div", { class: "legend" }, rs.map((r, i) => h("span", { "--series": seriesVar(i) }, h("i", { class: "key" }), runTitle(r))),
        rs.some((r) => r.reward === "per agent") ? h("span", {}, h("i", { class: "key dash" }), "second agent of the same algorithm") : null),
      h("div", { class: "facets" }, facets.map((f) => f.el)),
      h("p", { class: "small muted" }, `${METRICS[view.metric].title} against ${X_MODES[view.x].title.toLowerCase()}.`));
  }

  // ---- test results ----
  function resultsCard(rs) {
    const agentIds = [...new Set(rs.flatMap((r) => r.agents.map((a) => a.id)))];
    const cell = (r, id) => r.evaluation[id];
    const row = (label, values, best) => h("tr", {}, h("th", {}, label), values.map((v, i) => h("td", { class: "r" },
      v === null ? "–" : h("span", { class: best !== null && i === best ? "best" : "" }, v))));
    return h("section", { class: "card" },
      h("div", { class: "section-head" }, h("div", {}, h("h2", {}, "Test results"),
        h("p", {}, "Each saved best model plays the same 100 games (fixed seeds, no exploration). Higher is better; the best is bold.")),
        h("div", { class: "btn-row small muted" }, outcomeLegend(), info("c.game-end"))),
      h("div", { class: "table-scroll" }, h("table", {},
        h("thead", {}, h("tr", {}, h("th", {}, "Agent"), rs.map((r, i) => h("th", { class: "r", "--series": seriesVar(i) },
          h("span", { class: "swatch" }), " ", runTitle(r))))),
        h("tbody", {},
          agentIds.flatMap((id) => {
            const evs = rs.map((r) => cell(r, id));
            const means = evs.map((e) => (e ? e.mean : -1));
            const best = Math.max(...means) >= 0 ? means.indexOf(Math.max(...means)) : null;
            const sample = rs.find((r) => r.agents.some((a) => a.id === id)).agents.find((a) => a.id === id);
            const testCell = (e) => (e ? h("span", {}, fmtNum(e.mean, 2), h("span", { class: "muted small" }, `  median ${fmtNum(e.median, 1)} · max ${e.max}`)) : null);
            return [
              row(h("span", {}, agentLabel(sample), h("span", { class: "muted small" }, "  test score")), evs.map(testCell),
                rs.filter((_, i) => evs[i]).length > 1 ? best : null),
              row(h("span", { class: "muted small" }, "how test games ended"), evs.map((e) => e?.outcomes
                ? h("div", {}, outcomeBar(e), h("div", { class: "small muted" }, `${fmtNum(e.mean_steps, 0)} moves/game · ${fmtNum(e.steps_per_apple, 0)}/apple`))
                : null), null),
              row(h("span", { class: "muted small" }, "experience used"), rs.map((r) => (r.final[id]
                ? `${fmtCompact(r.final[id].total_steps)} steps · ${fmtDuration(r.final[id].elapsed_s)}` : null)), null),
            ];
          })))));
  }

  // ---- is the difference real? each run's test score across its seed group, against the first run ----
  function spreadCard(rs) {
    const agentIds = [...new Set(rs.flatMap((r) => r.agents.map((a) => a.id)))];
    const groups = rs.map((r) => groupMembers(r, app.runs));
    const max = Math.max(1, ...groups.flatMap((ms) => ms.flatMap((m) => Object.values(m.evaluation || {}).map((e) => e.mean))));
    const anyGroup = rs.some((r) => r.group);
    return h("section", { class: "card" },
      h("div", { class: "section-head" }, h("div", {}, h("h2", {}, "Is the difference real? ", info("c.seed-spread")),
        h("p", {}, anyGroup ? "Each dot is one seed's test score for that recipe; the bar is the range across seeds. "
          + "Ranges that don't overlap mean a real difference; overlapping ranges could be luck."
          : "Every run here is a single seed, so a difference may be luck. Train a recipe with Seeds ≥ 3 in New run to see its spread."))),
      agentIds.map((id) => {
        const sample = rs.find((r) => r.agents.some((a) => a.id === id)).agents.find((a) => a.id === id);
        const spreads = groups.map((ms) => spreadOf(ms, id));
        return h("div", { class: "facet" }, h("h3", {}, agentLabel(sample)),
          stripPlot(rs.map((r, i) => ({ label: runTitle(r), color: seriesVar(i), spread: spreads[i], highlight: r.id })), max),
          rs.slice(1).map((r, j) => spreads[0] && spreads[j + 1] ? h("p", { class: `verdict ${verdict(spreads[0], spreads[j + 1]).kind}` },
            h("b", {}, `${runTitle(r)} vs ${runTitle(rs[0])}: `), verdict(spreads[0], spreads[j + 1]).text) : null));
      }));
  }

  function gamesCard(rs) {
    removeBoards(boards);
    boards = [];
    const rows = rs.map((r, i) => h("div", { class: "row" },
      h("div", { class: "row-label", "--series": seriesVar(i) }, h("span", { class: "swatch" }), h("span", {}, runTitle(r))),
      r.agents.map((a) => {
        const b = new Board({ run: r.id, agent: a.id, title: agentLabel(a, r.reward === "per agent"), color: seriesVar(i),
          waitingText: "No saved model yet" });
        boards.push(b);
        return b.el;
      })));
    addBoards(boards);
    return h("details", { class: "card" }, h("summary", {}, "Watch them play side by side ",
      h("span", { class: "muted" }, "each run's saved best models")), playbackControls(), h("div", { class: "compare-boards" }, rows));
  }

  async function render(refetch = false) {
    const rs = runs();
    if (rs.length < 2) {
      removeBoards(boards); charts.forEach((c) => c.destroy()); charts = [];
      root.replaceChildren(h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Compare"),
        h("p", { class: "lead" }, "Put up to three runs side by side to see whether a change helped."))),
        picker(rs), h("div", { class: "card empty" }, h("strong", {}, rs.length ? "Pick one more run" : "Pick two or three runs"),
          "Use “Add a run” above, or tick runs in the Runs list. Tip: compare runs that differ in exactly one setting.",
          h("button", { onclick: () => app.go("#runs") }, "Go to Runs")));
      return;
    }
    if (refetch || rs.some((r) => !histories[r.id])) await load();
    const open = root.querySelector("details.card[open] .compare-boards") !== null;
    root.replaceChildren(
      h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Compare"),
        h("p", { class: "lead" }, "Did the change help? Read top to bottom: what changed, how they learned, how they test."))),
      picker(rs), Object.assign(changedCard(rs), { id: "changed" }), curvesCard(rs), resultsCard(rs), spreadCard(rs),
      Object.assign(gamesCard(rs), { open }));
  }

  render(true);
  timer = setInterval(async () => {  // keep curves of runs that are still training up to date
    const live = runs().filter((r) => r.status === "running");
    if (!live.length) return;
    await load();
    root.querySelector("#curves")?.replaceWith(curvesCard(runs()));
  }, 5000);
  let known = JSON.stringify(runs().map((r) => [r.id, r.status]));
  return {
    onRuns() {
      const now = JSON.stringify(runs().map((r) => [r.id, r.status]));
      if (now !== known) { known = now; render(true); }
    },
    unmount() { clearInterval(timer); removeBoards(boards); charts.forEach((c) => c.destroy()); },
  };
}
