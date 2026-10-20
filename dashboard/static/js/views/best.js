// Best: the strongest saved model of every algorithm, ranked by test score (the same 100 fixed games for
// every model), with the recipe that produced it. Headline tiles first, then one ranked table per group.
import { entry, info } from "../components/glossary.js";
import { fmtSpread, groupMembers, outcomeBar, spreadOf } from "../components/spread.js";
import { agentLabel, fmtCompact, fmtDuration, fmtNum, h, runTitle } from "../util.js";

const GROUPS = { algorithm: "Algorithm", vision: "Vision", reward: "Reward" };
const view = { group: "algorithm", all: false, open: new Set() };
const TOP = 5;

export function mount(main, param, app) {
  const root = h("div", { class: "view" });
  main.append(root);
  const algoOf = (name) => app.catalog.algorithms.find((a) => a.name === name);

  function entries() {
    const out = [];
    for (const r of app.runs) {
      for (const a of r.agents) {
        const ev = r.evaluation?.[a.id];
        if (!ev) continue;
        out.push({ run: r, agent: a, ev, final: r.final?.[a.id], cfg: r.config.agents.find((x) => x.id === a.id) });
      }
    }
    return out.sort((x, y) => y.ev.mean - x.ev.mean || y.ev.median - x.ev.median);
  }

  const groupKey = (e) => view.group === "algorithm" ? e.agent.algorithm : view.group === "vision" ? e.run.vision.spec : e.agent.reward;
  const groupTitle = (key) => view.group === "algorithm" ? algoOf(key)?.label || key : view.group === "vision" ? `Vision ${key}` : `Reward: ${key}`;

  // Settings of this agent that differ from the algorithm's defaults.
  function changed(e) {
    const algo = algoOf(e.agent.algorithm);
    return (algo?.params || []).filter((p) => e.cfg?.params[p.name] !== p.default)
      .map((p) => ({ id: `${algo.name}.${p.name}`, label: p.label, value: e.cfg.params[p.name], def: p.default }));
  }

  function tiles(all) {
    const best = {};
    for (const e of all) best[e.agent.algorithm] ||= e;
    return h("div", { class: "kpis" }, Object.values(best).map((e) => h("div", { class: "stat" },
      h("div", { class: "label" }, `Best ${algoOf(e.agent.algorithm)?.label || e.agent.algorithm}`, info(e.agent.algorithm)),
      h("div", { class: "hero" }, fmtNum(e.ev.mean, 1), h("small", {}, "apples · test (100 games)")),
      h("div", { class: "sub" }, h("span", {}, h("b", {}, runTitle(e.run))), h("span", {}, `vision ${e.run.vision.spec}`),
        h("span", {}, `median ${fmtNum(e.ev.median, 1)} · max ${e.ev.max}`)),
      h("div", { class: "btn-row" }, h("button", { class: "small", onclick: () => app.go(`#watch/${e.run.id}`) }, "Watch"),
        h("button", { class: "small", onclick: () => app.go(`#new/${e.run.id}`), title: "Open this recipe in New run" }, "Run again")))));
  }

  function recipe(e) {
    const r = e.run, c = r.config, algo = algoOf(e.agent.algorithm);
    const rewardFeats = c.reward || e.cfg?.reward || {};
    const kv = (label, value, id, isChanged) => h("div", { class: "kv" + (isChanged ? " changed" : "") },
      h("span", {}, label, id ? info(id, label) : null), h("b", {}, value));
    return h("div", { class: "recipe" },
      h("div", { class: "recipe-col" }, h("h3", {}, "Game"),
        kv("Vision", r.vision.spec, `vision.${c.vision.type}`), kv("Inputs", r.vision.inputs.toLocaleString(), "c.inputs"),
        kv("Board", `${c.width}×${c.height}`), kv("Training time", c.time_budget_min ? `${c.time_budget_min} min` : "until stopped"),
        kv("Seed", c.seed, "c.seed")),
      h("div", { class: "recipe-col" }, h("h3", {}, "Reward", r.reward === "per agent" ? " (this agent's own)" : ""),
        Object.entries(rewardFeats).map(([f, w]) => kv(entry(`reward.${f}`)?.term || f,
          `${app.catalog.rewards.features.find((x) => x.name === f)?.sign < 0 ? "−" : "+"}${w}`, `reward.${f}`))),
      h("div", { class: "recipe-col" }, h("h3", {}, `${algo?.label || e.agent.algorithm} settings`),
        (algo?.params || []).map((p) => kv(p.label, e.cfg?.params[p.name], `${algo.name}.${p.name}`, e.cfg?.params[p.name] !== p.default)),
        h("p", { class: "small muted" }, "Bold rows differ from the default.")));
  }

  function table(rows, max) {
    return h("div", { class: "table-scroll" }, h("table", { class: "best-table" },
      h("thead", {}, h("tr", {}, ["#", "Run", "Test mean (apples)", "Median · max", "How test games ended", "Vision", "Reward", "Changed settings", "Training", ""]
        .map((t, i) => h("th", { class: [2, 3].includes(i) ? "r" : i >= 4 && i <= 8 ? "hide-sm" : "" }, t)))),
      h("tbody", {}, rows.flatMap((e, i) => {
        const key = `${e.run.id}|${e.agent.id}`, open = view.open.has(key);
        const ch = changed(e);
        const tr = h("tr", { class: open ? "open" : "" },
          h("td", { class: "rank" }, i + 1),
          h("td", {}, h("div", { class: "name" }, runTitle(e.run),
            h("small", {}, `${agentLabel(e.agent, e.run.reward === "per agent")} · ${e.run.lesson}`),
            e.run.group ? h("small", {}, `seed ${e.run.group.index + 1} of ${e.run.group.size} · all seeds `,
              h("b", {}, fmtSpread(spreadOf(groupMembers(e.run, app.runs), e.agent.id)))) : null)),
          h("td", { class: "r wide-sm" }, h("div", { class: "barcell" },
            h("span", { class: "bar" }, h("i", { style: { width: `${max ? (e.ev.mean / max) * 100 : 0}%` } })),
            h("b", {}, fmtNum(e.ev.mean, 1)))),
          h("td", { class: "r" }, `${fmtNum(e.ev.median, 1)} · ${e.ev.max}`),
          h("td", { class: "hide-sm" }, outcomeBar(e.ev) || h("span", { class: "muted small" }, "–")),
          h("td", { class: "hide-sm" }, e.run.vision.spec),
          h("td", { class: "hide-sm small" }, e.agent.reward),
          h("td", { class: "hide-sm" }, ch.length ? h("div", { class: "chips" }, ch.map((c) =>
            h("span", { class: "chip", title: `default ${c.def}` }, `${c.label} `, h("b", {}, c.value)))) : h("span", { class: "muted small" }, "defaults")),
          h("td", { class: "hide-sm small ink2" }, e.final ? `${fmtDuration(e.final.elapsed_s)} · ${fmtCompact(e.final.total_steps)} steps` : "–"),
          h("td", {}, h("div", { class: "actions" },
            h("button", { "aria-expanded": String(open), onclick: () => { open ? view.open.delete(key) : view.open.add(key); render(); } },
              open ? "Hide recipe" : "Recipe"),
            h("button", { onclick: () => app.go(`#watch/${e.run.id}`) }, "Watch"),
            h("button", { onclick: () => app.go(`#new/${e.run.id}`), title: "Open this recipe in New run" }, "Run again"))));
        return open ? [tr, h("tr", { class: "detail" }, h("td", { colspan: 10 }, recipe(e)))] : [tr];
      }))));
  }

  function render() {
    const all = entries();
    if (!all.length) {
      root.replaceChildren(h("div", { class: "card empty" }, h("strong", {}, "No tested models yet"),
        "Finished runs are tested on 100 fixed games; the best ones appear here.",
        h("button", { class: "primary", onclick: () => app.go("#new") }, "New run")));
      return;
    }
    const max = all[0].ev.mean;
    const groups = new Map();
    for (const e of all) { const k = groupKey(e); if (!groups.has(k)) groups.set(k, []); groups.get(k).push(e); }
    const seg = (options, key) => h("div", { class: "segmented", role: "group" }, Object.entries(options).map(([k, label]) =>
      h("button", { "aria-pressed": String(view[key] === k || String(view[key]) === k), onclick: () => {
        view[key] = key === "all" ? k === "true" : k; render(); } }, label)));
    root.replaceChildren(
      h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Best models"),
        h("p", { class: "lead" }, "Every saved model ranked by its test score: the mean apples over the same 100 fixed games, "
          + "so models from different runs compare fairly. Open a recipe to see exactly what produced it."))),
      tiles(all),
      h("div", { class: "controls" }, h("div", { class: "ctl", role: "group", "aria-label": "Group by" }, "Group by", seg(GROUPS, "group")),
        h("div", { class: "ctl", role: "group", "aria-label": "Show" }, "Show", seg({ false: `Top ${TOP}`, true: "All" }, "all"))),
      ...[...groups.entries()].map(([k, rows]) => h("section", { class: "card" },
        h("div", { class: "section-head" }, h("h2", {}, groupTitle(k), view.group === "algorithm" ? info(k) : null),
          h("p", {}, `${rows.length} tested model${rows.length > 1 ? "s" : ""}${!view.all && rows.length > TOP ? ` · showing the top ${TOP}` : ""}`)),
        table(view.all ? rows : rows.slice(0, TOP), max))),
      h("p", { class: "small muted" }, "Bars share one scale across all tables. A single run is a single seed: "
        + "small gaps between models can be noise."));
  }

  render();
  return { onRuns: render };
}
