// Runs: the library. Active runs first, then every saved run grouped by lesson.
import { fmtSpread, spreadOf } from "../components/spread.js";
import { agentLabel, api, closeDialog, confirmDialog, fmtDate, fmtDuration, fmtMB, h, openDialog, q,
         runTitle, statusPill, toast } from "../util.js";

export function mount(main, _param, app) {
  const state = { search: "", lesson: "", status: "" };
  const openGroups = new Set();  // seed groups whose runs are shown
  const active = h("div");
  const list = h("div", { class: "view" });
  const bar = h("div");
  const machine = h("div");
  const search = h("input", { type: "search", placeholder: "Search name, lesson, vision, reward…", "aria-label": "Search runs",
    oninput: (e) => { state.search = e.target.value.toLowerCase(); renderList(); } });
  const lessonSel = h("select", { "aria-label": "Lesson", onchange: (e) => { state.lesson = e.target.value; renderList(); } });
  const statusSel = h("select", { "aria-label": "Status", onchange: (e) => { state.status = e.target.value; renderList(); } },
    h("option", { value: "" }, "Any status"), ["finished", "stopped", "running", "queued", "interrupted", "failed"]
      .map((s) => h("option", { value: s }, s)));

  main.append(h("div", { class: "view" },
    h("div", { class: "page-head" },
      h("div", {}, h("h1", {}, "Runs"),
        h("p", { class: "lead" }, "Every training run is saved here with its recipe, its stats and the best model of each "
          + "agent. Watch one, run it again with one change, or pick up to three to compare.")),
      h("button", { class: "primary", onclick: () => app.go("#new") }, "New run")),
    active, machine,
    h("div", { class: "toolbar" }, search, lessonSel, statusSel),
    list, bar));

  function renderActive() {
    const runs = app.runs.filter((r) => r.status === "running" || r.status === "queued");
    if (!runs.length) { active.replaceChildren(); return; }
    active.replaceChildren(h("section", { class: "card" },
      h("div", { class: "section-head" }, h("h2", {}, "Training now"),
        h("p", {}, `${app.queue.concurrency} run(s) train at a time; the rest wait in the queue.`)),
      h("div", { class: "active-runs" }, runs.map((r) => {
        const elapsed = r.started ? (Date.now() - Date.parse(r.started.replace(" ", "T"))) / 1000 : 0;
        const pct = r.status === "running" && r.budget_min ? Math.min(100, (elapsed / 60 / r.budget_min) * 100) : 0;
        return h("div", { class: "active-run card" },
          h("div", { class: "top" }, h("b", {}, runTitle(r)), statusPill(r.status)),
          h("div", { class: "chips" }, h("span", { class: "chip" }, r.lesson), h("span", { class: "chip" }, r.vision.label),
            h("span", { class: "chip" }, r.agents.map((a) => agentLabel(a)).join(" + "))),
          r.status === "running" ? h("div", { class: "meter", title: `${pct.toFixed(0)}% of the time budget` },
            h("i", { style: { width: `${pct}%` } })) : null,
          h("div", { class: "section-head" },
            h("span", { class: "small ink2" }, r.status === "running"
              ? `${fmtDuration(elapsed)} of ${r.budget_min ? fmtDuration(r.budget_min * 60) : "∞"}` : "waiting for a free slot"),
            h("div", { class: "btn-row" },
              r.status === "running" ? h("button", { onclick: () => app.go(`#watch/${r.id}`) }, "Watch") : null,
              h("button", { onclick: () => stop(r) }, r.status === "running" ? "Stop & save" : "Remove"))));
      }))));
  }

  function renderMachine() {
    const s = app.system;
    if (!s) return;
    const sel = h("select", { "aria-label": "Runs at a time", onchange: async (e) => {
      app.queue = await api.post("/api/queue", { concurrency: +e.target.value });
      toast(`Up to ${app.queue.concurrency} run(s) will train at the same time.`);
      app.refreshRuns();
    } }, Array.from({ length: Math.min(8, s.cpus) }, (_, i) => h("option", { value: i + 1, selected: i + 1 === app.queue.concurrency }, i + 1)));
    machine.replaceChildren(h("details", { class: "card" },
      h("summary", {}, "This machine ", h("span", { class: "muted" },
        `${s.cpus} CPU cores · ${fmtMB(s.ram_total_mb)} RAM, ${fmtMB(s.ram_available_mb)} free`
        + (s.on_battery ? " · on battery (slower)" : ""))),
      h("p", { class: "ink2" }, "Each run gets its CPU cores and memory planned from what this machine has: GA worker "
        + "processes, DQN threads and replay-buffer size are chosen automatically (see the plan before starting a run)."),
      h("div", { class: "controls" }, h("label", {}, "Runs training at the same time", sel),
        h("span", { class: "small muted" }, "More at once = each gets fewer cores, so wall-clock results are not comparable."))));
  }

  function matches(r) {
    if (state.lesson && r.lesson !== state.lesson) return false;
    if (state.status && r.status !== state.status) return false;
    if (!state.search) return true;
    const text = [r.id, r.name, r.note, r.vision.label, r.vision.spec, r.reward, r.lesson_title,
                  ...r.agents.map((a) => a.algorithm)].join(" ").toLowerCase();
    return text.includes(state.search);
  }

  function renderList() {
    const lessons = [...new Set(app.runs.map((r) => r.lesson))].sort();
    lessonSel.replaceChildren(h("option", { value: "" }, "All lessons"),
      lessons.map((l) => h("option", { value: l, selected: l === state.lesson }, l)));
    const runs = app.runs.filter(matches);
    if (!app.runs.length) {
      list.replaceChildren(h("div", { class: "card empty" }, h("strong", {}, "No runs yet"),
        "Start one: pick what the snake sees, how it is rewarded and which algorithms learn.",
        h("button", { class: "primary", onclick: () => app.go("#new") }, "New run")));
    } else if (!runs.length) {
      list.replaceChildren(h("div", { class: "card empty" }, "No runs match these filters."));
    } else {
      list.replaceChildren(...lessons.filter((l) => runs.some((r) => r.lesson === l)).map((lesson) => {
        const lr = runs.filter((r) => r.lesson === lesson).reverse(); // newest first
        return h("section", { class: "card" },
          h("div", { class: "section-head" },
            h("div", { class: "lesson-head" }, h("h2", {}, lr[0].lesson_title), h("span", { class: "muted small" },
              `${lesson} · ${lr.length} run${lr.length > 1 ? "s" : ""}`)),
            h("button", { onclick: () => readLesson(lesson) }, "Read lesson")),
          h("div", { class: "table-scroll" }, h("table", { class: "runs-table" },
            h("thead", {}, h("tr", {}, h("th", {}, h("span", { class: "hidden" }, "Compare")), h("th", {}, "Run"),
              h("th", {}, "Status"), h("th", {}, "Vision"), h("th", {}, "Reward"),
              h("th", {}, "Eval (mean apples, 100 games)"), h("th", {}, "Time"), h("th", { class: "r" }, "Actions"))),
            h("tbody", {}, rowsFor(lr)))));
      }));
    }
    renderBar();
  }

  // A seed group is one summary row (test score as mean ± spread); its runs show when expanded.
  function rowsFor(lr) {
    const out = [], seen = new Set();
    for (const r of lr) {
      if (!r.group) { out.push(row(r)); continue; }
      if (seen.has(r.group.id)) continue;
      seen.add(r.group.id);
      const members = lr.filter((m) => m.group?.id === r.group.id).sort((a, b) => a.group.index - b.group.index);
      out.push(groupRow(members));
      if (openGroups.has(r.group.id)) out.push(...members.map((m) => row(m, true)));
    }
    return out;
  }

  function groupRow(members) {
    const first = members[0], g = first.group, open = openGroups.has(g.id), perAgent = first.reward === "per agent";
    const toggle = () => { open ? openGroups.delete(g.id) : openGroups.add(g.id); renderList(); };
    const counts = {};
    for (const m of members) counts[m.status] = (counts[m.status] || 0) + 1;
    const analysed = members.find((m) => m.analysis?.recommended) || members.find((m) => m.analysis?.state === "done");
    return h("tr", { class: "group-row" },
      h("td", { class: "check" }, h("button", { class: "ghost small", "aria-expanded": String(open),
        "aria-label": `${open ? "Hide" : "Show"} the ${g.size} seeds`, onclick: toggle }, open ? "▾" : "▸")),
      h("td", {}, h("div", { class: "name" }, runTitle(first), h("span", { class: "seed-chip" }, `${g.size} seeds`),
        h("small", {}, first.note || `seeds ${g.base_seed}–${g.base_seed + g.size - 1}`),
        analysed?.analysis?.headline ? h("small", { class: "ai-line", title: analysed.analysis.headline },
          h("b", {}, "AI "), analysed.analysis.headline) : null)),
      h("td", {}, Object.keys(counts).length === 1 ? statusPill(first.status)
        : h("span", { class: "small ink2" }, Object.entries(counts).map(([s, n]) => `${n} ${s}`).join(" · "))),
      h("td", { class: "hide-sm" }, first.vision.label, h("div", { class: "small muted" }, `${first.vision.inputs.toLocaleString()} inputs`)),
      h("td", { class: "hide-sm" }, first.reward),
      h("td", { class: "wide" }, h("div", { class: "evals" }, first.agents.map((a) => {
        const s = spreadOf(members, a.id);
        return h("span", { title: s?.n > 1 ? `range ${s.min.toFixed(1)}–${s.max.toFixed(1)} over ${s.n} seeds` : "" },
          `${agentLabel(a, perAgent)} `, h("b", {}, fmtSpread(s)));
      }))),
      h("td", { class: "hide-sm small ink2 when" }, first.budget_min ? `${first.budget_min} min × ${g.size}` : "no limit",
        h("div", { class: "muted" }, fmtDate(first.created))),
      h("td", { class: "wide" }, h("div", { class: "actions" },
        analysed ? recommendedButton(analysed) : null,
        h("button", { onclick: () => app.go(`#watch/${first.id}`) }, "Watch"),
        h("button", { onclick: () => app.go(`#new/${first.id}`), title: "Open this recipe (all its seeds) in New run" }, "Run again"),
        h("button", { onclick: toggle }, open ? "Hide seeds" : "Show seeds"))));
  }

  function row(r, member = false) {
    const selected = app.compareIds.includes(r.id);
    const full = !selected && app.compareIds.length >= app.maxCompare;
    const perAgent = r.reward === "per agent";
    return h("tr", { class: [selected ? "selected" : "", member ? "member" : ""].join(" ").trim() },
      h("td", { class: "check" }, h("input", { type: "checkbox", checked: selected, disabled: full,
        title: full ? `Up to ${app.maxCompare} runs can be compared at once` : "Select to compare",
        "aria-label": `Compare ${runTitle(r)}`, onchange: () => { app.toggleCompare(r.id); renderList(); } })),
      h("td", {}, h("div", { class: "name" }, member ? `Seed ${r.config.seed}` : runTitle(r),
        member ? null : h("small", {}, r.note || r.run_id), member ? h("small", {}, r.run_id) : null,
        r.analysis?.state === "done" && r.analysis.headline ? h("small", { class: "ai-line", title: r.analysis.headline },
          h("b", {}, "AI "), r.analysis.headline)
        : r.analysis?.state === "running" ? h("small", { class: "ai-line" }, h("b", {}, "AI "), "analysing…") : null)),
      h("td", {}, statusPill(r.status)),
      member ? h("td", { class: "hide-sm small muted", colspan: 2 }, "same recipe, seed " + r.config.seed)
        : [h("td", { class: "hide-sm" }, r.vision.label, h("div", { class: "small muted" }, `${r.vision.inputs.toLocaleString()} inputs`)),
          h("td", { class: "hide-sm" }, r.reward)],
      h("td", { class: "wide" }, h("div", { class: "evals" }, r.agents.map((a) => {
        const ev = r.evaluation[a.id];
        return h("span", { "--series": "var(--ink-2)" }, `${agentLabel(a, perAgent)} `, h("b", {}, ev ? ev.mean.toFixed(1) : "–"));
      }))),
      h("td", { class: "hide-sm small ink2 when" }, r.budget_min ? `${r.budget_min} min` : "no limit",
        h("div", { class: "muted" }, fmtDate(r.created || r.started))),
      h("td", { class: "wide" }, h("div", { class: "actions" },
        member ? null : recommendedButton(r),
        h("button", { onclick: () => app.go(`#watch/${r.id}`) }, "Watch"),
        h("button", { onclick: () => app.go(`#new/${r.id}`), title: "Open this recipe in New run" }, "Run again"),
        h("button", { onclick: () => edit(r) }, "Edit"),
        h("button", { class: "danger", disabled: r.status === "running", onclick: () => remove(r),
          title: r.status === "running" ? "Stop the run first" : "Delete this run and its models" }, "Delete"))));
  }

  // One press: queue the run the AI analysis recommends (its first recipe change that can be launched).
  function recommendedButton(r) {
    const rec = r.analysis?.recommended;
    if (!rec) return null;
    if (rec.launched.length) {
      return h("button", { title: `“${rec.title}” was launched as ${rec.launched.at(-1)}`,
        onclick: () => app.go(`#watch/${rec.launched.at(-1)}`) }, "Next run →");
    }
    return h("button", { class: "primary", title: `${rec.title}: ${rec.changes.join("; ") || "same recipe"}`
        + `${rec.prediction ? `\nPrediction: ${rec.prediction}` : ""}\nRecorded in ${rec.lesson}`,
      onclick: async (e) => {
        e.currentTarget.disabled = true;
        try {
          const res = await api.post(`/api/analysis/launch?id=${q(r.id)}&index=${rec.index}`);
          toast(`Queued “${rec.title}” in ${res.id.split("/")[0]}.`);
          await app.refreshRuns();
        } catch (err) { toast(err.message); e.target.disabled = false; }
      } }, "Run recommended");
  }

  function renderBar() {
    const n = app.compareIds.length;
    bar.replaceChildren(n ? h("div", { class: "actionbar" },
      h("span", {}, `${n} run${n > 1 ? "s" : ""} selected for comparison (max ${app.maxCompare})`),
      h("div", { class: "btn-row" },
        h("button", { onclick: () => { app.setCompare([]); renderList(); } }, "Clear"),
        h("button", { disabled: n < 2, onclick: () => app.go(`#compare/${app.compareIds.join(",")}`) },
          n < 2 ? "Pick one more" : "Compare"))) : "");
  }

  async function stop(r) {
    try {
      await api.post(`/api/run/stop?id=${q(r.id)}`);
      toast(r.status === "running" ? "Stopping… the run will be evaluated and saved." : "Removed from the queue.");
      app.refreshRuns();
    } catch (e) { toast(e.message); }
  }

  async function remove(r) {
    if (!(await confirmDialog(`Delete ${runTitle(r)}?`, "Its stats, models and logs are deleted for good. "
      + "The lesson write-up keeps your own notes."))) return;
    try {
      await api.del(`/api/run?id=${q(r.id)}`);
      app.setCompare(app.compareIds.filter((x) => x !== r.id));
      toast("Run deleted.");
      app.refreshRuns();
    } catch (e) { toast(e.message); }
  }

  function edit(r) {
    const name = h("input", { value: r.name, placeholder: r.run_id });
    const note = h("textarea", {}, r.note);
    const lesson = h("input", { value: r.lesson, list: "lesson-names" });
    openDialog(h("h2", {}, "Edit run"),
      h("label", { class: "field" }, h("span", {}, "Name"), name),
      h("label", { class: "field" }, h("span", {}, "Note"), note),
      h("label", { class: "field" }, h("span", {}, "Lesson (moving it re-files the run)"), lesson,
        h("datalist", { id: "lesson-names" }, app.catalog.lessons.map((l) => h("option", { value: l.name })))),
      h("div", { class: "btn-row" },
        h("button", { class: "primary", onclick: async () => {
          try {
            const res = await api.patch(`/api/run?id=${q(r.id)}`, { name: name.value, note: note.value, lesson: lesson.value });
            if (res.id !== r.id) app.setCompare(app.compareIds.map((x) => (x === r.id ? res.id : x)));
            closeDialog(); toast("Saved."); app.refreshRuns();
          } catch (e) { toast(e.message); }
        } }, "Save"),
        h("button", { onclick: closeDialog }, "Cancel")));
  }

  async function readLesson(name) {
    const d = await api.get(`/api/lesson?name=${q(name)}`);
    const md = h("div", { class: "markdown" });
    md.innerHTML = marked.parse(d.markdown); // lessons are local files written by you and lessons.py
    openDialog(md, h("div", { class: "btn-row" }, h("button", { onclick: closeDialog }, "Close")));
  }

  const refresh = () => { renderActive(); renderMachine(); renderList(); };
  refresh();
  const ticker = setInterval(renderActive, 2000); // progress bars of running runs
  return { onRuns: refresh, unmount: () => clearInterval(ticker) };
}
