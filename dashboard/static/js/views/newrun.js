// New run: build a recipe (lesson, algorithms, vision, reward, time) and start it.
// Every option comes from the server's catalog, so new visions / rewards / algorithms show up here.
import { info } from "../components/glossary.js";
import { api, fmtMB, h, toast } from "../util.js";

const clone = (o) => JSON.parse(JSON.stringify(o));

export function mount(main, fromId, app) {
  const cat = app.catalog;
  // #new/draft = an AI recommendation opened with "Edit first" (keeps its name, prediction and origin),
  // or a course exercise ("Try this": draft.exercise names the lesson)
  const draft = fromId === "draft" ? app.draft : null;
  const source = fromId && !draft ? app.run(fromId) : null;
  const cfg = clone(draft ? draft.config : source ? source.config : cat.defaults);
  if (source) { cfg.name = ""; cfg.note = ""; }
  let plan = null, timer = null, newLesson = !cat.lessons.some((l) => l.name === cfg.lesson);
  let autoAnalyze = !!cat.llm?.configured;
  let seeds = draft?.seeds || source?.group?.size || 1;
  if (source?.group) cfg.seed = source.group.base_seed;  // repeat the whole group, not one of its seeds

  const cards = { lesson: h("section", { class: "card" }), algos: h("section", { class: "card" }),
                  vision: h("section", { class: "card" }), reward: h("section", { class: "card" }),
                  time: h("section", { class: "card" }) };
  const summary = h("aside", { class: "summary" });
  main.append(h("div", { class: "view" },
    h("div", { class: "page-head" }, h("div", {}, h("h1", {}, draft?.exercise ? "Try this" : draft ? "Recommended run" : source ? "Run again" : "New run"),
      h("p", { class: "lead" }, draft?.exercise
        ? `“${draft.title}”, an exercise from ${draft.exercise}. ${draft.why || ""} Adjust anything, then start it.`
        : draft
        ? `“${draft.title}”, recommended by the AI analysis of ${draft.origin.run}. Adjust anything, then start it.`
        : source
        ? `Settings copied from ${source.id}. Change one thing so you can tell what caused the difference.`
        : "Choose what the snake sees, how it is rewarded and which algorithms learn. The run is saved "
          + "with this exact recipe, so anyone can repeat it."))),
    h("div", { class: "form-grid" },
      h("div", { class: "form-main" }, cards.lesson, cards.algos, cards.vision, cards.reward, cards.time), summary)));

  const step = (n, title, help) => h("div", { class: "step-title" }, h("span", { class: "n" }, n),
    h("div", {}, h("h2", {}, title), help ? h("p", {}, help) : null));
  const changed = () => { clearTimeout(timer); timer = setTimeout(replan, 250); };
  function numberField(p, value, onValue, gid) {
    return h("label", { class: "field", title: p.help || "" }, h("span", {}, p.label, gid ? info(gid, p.label) : null),
      h("input", { type: "number", min: p.min, max: p.max, step: p.type === "int" ? 1 : "any", value,
        oninput: (e) => { if (e.target.value !== "") { onValue(+e.target.value); changed(); } } }),
      p.help ? h("small", {}, p.help) : null);
  }

  // ---- 1. lesson ----
  function renderLesson() {
    const sel = h("select", { onchange: (e) => {
      newLesson = e.target.value === "__new";
      if (!newLesson) cfg.lesson = e.target.value;
      renderLesson(); changed();
    } }, cat.lessons.map((l) => h("option", { value: l.name, selected: !newLesson && l.name === cfg.lesson }, l.title)),
      h("option", { value: "__new", selected: newLesson }, "+ New lesson…"));
    cards.lesson.replaceChildren(step(1, "Lesson", "Runs are grouped by the question they answer."),
      h("div", { class: "fields" },
        h("label", { class: "field" }, h("span", {}, "Lesson"), sel),
        newLesson ? h("label", { class: "field" }, h("span", {}, "New lesson name"),
          h("input", { value: newLesson && cfg.lesson !== "scratch" ? cfg.lesson : "",
            placeholder: `e.g. ${String(cat.lessons.filter((l) => /^\d/.test(l.name)).length + 1).padStart(2, "0")}-my-question`,
            oninput: (e) => { cfg.lesson = e.target.value; changed(); } })) : null,
        h("label", { class: "field" }, h("span", {}, "Run name (optional)"),
          h("input", { value: cfg.name, placeholder: "e.g. rays, depth 4", oninput: (e) => { cfg.name = e.target.value; } }))),
      h("label", { class: "field" }, h("span", {}, "Prediction / note"),
        h("textarea", { placeholder: "What do you expect to happen, and why? Write it before you look at the results.",
          oninput: (e) => { cfg.note = e.target.value; } }, cfg.note)));
  }

  // ---- 2. algorithms ----
  function renderAlgos() {
    cards.algos.replaceChildren(step(2, "Training algorithms", "Each one trains its own snake at the same time."),
      h("div", { class: "options" }, cat.algorithms.map((algo) => {
        const agent = cfg.agents.find((a) => a.algorithm === algo.name);
        return h("div", { class: "option" },
          h("label", { class: "title" }, h("input", { type: "checkbox", checked: !!agent, onchange: (e) => {
            if (e.target.checked) cfg.agents.push({ algorithm: algo.name, params: {} });
            else cfg.agents = cfg.agents.filter((a) => a.algorithm !== algo.name);
            renderAlgos(); changed();
          } }), algo.label, info(algo.name)),
          h("p", { class: "desc" }, algo.description),
          agent ? h("details", {}, h("summary", {}, "Settings"), h("div", { class: "fields", style: { marginTop: "8px" } },
            algo.params.map((p) => numberField(p, agent.params[p.name] ?? p.default, (v) => { agent.params[p.name] = v; }, `${algo.name}.${p.name}`)),
            h("label", { class: "field" }, h("span", {}, `CPU (${algo.resource.label})`),
              h("input", { type: "number", min: 1, max: app.system?.cpus || 64, placeholder: "auto", value: agent.cores || "",
                oninput: (e) => { agent.cores = e.target.value ? +e.target.value : undefined; changed(); } }),
              h("small", {}, "empty = planned for this machine")))) : null);
      })));
  }

  // ---- 3. vision ----
  function renderVision() {
    cards.vision.replaceChildren(step(3, "Vision", "What the network gets to see. More inputs = more information, but slower learning."),
      h("div", { class: "options" }, cat.visions.map((v) => {
        const selected = cfg.vision.type === v.name;
        return h("label", { class: "option" },
          h("span", { class: "title" }, h("input", { type: "radio", name: "vision", checked: selected, onchange: () => {
            cfg.vision = { type: v.name, ...Object.fromEntries(v.params.map((p) => [p.name, p.default])) };
            renderVision(); changed();
          } }), v.label.split(" · ")[0], info(`vision.${v.name}`),
            selected && plan?.plan ? h("span", { class: "inputs" }, `${plan.plan.inputs.toLocaleString()} inputs`) : null),
          h("p", { class: "desc" }, v.description),
          selected && v.params.length ? h("div", { class: "params" },
            v.params.map((p) => numberField(p, cfg.vision[p.name] ?? p.default, (x) => { cfg.vision[p.name] = x; }, `vision.${v.name}.${p.name}`))) : null);
      })));
  }

  // ---- 4. reward ----
  function renderReward() {
    const feats = cat.rewards.features;
    if (cfg.reward === null) {
      cards.reward.replaceChildren(step(4, "Reward"), h("div", { class: "msg warn" },
        "This recipe gives each agent its own reward (lessons 01-02 did this). It will be repeated exactly as recorded."),
      h("button", { onclick: () => {
        cfg.reward = clone(cfg.agents[0]?.reward || cat.rewards.presets.score);
        for (const a of cfg.agents) delete a.reward;
        renderReward(); changed();
      } }, "Use one reward for every agent instead"));
      return;
    }
    const conflicts = (name) => feats.find((f) => f.name === name).incompatible.filter((o) => o in cfg.reward);
    const presetName = Object.entries(cat.rewards.presets).find(([, p]) =>
      JSON.stringify(Object.entries(p).sort()) === JSON.stringify(Object.entries(cfg.reward).sort()))?.[0];
    cards.reward.replaceChildren(step(4, "Reward", "Tick the signals the snake learns from. The GA uses the total as its fitness."),
      h("div", { class: "presets" }, h("span", { class: "small muted" }, "Presets"),
        h("div", { class: "segmented" }, Object.keys(cat.rewards.presets).map((name) => h("button", {
          "aria-pressed": String(presetName === name),
          onclick: () => { cfg.reward = clone(cat.rewards.presets[name]); renderReward(); changed(); } }, name)))),
      h("div", {}, feats.map((f) => {
        const on = f.name in cfg.reward, blocked = !on && conflicts(f.name).length > 0;
        const other = blocked ? feats.find((x) => x.name === conflicts(f.name)[0]).label : "";
        const sign = f.sign < 0 ? "−" : "+";
        return h("div", { class: "feature" + (blocked ? " disabled" : "") },
          h("input", { type: "checkbox", checked: on, disabled: blocked, "aria-label": f.label, onchange: (e) => {
            if (e.target.checked) cfg.reward[f.name] = f.default; else delete cfg.reward[f.name];
            renderReward(); changed();
          } }),
          h("div", { class: "what" }, h("b", {}, f.label, info(`reward.${f.name}`)), h("small", {}, f.description)),
          h("label", { class: "weight" }, sign, h("input", { type: "number", min: 0, step: "any", disabled: !on,
            value: on ? cfg.reward[f.name] : f.default, "aria-label": `${f.label} weight`,
            oninput: (e) => { if (on && e.target.value !== "") { cfg.reward[f.name] = +e.target.value; changed(); } } }), f.when),
          blocked ? h("div", { class: "conflict" }, `Can't be combined with “${other}”.`) : null);
      })));
  }

  // ---- 5. time & board ----
  function renderTime() {
    const minutes = h("input", { type: "number", min: 0.1, max: 1440, step: "any", value: cfg.time_budget_min,
      oninput: (e) => { cfg.time_budget_min = +e.target.value; changed(); } });
    cards.time.replaceChildren(step(5, "Time and board", "Every algorithm gets the same wall-clock time, so the comparison is fair."),
      h("div", { class: "fields" },
        h("label", { class: "field" }, h("span", {}, "Training time (minutes)"), minutes,
          h("div", { class: "btn-row" }, [1, 3, 10, 30].map((m) => h("button", { class: "ghost small",
            onclick: () => { cfg.time_budget_min = m; minutes.value = m; changed(); } }, `${m} min`)))),
        h("label", { class: "field" }, h("span", {}, "Board width"), h("input", { type: "number", min: 6, max: 30, value: cfg.width,
          oninput: (e) => { cfg.width = +e.target.value; changed(); } })),
        h("label", { class: "field" }, h("span", {}, "Board height"), h("input", { type: "number", min: 6, max: 30, value: cfg.height,
          oninput: (e) => { cfg.height = +e.target.value; changed(); } })),
        h("label", { class: "field" }, h("span", {}, "Random seed", info("c.seed")), h("input", { type: "number", value: cfg.seed,
          oninput: (e) => { cfg.seed = +e.target.value; changed(); } }), h("small", {}, "same seed + recipe = same run")),
        h("label", { class: "field" }, h("span", {}, "Seeds", info("c.seed-spread")),
          h("input", { type: "number", min: 1, max: 10, value: seeds,
            oninput: (e) => { seeds = Math.max(1, Math.min(10, Math.round(+e.target.value) || 1)); renderSummary(); } }),
          h("small", {}, "train the recipe this many times (seed, seed+1, …) to see how much is luck"))));
  }

  // ---- summary + plan ----
  function meter(label, used, avail, fmt) {
    const pct = avail ? Math.min(100, (used / avail) * 100) : 0;
    return h("div", { class: "meter-row" }, h("span", {}, label),
      h("div", { class: "meter" + (used > avail ? " over" : "") }, h("i", { style: { width: `${pct}%` } })),
      h("span", {}, `${fmt(used)} / ${fmt(avail)}`));
  }

  function renderSummary() {
    const p = plan?.plan, d = plan?.describe;
    const errors = plan?.errors || [], warnings = plan?.warnings || [];
    const ahead = app.runs.filter((r) => r.status === "queued").length
      + Math.max(0, app.runs.filter((r) => r.status === "running").length - app.queue.concurrency + 1);
    const start = h("button", { class: "primary", disabled: !p || errors.length > 0, onclick: launch },
      (ahead > 0 ? `Add to queue (${ahead} ahead)` : "Start training") + (seeds > 1 ? ` · ${seeds} seeds` : ""));
    summary.replaceChildren(
      h("section", { class: "card" }, h("h2", {}, "Recipe"),
        d ? h("div", { class: "chips" },
          h("span", { class: "chip" }, "lesson ", h("b", {}, plan.config.lesson)),
          h("span", { class: "chip" }, "vision ", h("b", {}, d.vision_spec)),
          h("span", { class: "chip" }, h("b", {}, d.inputs.toLocaleString()), " inputs"),
          h("span", { class: "chip" }, "reward ", h("b", {}, d.reward)),
          d.agents.map((a) => h("span", { class: "chip" }, h("b", {}, a.label))),
          h("span", { class: "chip" }, h("b", {}, `${plan.config.time_budget_min} min`)),
          seeds > 1 ? h("span", { class: "chip" }, h("b", {}, `${seeds} seeds`), ` ${plan.config.seed}–${plan.config.seed + seeds - 1}`) : null)
          : h("p", { class: "muted" }, "Checking…"),
        seeds > 1 && plan?.config ? h("p", { class: "small ink2" }, `${seeds} runs of ${plan.config.time_budget_min} min. `
          + (app.queue.concurrency >= seeds ? "They train side by side, sharing this machine."
            : `With ${app.queue.concurrency} run(s) at a time that takes about ${Math.ceil(seeds / Math.max(1, app.queue.concurrency))
              * plan.config.time_budget_min} min; allow more at once in Runs to finish sooner.`)) : null,
        errors.length || warnings.length ? h("div", { class: "msgs" },
          errors.map((e) => h("div", { class: "msg error" }, e)), warnings.map((w) => h("div", { class: "msg warn" }, w))) : null,
        cat.llm?.configured ? h("label", { class: "check-line", title: `Uses ${cat.llm.model}` },
          h("input", { type: "checkbox", checked: autoAnalyze, onchange: (e) => { autoAnalyze = e.target.checked; } }),
          "Analyze with AI when it ends") : null,
        h("div", { class: "btn-row" }, start, h("button", { class: "ghost", onclick: () => app.go("#runs") }, "Cancel"))),
      p ? h("section", { class: "card" }, h("h2", {}, "How this machine will run it"),
        meter("CPU cores", p.cores_used, p.cores_per_run, (v) => v),
        meter("Memory", p.mem_mb, p.ram_budget_mb, fmtMB),
        h("div", { class: "msgs small ink2" }, Object.entries(p.agents).map(([id, a]) =>
          h("div", {}, h("b", {}, id), ` · ${a.cores} ${a.flag === "workers" ? "worker processes" : "torch thread(s)"} · ~${fmtMB(a.mem_mb)}`
            + (a.params.buffer ? ` · buffer ${a.params.buffer.toLocaleString()}` : "")))),
        h("p", { class: "small muted" }, `Up to ${p.max_parallel} runs like this fit on this machine at once `
          + `(currently ${app.queue.concurrency} at a time, change it in Runs).`)) : null);
  }

  async function replan() {
    try {
      plan = await api.post("/api/plan", { config: cfg });
    } catch (e) { plan = { errors: [e.message], warnings: [] }; }
    renderSummary();
    if (plan.plan) {  // refresh the inputs badge on the selected vision
      const badge = cards.vision.querySelector("input[type=radio]:checked")?.closest(".title");
      badge?.querySelector(".inputs")?.remove();
      badge?.append(h("span", { class: "inputs" }, `${plan.plan.inputs.toLocaleString()} inputs`));
    }
  }

  async function launch() {
    try {
      const res = await api.post("/api/runs", { config: cfg, auto_analyze: autoAnalyze, origin: draft?.origin, seeds });
      toast(res.queue.running.length >= res.queue.concurrency && !res.queue.running.includes(res.id)
        ? "Added to the queue." : "Starting… the first results appear within a few seconds.");
      await app.refreshRuns();
      app.go(`#watch/${res.id}`);
    } catch (e) { toast(e.message); }
  }

  renderLesson(); renderAlgos(); renderVision(); renderReward(); renderTime(); renderSummary(); replan();
  api.get("/api/catalog").then((c) => { app.catalog = c; });
  return { onRuns: renderSummary };
}
