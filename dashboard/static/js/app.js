// App shell: tabs (hash routes), shared state and the server-sent-events connection.
//   #runs   #new[/<run id to copy>]   #watch[/<run id>]   #compare[/<id>,<id>,<id>]   #best   #ideas[/<idea id>]
//   #learn[/<term id>]
import { setGlossary } from "./components/glossary.js";
import { api, fmtMB, q, SITE } from "./util.js";
import * as bestView from "./views/best.js";
import * as compareView from "./views/compare.js";
import * as ideasView from "./views/ideas.js";
import * as learnView from "./views/learn.js";
import * as newView from "./views/newrun.js";
import * as runsView from "./views/runs.js";
import * as watchView from "./views/watch.js";

const VIEWS = { runs: runsView, new: newView, watch: watchView, compare: compareView, best: bestView, ideas: ideasView, learn: learnView };
const MAX_COMPARE = 3; // the palette's colors stay distinguishable for everyone up to 3 series

function loadCompare() {
  try { return JSON.parse(localStorage.getItem("compare") || "[]").slice(0, MAX_COMPARE); } catch (e) { return []; }
}

export const app = {
  catalog: null, runs: [], queue: { running: [], queued: [], concurrency: 1 }, system: null,
  compareIds: loadCompare(), maxCompare: MAX_COMPARE, view: null, es: null, esRun: undefined,

  run(id) { return this.runs.find((r) => r.id === id); },
  go(hash) { if (location.hash !== hash) location.hash = hash; else render(); },
  setCompare(ids) {
    this.compareIds = [...new Set(ids)].slice(0, MAX_COMPARE);
    try { localStorage.setItem("compare", JSON.stringify(this.compareIds)); } catch (e) { /* private mode */ }
    updateChrome();
  },
  toggleCompare(id) {
    const ids = this.compareIds.includes(id) ? this.compareIds.filter((x) => x !== id) : [...this.compareIds, id];
    this.setCompare(ids);
  },
  async refreshRuns() {
    const d = await api.get("/api/runs");
    this.runs = d.runs; this.queue = d.queue;
    updateChrome();
    this.view?.onRuns?.();
  },

  // One SSE connection; it also streams one run's live data when `runId` is given.
  connect(runId) {
    if (this.es && this.esRun === runId) return;
    this.es?.close();
    this.esRun = runId;
    const path = "/api/events" + (runId ? `?run=${q(runId)}` : "");
    const handlers = {};
    const on = (ev, fn) => { handlers[ev] = fn; this.es.addEventListener?.(ev, (e) => fn(JSON.parse(e.data))); };
    if (SITE) {  // the static demo: everything is finished, so the first message is the only one
      this.es = { close() { this.closed = true; } };
      const es = this.es;
      api.get(path).then((d) => { if (!es.closed) handlers.init(d); });
    } else this.es = new EventSource(path);
    on("init", (d) => {
      this.runs = d.runs; this.queue = d.queue; this.system = d.system;
      updateChrome();
      this.view?.onRuns?.();
      if (d.watch) this.view?.onWatch?.(d.watch);
    });
    on("runs", (d) => { this.runs = d.runs; this.queue = d.queue; updateChrome(); this.view?.onRuns?.(); });
    on("update", (d) => this.view?.onUpdate?.(d));
    on("watch", (d) => this.view?.onWatch?.(d));
  },
};

function updateChrome() {
  const s = app.system;
  if (s) {
    const ram = s.ram_total_mb ? ` · ${fmtMB(s.ram_total_mb)} RAM (${fmtMB(s.ram_available_mb)} free)` : "";
    document.getElementById("machine").textContent =
      `${s.cpus} CPUs${ram}${s.on_battery ? " · on battery" : ""} · ${app.queue.concurrency} run(s) at a time`;
  }
  const live = app.runs.filter((r) => r.status === "running").length;
  const queued = app.runs.filter((r) => r.status === "queued").length;
  document.getElementById("count-runs").textContent = app.runs.length || "";
  document.getElementById("count-live").textContent = live ? `${live} live` + (queued ? ` · ${queued} queued` : "") : "";
  document.getElementById("count-compare").textContent = app.compareIds.length || "";
  document.getElementById("count-ideas").textContent = app.queue.ideas_open || "";
}

function parseHash() {
  const [tab, ...rest] = location.hash.replace(/^#/, "").split("/");
  return { tab: VIEWS[tab] ? tab : "runs", param: decodeURIComponent(rest.join("/")) };
}

function render() {
  const { tab, param } = parseHash();
  if (app.catalog?.read_only && tab === "new") { location.hash = "#runs"; return; }  // nothing can be launched
  app.view?.unmount?.();
  const main = document.getElementById("main");
  main.replaceChildren();
  for (const a of document.querySelectorAll(".tabs a")) {
    a.toggleAttribute("aria-current", a.dataset.tab === tab);
    if (a.dataset.tab === tab) {
      a.setAttribute("aria-current", "page");
      a.scrollIntoView({ block: "nearest", inline: "nearest" });  // narrow screens scroll the tab row
    }
  }
  app.view = VIEWS[tab].mount(main, param, app) || {};
  if (!app.view.ownsConnection) app.connect(null);
  window.scrollTo(0, 0);
}

window.addEventListener("hashchange", render);
(async () => {
  app.catalog = await api.get("/api/catalog");
  setGlossary(app.catalog.glossary);
  if (app.catalog.read_only) {  // a shared demo: every button that changes something is hidden (data-write)
    document.body.classList.add("read-only");
    document.getElementById("demo-badge").hidden = false;
  }
  render();
})();
window.snakeLab = app; // handy in devtools
