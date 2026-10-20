// Learn: the glossary. Every algorithm, setting, vision and reward feature (from the registries) and the
// words the dashboard and the AI analysis use, with what happens when you raise or lower a setting.
import { entryEl } from "../components/glossary.js";
import { h } from "../util.js";

const FILTERS = { all: "All", algorithm: "Algorithms", vision: "Vision", reward: "Rewards", concept: "Concepts" };
const view = { query: "", filter: "all" };

export function mount(main, focusId, app) {
  const entries = app.catalog.glossary || [];
  const root = h("div", { class: "view" });
  const list = h("div", { class: "gl-sections" });
  main.append(root);
  if (focusId) { view.query = ""; view.filter = "all"; }

  const matches = (e) => {
    const qy = view.query.trim().toLowerCase();
    return !qy || [e.term, e.short, e.long, ...e.aliases].join(" ").toLowerCase().includes(qy);
  };
  const childrenOf = (id) => entries.filter((e) => e.group === id);

  function sections() {
    const top = (kind) => entries.filter((e) => e.kind === kind);
    return [
      ["algorithm", "Algorithms and their settings", "How each method learns, and what every setting in New run does.",
        top("algorithm").flatMap((e) => [e, ...childrenOf(e.id)])],
      ["vision", "Vision", "What the snake's network gets to see each move.",
        top("vision").flatMap((e) => [e, ...childrenOf(e.id)])],
      ["reward", "Rewards", "The signals the snake learns from. Combine features, or start from a preset.",
        [...top("reward"), ...top("preset")]],
      ["concept", "Concepts", "Words used on the charts and in the AI analysis.",
        [...top("concept")].sort((a, b) => a.term.localeCompare(b.term))],
    ];
  }

  function renderList() {
    const secs = sections().filter(([kind]) => view.filter === "all" || view.filter === kind)
      .map(([kind, title, sub, es]) => [kind, title, sub, es.filter(matches)]).filter(([, , , es]) => es.length);
    list.replaceChildren(...(secs.length ? secs.map(([kind, title, sub, es]) => h("section", { class: "card" },
      h("div", { class: "section-head" }, h("div", {}, h("h2", {}, title), h("p", {}, sub))),
      h("div", { class: "gl-list" }, es.map((e) => {
        const el = entryEl(e);
        if (e.group) el.classList.add("child");
        return el;
      }))))
      : [h("div", { class: "card empty" }, h("strong", {}, "No matching terms"), "Try a shorter word.")]));
  }

  const search = h("input", { type: "search", placeholder: "Search, e.g. gamma, survival, rays", value: view.query,
    "aria-label": "Search the glossary", oninput: (e) => { view.query = e.target.value; renderList(); } });
  const filters = h("div", { class: "segmented", role: "group", "aria-label": "Show" });
  const renderFilters = () => filters.replaceChildren(...Object.entries(FILTERS).map(([k, label]) => h("button", {
    "aria-pressed": String(view.filter === k), onclick: () => { view.filter = k; renderFilters(); renderList(); } }, label)));
  renderFilters();

  root.append(
    h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Learn"),
      h("p", { class: "lead" }, "What every option in New run does and what the words in the charts and the AI analysis mean. "
        + "Settings say what happens when you raise or lower them. The “?” next to any option opens its entry."))),
    h("div", { class: "toolbar" }, search, filters), list);
  renderList();

  if (focusId) {
    requestAnimationFrame(() => {
      const el = document.getElementById(`g-${focusId}`);
      if (el) { el.scrollIntoView({ block: "center" }); el.classList.add("focus"); }
    });
  }
  return {};
}
