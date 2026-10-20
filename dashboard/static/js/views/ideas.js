// Ideas: improvements the AI analyses proposed that need new code (new visions, rewards, algorithms,
// settings, tools). They pile up from every analysed run; a review asks the LLM which to build next.
import { termsRow } from "../components/glossary.js";
import { api, fmtDate, h, markdown, q, toast } from "../util.js";

const FILTERS = { open: "Open", planned: "Planned", done: "Done", dismissed: "Dismissed", all: "All" };
const OPEN = ["new", "reviewed", "planned"];
const view = { filter: "open", review: null };

export function mount(main, focusId, app) {
  const root = h("div", { class: "view" });
  main.append(root);
  let data = null, poll = null;

  async function load() {
    try {
      data = await api.get("/api/ideas" + (view.review ? `?review=${q(view.review)}` : ""));
    } catch (e) { toast(e.message); return; }
    render();
    clearTimeout(poll);
    if (data.review?.state === "running") poll = setTimeout(load, 2000);
  }

  async function startReview() {
    try {
      await api.post("/api/ideas/review");
      view.review = null;
      await load();
    } catch (e) { toast(e.message); }
  }

  async function setStatus(idea, status) {
    try {
      await api.patch(`/api/idea?id=${q(idea.id)}`, { status });
      idea.status = status;
      render();
      app.refreshRuns();  // updates the tab's count
    } catch (e) { toast(e.message); }
  }

  const ideaChip = (id) => h("a", { class: "chip", href: `#ideas/${id}`, onclick: (e) => {
    e.preventDefault();
    const el = document.getElementById(`idea-${id}`);
    if (el) { el.scrollIntoView({ block: "center" }); el.classList.add("focus"); setTimeout(() => el.classList.remove("focus"), 2000); }
  } }, id);

  function reviewCard() {
    const rev = data.review, open = data.ideas.filter((i) => OPEN.includes(i.status));
    const unreviewed = open.filter((i) => i.status === "new").length;
    const running = rev?.state === "running";
    const button = h("button", { class: unreviewed ? "primary" : "", disabled: running || !open.length || !data.llm.configured,
      title: data.llm.configured ? `Uses ${data.llm.model}` : "Add an API key to .env to enable reviews", onclick: startReview },
      running ? "Reviewing…" : `Review ${open.length} open idea${open.length === 1 ? "" : "s"}`);
    const picker = data.reviews.length > 1 ? h("select", { "aria-label": "Earlier reviews", onchange: (e) => { view.review = e.target.value || null; load(); } },
      data.reviews.map((r, i) => h("option", { value: i ? r.name : "", selected: (view.review || data.reviews[0].name) === r.name },
        `${fmtDate(r.created)} · ${r.ideas} ideas${i ? "" : " (latest)"}`))) : null;
    const done = rev?.state === "done" ? rev : null;
    const card = h("section", { class: "card analysis" },
      h("div", { class: "section-head" }, h("div", {}, h("h2", {}, "What to build next"),
        h("p", {}, done ? `Review by ${done.model} · ${fmtDate(done.created)} · ${done.ideas_reviewed} ideas`
          : "An LLM merges duplicate ideas, weighs them against the results so far and ranks what to build.")),
        h("div", { class: "btn-row" }, picker, button)),
      unreviewed && done ? h("p", { class: "small ink2" }, `${unreviewed} idea${unreviewed === 1 ? " has" : "s have"} arrived since this review.`) : null);
    const add = (...els) => card.append(...els.flat().filter(Boolean));
    if (running) add(h("p", { class: "ink2 working" }, h("i", { class: "dot" }), `${rev.model || "The model"} is reviewing the ideas… usually under a minute.`));
    if (rev?.state === "error") add(h("div", { class: "msg error" }, rev.error || "The review failed."));
    if (done?.last_error) add(h("div", { class: "msg warn" }, `Reviewing again failed (${done.last_error}). This is the earlier review.`));
    if (!done) {
      if (!running && rev?.state !== "error") add(h("p", { class: "muted" }, open.length
        ? "No review yet. Let a few runs' ideas pile up, then press Review."
        : "No ideas yet. They appear here when an AI analysis proposes something the lab can't do yet."));
      return card;
    }
    const r = done.result;
    add(markdown(r.summary),
      r.build_next.length ? h("div", {}, h("h3", { class: "try-next" }, "Build next"),
        h("div", { class: "recs" }, r.build_next.map((b, i) => h("article", { class: "rec" },
          h("div", { class: "rec-head" }, h("span", { class: "n" }, i + 1), h("h3", {}, b.title)),
          h("div", { class: "chips" }, h("span", { class: "chip" }, b.kind), h("span", { class: "chip" }, `${b.effort} effort`)),
          h("p", {}, b.what), h("p", { class: "small ink2" }, h("b", {}, "Why: "), b.why),
          b.how ? h("p", { class: "small ink2" }, h("b", {}, "How: "), b.how) : null,
          b.first_experiment ? h("p", { class: "small ink2" }, h("b", {}, "First experiment: "), b.first_experiment) : null,
          termsRow([b.what, b.why, b.first_experiment], "Terms"),
          b.ideas.length ? h("div", { class: "chips" }, h("span", { class: "small muted" }, "Covers"), b.ideas.map(ideaChip)) : null)))) : null,
      r.later.length || r.drop.length ? h("div", { class: "later" },
        r.later.length ? h("div", {}, h("h3", {}, "Later"), h("ul", {}, r.later.map((x) =>
          h("li", {}, h("b", {}, x.title), ` — ${x.reason} `, x.ideas.map(ideaChip))))) : null,
        r.drop.length ? h("div", {}, h("h3", {}, "Not worth it"), h("ul", {}, r.drop.map((x) =>
          h("li", {}, x.ideas.map(ideaChip), ` ${x.reason}`)))) : null) : null,
      h("p", { class: "small muted" }, "Mark an idea Planned or Done below once you decide; reviews only look at open ideas."));
    return card;
  }

  function list() {
    const shown = data.ideas.filter((i) => view.filter === "all" || (view.filter === "open" ? OPEN.includes(i.status) : i.status === view.filter))
      .slice().reverse();
    const seg = h("div", { class: "segmented", role: "group", "aria-label": "Show" }, Object.entries(FILTERS).map(([k, label]) =>
      h("button", { "aria-pressed": String(view.filter === k), onclick: () => { view.filter = k; render(); } },
        label, k !== "all" ? ` ${data.ideas.filter((i) => k === "open" ? OPEN.includes(i.status) : i.status === k).length}` : "")));
    return h("section", { class: "card" },
      h("div", { class: "section-head" }, h("div", {}, h("h2", {}, "All ideas"),
        h("p", {}, "Collected from every AI analysis. Recipe changes the lab can already run are not here: they are the "
          + "“Next runs” of each analysis.")), seg),
      shown.length ? h("div", { class: "idea-list" }, shown.map((i) => h("article", { class: "idea", id: `idea-${i.id}` },
        h("div", { class: "idea-head" },
          h("span", { class: "muted small" }, i.id), h("h3", {}, i.title),
          h("span", { class: "chip" }, i.kind), h("span", { class: "chip" }, `${i.effort} effort`),
          i.status === "new" ? h("span", { class: "pill running" }, "new") : null,
          h("select", { "aria-label": `Status of ${i.id}`, onchange: (e) => setStatus(i, e.target.value) },
            data.statuses.map((s) => h("option", { value: s, selected: s === i.status }, s)))),
        i.what ? h("p", {}, i.what) : null,
        i.why ? h("p", { class: "small ink2" }, h("b", {}, "Why: "), i.why) : null,
        i.expected ? h("p", { class: "small ink2" }, h("b", {}, "Expected: "), i.expected) : null,
        i.note ? h("p", { class: "small muted" }, i.note) : null,
        h("p", { class: "small muted" }, i.source.run ? ["From ", h("a", { href: `#watch/${i.source.run}` }, i.source.run)]
          : "From a run that was deleted", ` · ${fmtDate(i.created)}`,
          i.review ? ` · reviewed ${i.review}` : ""))))
        : h("p", { class: "muted" }, view.filter === "open" ? "No open ideas." : "None here."));
  }

  function render() {
    if (!data) { root.replaceChildren(h("div", { class: "card empty" }, "Loading…")); return; }
    root.replaceChildren(
      h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Ideas"),
        h("p", { class: "lead" }, "Improvements the AI analyses proposed that the lab can't do yet: new visions, rewards, algorithms, "
          + "settings or tools. Let them pile up, then review them to decide what to add to the program. Also saved in ideas/IDEAS.md."))),
      reviewCard(), list());
    if (focusId) {
      const el = document.getElementById(`idea-${focusId}`);
      if (el) { el.scrollIntoView({ block: "center" }); el.classList.add("focus"); }
      focusId = null;
    }
  }

  render();
  load();
  return { unmount() { clearTimeout(poll); } };
}
