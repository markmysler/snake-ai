// The glossary (glossary.py, sent with the catalog): "?" buttons that explain an option in place,
// entry cards for the Learn tab, and finding which terms a piece of text uses.
import { h, openDialog } from "../util.js";

let byId = new Map(), matchers = [];

export function setGlossary(entries) {
  byId = new Map((entries || []).map((e) => [e.id, e]));
  const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  matchers = (entries || []).flatMap((e) => e.aliases.map((a) => ({ e, re: new RegExp(`(^|[^\\w-])${esc(a)}(?![\\w-])`, "i") })));
}
export const entry = (id) => byId.get(id);

// Entries mentioned in the texts, in order of first appearance across them (most relevant first).
export function termsIn(texts, limit = 6) {
  const found = [];
  for (const t of texts.filter(Boolean)) {
    const hits = matchers.map((m) => ({ m, i: t.search(m.re) })).filter((x) => x.i >= 0).sort((a, b) => a.i - b.i);
    for (const { m } of hits) if (!found.includes(m.e)) found.push(m.e);
  }
  return found.slice(0, limit);
}

const KIND = { algorithm: "algorithm", setting: "setting", vision: "vision", reward: "reward feature",
               preset: "reward preset", concept: "concept" };

function range(e) {
  if (e.kind !== "setting" && e.kind !== "reward") return null;
  const bits = [];
  if (e.default !== undefined) bits.push(`default ${e.default}`);
  if (e.min !== undefined && e.max !== undefined) bits.push(`range ${e.min}–${e.max}`);
  if (e.when) bits.push(e.when);
  return bits.length ? h("span", { class: "small muted" }, bits.join(" · ")) : null;
}

export function entryEl(e, { link = true, heading = "h3" } = {}) {
  const parent = e.group ? entry(e.group) : null;
  return h("article", { class: "gl-entry", id: `g-${e.id}` },
    h("div", { class: "gl-head" }, h(heading, {}, e.term), h("span", { class: "chip" }, parent ? `${parent.term.split(" · ")[0]} ${KIND[e.kind]}` : KIND[e.kind]), range(e)),
    h("p", { class: "gl-short" }, e.short),
    e.long ? h("p", { class: "ink2" }, e.long) : null,
    e.up || e.down ? h("div", { class: "gl-effects" },
      e.up ? h("div", {}, h("b", {}, "↑ Raise it: "), e.up) : null,
      e.down ? h("div", {}, h("b", {}, "↓ Lower it: "), e.down) : null) : null,
    e.incompatible?.length ? h("p", { class: "small" }, "Can't be combined with ",
      e.incompatible.map((id) => entry(id)?.term || id).join(", "), ".") : null,
    link && e.related.length ? h("div", { class: "chips" }, h("span", { class: "small muted" }, "See also"),
      e.related.map((id) => termChip(id))) : null);
}

// A chip that explains a term in a dialog.
export function termChip(id) {
  const e = entry(id);
  if (!e) return null;
  return h("button", { class: "chip term", title: e.short, onclick: (ev) => { ev.preventDefault(); ev.stopPropagation(); explain(id); } }, e.term);
}

// A small "?" next to an option.
export function info(id, label) {
  const e = entry(id);
  if (!e) return null;
  return h("button", { class: "info", type: "button", "aria-label": `What is ${label || e.term}?`, title: e.short,
    onclick: (ev) => { ev.preventDefault(); ev.stopPropagation(); explain(id); } }, "?");
}

export function explain(id) {
  const e = entry(id);
  if (!e) return;
  openDialog(entryEl(e, { heading: "h2" }), h("div", { class: "btn-row" },
    h("a", { class: "button", href: `#learn/${e.id}`, onclick: () => document.getElementById("dialog").close() }, "Open in Learn"),
    h("button", { onclick: () => document.getElementById("dialog").close() }, "Close")));
}

// "Terms in this…" row: the glossary entries a piece of text uses, as clickable chips.
export function termsRow(texts, label = "Terms") {
  const found = termsIn(texts);
  return found.length ? h("div", { class: "chips terms" }, h("span", { class: "small muted" }, label),
    found.map((e) => termChip(e.id))) : null;
}
