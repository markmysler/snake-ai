// How test games ended (a 100% bar) and how much a recipe's test score varies across seeds (a strip plot),
// plus the plain-English verdict on whether two recipes really differ.
import { fmtNum, h } from "../util.js";

// Outcome colors are their own validated set (see style.css --end-*), separate from the agent / run colors.
export const ENDS = [["wall", "hit a wall"], ["body", "hit itself"], ["starved", "starved"], ["won", "filled the board"]];

export function outcomeBar(ev, { labels = true } = {}) {
  if (!ev?.outcomes) return null;
  const total = Object.values(ev.outcomes).reduce((a, b) => a + b, 0) || 1;
  const parts = ENDS.filter(([k]) => ev.outcomes[k] > 0);
  return h("div", { class: "endbar", role: "img",
    "aria-label": "How test games ended: " + parts.map(([k, l]) => `${l} ${Math.round((ev.outcomes[k] / total) * 100)}%`).join(", ") },
    parts.map(([k, label]) => {
      const pct = (ev.outcomes[k] / total) * 100;
      return h("span", { class: `end-${k}`, style: { flexGrow: String(pct) }, title: `${label}: ${ev.outcomes[k]} of ${total} games` },
        labels && pct >= 14 ? `${Math.round(pct)}%` : "");
    }));
}

export function outcomeLegend() {
  return h("div", { class: "legend end-legend" }, ENDS.map(([k, label]) => h("span", {}, h("i", { class: `swatch end-${k}` }), label)));
}

export function outcomeText(ev) {
  if (!ev?.outcomes) return null;
  const total = Object.values(ev.outcomes).reduce((a, b) => a + b, 0) || 1;
  const top = ENDS.map(([k, l]) => [l, ev.outcomes[k] / total]).sort((a, b) => b[1] - a[1])[0];
  return `${Math.round(top[1] * 100)}% ${top[0]}`;
}

// ---- seed groups ----
export function groupMembers(run, runs) {
  if (!run.group) return [run];
  return runs.filter((r) => r.group?.id === run.group.id).sort((a, b) => a.group.index - b.group.index);
}

export function spreadOf(members, agentId) {
  const values = members.filter((m) => m.evaluation?.[agentId]).map((m) => ({ v: m.evaluation[agentId].mean, run: m }));
  if (!values.length) return null;
  const vs = values.map((x) => x.v), n = vs.length, mean = vs.reduce((a, b) => a + b, 0) / n;
  const std = n > 1 ? Math.sqrt(vs.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)) : null;
  return { n, mean, std, min: Math.min(...vs), max: Math.max(...vs), values };
}

export const fmtSpread = (s) => !s ? "–" : s.n > 1 ? `${fmtNum(s.mean, 1)} ± ${fmtNum(s.std, 1)}` : fmtNum(s.mean, 1);

// Does B differ from A beyond luck? Ranges across seeds that don't overlap = a clear difference.
export function verdict(a, b) {
  if (!a || !b) return { kind: "none", text: "No test score to compare." };
  const diff = b.mean - a.mean, word = diff >= 0 ? "higher" : "lower";
  if ((a.n < 2) !== (b.n < 2)) {  // one recipe has a spread: is the single-seed score outside it?
    const [many, one] = a.n > 1 ? [a, b] : [b, a];
    // The single run has luck of its own (assume about as much): only call it real when it sits well outside,
    // more than two standard deviations of the other recipe's seeds from their mean.
    if ((one.mean > many.max || one.mean < many.min) && Math.abs(one.mean - many.mean) > 2 * many.std) {
      return { kind: "clear", text: `${fmtNum(Math.abs(diff), 1)} apples ${word}. The single-seed score is far outside the other recipe's `
        + `range across ${many.n} seeds (${fmtNum(many.min, 1)}–${fmtNum(many.max, 1)}), so it is likely real; more seeds would confirm it.` };
    }
    return { kind: "overlap", text: `${fmtNum(Math.abs(diff), 1)} apples ${word}, but the single-seed score is within or close to the other `
      + `recipe's range across ${many.n} seeds (${fmtNum(many.min, 1)}–${fmtNum(many.max, 1)}): it could be luck.` };
  }
  if (a.n < 2 || b.n < 2) {
    return { kind: "single", text: `${fmtNum(Math.abs(diff), 1)} apples ${word}, but ${a.n < 2 && b.n < 2 ? "both are" : "one is"} a single seed, `
      + "so this can't tell luck from a real effect. Run them with Seeds ≥ 3." };
  }
  if (b.min > a.max || b.max < a.min) {
    return { kind: "clear", text: `${fmtNum(Math.abs(diff), 1)} apples ${word}, and the ranges across seeds don't overlap: a real difference.` };
  }
  return { kind: "overlap", text: `${fmtNum(Math.abs(diff), 1)} apples ${word}, but the ranges across seeds overlap: it could be luck.` };
}

// rows: [{label, color, spread, highlight (run id)}]. One shared 0..max axis.
export function stripPlot(rows, max) {
  max = max || Math.max(1, ...rows.map((r) => r.spread?.max || 0));
  const x = (v) => `${Math.min(100, (v / max) * 100)}%`;
  const ticks = niceTicks(max);
  return h("div", { class: "strip" },
    rows.map((r) => h("div", { class: "strip-row", "--series": r.color },
      h("div", { class: "strip-label" }, r.color ? h("span", { class: "swatch" }) : null, h("span", {}, r.label)),
      h("div", { class: "strip-track" },
        ticks.map((t) => h("i", { class: "grid", style: { left: x(t) } })),
        r.spread && r.spread.n > 1 ? h("i", { class: "range", style: { left: x(r.spread.min), width: `calc(${x(r.spread.max)} - ${x(r.spread.min)})` } }) : null,
        r.spread ? h("i", { class: "mean", style: { left: x(r.spread.mean) }, title: `mean ${fmtNum(r.spread.mean, 2)}` }) : null,
        (r.spread?.values || []).map(({ v, run }) => h("i", { class: "dot" + (run.id === r.highlight ? " me" : ""), style: { left: x(v) },
          title: `${run.id} (seed ${run.config.seed}): ${fmtNum(v, 2)} apples` }))),
      h("div", { class: "strip-value" }, r.spread ? fmtSpread(r.spread) : "–",
        r.spread?.n > 1 ? h("small", {}, `${fmtNum(r.spread.min, 1)}–${fmtNum(r.spread.max, 1)} · ${r.spread.n} seeds`) : h("small", {}, "1 seed")))),
    h("div", { class: "strip-row axis" }, h("div", {}), h("div", { class: "strip-track" },
      ticks.map((t) => h("span", { style: { left: x(t) } }, t))), h("div", { class: "small muted" }, "apples")));
}

function niceTicks(max) {
  const step = [1, 2, 5, 10, 20, 25, 50].find((s) => max / s <= 6) || 100;
  const out = [];
  for (let t = 0; t <= max + 1e-9; t += step) out.push(t);
  return out;
}
