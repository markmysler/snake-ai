// Inspect mode: step through one replay game move by move and see what the snake saw (the vision's
// drawing on the board, the raw inputs) and what its network thought of each move (GET /api/frame).
// The browser only draws the primitives the server sends, so every vision explains itself (vision.py).
import { api, cssVar, fmtNum, h, toast } from "../util.js";

const MOVES = ["Left", "Straight", "Right"];  // action order (snake_env: 0 = left, 1 = straight, 2 = right)
const TONES = { danger: "--ink", safe: "--muted", apple: "--food", wall: "--ink", body: "--end-body", info: "--ink-2",
                room0: "--c5", room1: "--c4", room2: "--c3" };
const LEGEND = { danger: "deadly move", safe: "safe move", apple: "apple / its direction", wall: "wall", body: "body",
                 room0: "pocket 1", room1: "pocket 2", room2: "pocket 3" };
const END_TEXT = { wall: "hit a wall", body: "hit itself", starved: "starved", won: "filled the board", limit: "step limit" };
const PLAY_MS = 220;
const PAD = 0.6;  // squares of wall drawn around the board, so what lies just outside it (walls) stays visible

const softmax = (xs) => { const m = Math.max(...xs), e = xs.map((x) => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); };

export class Inspector {
  // opts: {run, agent, title, color, seed, version, step, onClose}
  constructor(opts) {
    this.o = opts; this.step = opts.step || 0; this.frame = null; this.token = 0; this.timer = null;
    this.canvas = h("canvas", { role: "img" });
    this.slider = h("input", { type: "range", min: 0, max: 0, value: 0, "aria-label": "Move", oninput: (e) => this.go(+e.target.value) });
    this.moveLabel = h("span", { class: "move-label" }, "–");
    this.playBtn = h("button", { title: "Play (space)", onclick: () => this.toggle() }, "Play");
    const btn = (label, title, fn) => h("button", { title, "aria-label": title, onclick: fn }, label);
    this.panel = h("div", { class: "insp-panel" });
    this.legend = h("div", { class: "legend insp-legend" });
    this.text = h("p", { class: "small ink2" });
    this.el = h("section", { class: "card inspector", tabindex: "0", "aria-label": `Inspecting ${opts.title}`, "--series": opts.color,
      onkeydown: (e) => this.key(e) },
      h("div", { class: "section-head" }, h("div", {}, h("h2", {}, `Inspect: ${opts.title}`),
        h("p", {}, `One test of its decisions, move by move: game seed ${opts.seed}. Use ← → to step, space to play.`)),
        h("button", { onclick: () => { this.dispose(); opts.onClose?.(); } }, "Close")),
      h("div", { class: "insp-grid" },
        h("div", { class: "insp-board" }, h("div", { class: "canvas-wrap" }, this.canvas), this.legend,
          h("div", { class: "insp-steps" }, btn("⏮", "First move (Home)", () => this.go(0)), btn("◀", "Previous move (←)", () => this.go(this.step - 1)),
            this.playBtn, btn("▶", "Next move (→)", () => this.go(this.step + 1)),
            btn("⏭", "Last move (End)", () => this.go(this.frame ? this.frame.moves : 0))),
          this.slider, this.moveLabel, this.text),
        this.panel));
    this.go(this.step);
  }

  dispose() { this.token++; clearInterval(this.timer); }

  key(e) {
    const map = { ArrowLeft: () => this.go(this.step - 1), ArrowRight: () => this.go(this.step + 1), Home: () => this.go(0),
                  End: () => this.go(this.frame?.moves ?? 0), " ": () => this.toggle() };
    if (map[e.key] && e.target.tagName !== "INPUT") { e.preventDefault(); map[e.key](); }
  }

  toggle() {
    if (this.timer) { clearInterval(this.timer); this.timer = null; this.playBtn.textContent = "Play"; return; }
    if (this.frame && this.step >= this.frame.moves) this.go(0);
    this.playBtn.textContent = "Pause";
    this.timer = setInterval(() => {
      if (!this.frame || this.step >= this.frame.moves) { this.toggle(); return; }
      if (!this.busy) this.go(this.step + 1);
    }, PLAY_MS);
  }

  async go(step) {
    const max = this.frame ? this.frame.moves : Infinity;
    step = Math.max(0, Math.min(step, max));
    const token = ++this.token;
    this.busy = true;
    const p = new URLSearchParams({ run: this.o.run, agent: this.o.agent, seed: this.o.seed, step });
    if (this.o.version) p.set("version", this.o.version);
    try {
      const f = await api.get(`/api/frame?${p}`);
      if (token !== this.token) return;
      this.frame = f; this.step = f.step;
      this.render();
    } catch (e) {
      if (token === this.token) { toast(e.message); if (this.timer) this.toggle(); }
    } finally { if (token === this.token) this.busy = false; }
  }

  render() {
    const f = this.frame;
    this.slider.max = f.moves; this.slider.value = f.step;
    this.moveLabel.textContent = f.end ? `End after ${f.moves} moves: ${END_TEXT[f.end] || f.end} · score ${f.score}`
      : `Move ${f.step + 1} of ${f.moves} · score ${f.score}`;
    this.text.textContent = f.explain.text;
    this.draw();
    const tones = new Set([...f.explain.lines, ...f.explain.marks, ...f.explain.cells].map((p) => p.tone).filter((t) => LEGEND[t]));
    this.legend.replaceChildren(...[...tones].sort().map((t) => h("span", { "--series": `var(${TONES[t]})` },
      h("i", { class: t.startsWith("room") ? "key cell" : "key" }), LEGEND[t])));
    this.panel.replaceChildren(this.outputs(f), this.inputs(f));
  }

  // The network's output for each move, as a share (softmax) with the raw value beside it.
  outputs(f) {
    const q = f.info.outputs === "q", probs = f.info.outputs === "probs";
    const title = q ? "Q-values: expected future reward of each move" : probs ? "jev's probability for each move"
      : "Network outputs: the highest one is the move";
    if (!f.out) return h("div", { class: "insp-block" }, h("h3", {}, title), h("p", { class: "muted small" }, "The game is over: no decision to show."));
    const share = softmax(f.out);
    return h("div", { class: "insp-block" }, h("h3", {}, title),
      h("div", { class: "prefs", role: "table", "aria-label": "Move preferences" }, MOVES.map((m, i) => h("div", { class: "pref" + (i === f.action ? " chosen" : ""), role: "row" },
        h("span", { role: "cell", class: "pref-name" }, m, i === f.action ? h("b", {}, " ← chosen") : null),
        h("span", { role: "cell", class: "pref-bar" }, h("i", { style: { width: `${Math.max(1, share[i] * 100)}%` } })),
        h("span", { role: "cell", class: "pref-val" }, `${Math.round(share[i] * 100)}%`, probs ? null : h("small", {}, ` (${fmtNum(f.out[i], 2)})`))))),
      h("p", { class: "small muted" }, probs ? "jev answers with a probability for each move and plays the one it picked; "
        + "these are its own numbers, recorded when it played." : "Bars are the softmax of the raw values in brackets: a rough "
        + "measure of how sure it is. " + (q ? "DQN picks the move with the highest Q-value." : "The GA's network picks its highest output.")));
  }

  // The numbers the network read: a named list for small visions, the pictures for window / full.
  inputs(f) {
    const { names, picture } = f.info;
    if (names) {
      return h("div", { class: "insp-block" }, h("h3", {}, `Inputs (${f.inputs.length})`),
        h("div", { class: "table-scroll inputs" }, h("table", {}, h("tbody", {}, names.map((n, i) => h("tr", { class: f.inputs[i] ? "on" : "" },
          h("th", {}, n), h("td", { class: "r" }, fmtNum(f.inputs[i], 2)),
          h("td", { class: "ibar" }, h("i", { style: { width: `${Math.min(1, Math.abs(f.inputs[i])) * 100}%` } }))))))));
    }
    if (picture) {
      const n = picture.side * picture.side;
      const pics = picture.channels.map((label, c) => {
        const cv = h("canvas", { class: "pic", width: picture.side, height: picture.side, role: "img", "aria-label": label });
        const ctx = cv.getContext("2d"), ink = cssVar("--ink");
        ctx.fillStyle = cssVar("--board-bg"); ctx.fillRect(0, 0, picture.side, picture.side);
        ctx.fillStyle = ink;
        for (let i = 0; i < n; i++) {
          const v = f.inputs[c * n + i];
          if (v > 0) { ctx.globalAlpha = Math.min(1, v); ctx.fillRect(i % picture.side, Math.floor(i / picture.side), 1, 1); }
        }
        return h("figure", {}, cv, h("figcaption", {}, label));
      });
      const extra = f.inputs.slice(picture.channels.length * n);
      return h("div", { class: "insp-block" }, h("h3", {}, `Inputs: ${picture.channels.length} pictures of ${picture.side}×${picture.side}`
        + (extra.length ? ` + ${extra.length} apple flags` : "")),
        h("div", { class: "pics" }, pics),
        extra.length ? h("p", { class: "small ink2" }, picture.extra.map((nm, i) => `${nm}: ${extra[i] ? "yes" : "no"}`).join(" · ")) : null,
        h("p", { class: "small muted" }, "Exactly what the network gets: centred on the head and turned so the snake faces up. "
          + "Stronger colour = higher value."));
    }
    return null;
  }

  draw() {
    const f = this.frame, g = { w: f.info.width, h: f.info.height }, ctx = this.canvas.getContext("2d");
    const style = getComputedStyle(this.el), css = (k) => style.getPropertyValue(k).trim();
    const dpr = window.devicePixelRatio || 1, px = Math.round((this.canvas.clientWidth || 360) * dpr);
    if (this.canvas.width !== px) { this.canvas.width = px; this.canvas.height = Math.round(px * (g.h + 2 * PAD) / (g.w + 2 * PAD)); }
    const cell = this.canvas.width / (g.w + 2 * PAD), gap = Math.max(1, cell * 0.08);
    const X = (x) => x * cell, tone = (t) => css(TONES[t] || "--ink-2");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = css("--axis"); ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);  // the wall
    ctx.translate(PAD * cell, PAD * cell);
    ctx.fillStyle = css("--board-bg"); ctx.fillRect(0, 0, g.w * cell, g.h * cell);
    const rect = ([x, y], color, inset = gap) => {
      ctx.fillStyle = color; ctx.beginPath();
      ctx.roundRect(x * cell + inset, y * cell + inset, cell - 2 * inset, cell - 2 * inset, cell * 0.2); ctx.fill();
    };
    const ex = f.explain;
    ctx.globalAlpha = 0.2;
    for (const c of ex.cells) { ctx.fillStyle = tone(c.tone); ctx.fillRect(c.x * cell, c.y * cell, cell, cell); }
    ctx.globalAlpha = 1;
    if (f.food) rect(f.food, css("--food"), cell * 0.22);
    const series = css("--series");
    ctx.globalAlpha = 0.55;
    for (let i = f.snake.length - 1; i > 0; i--) rect(f.snake[i], series);
    ctx.globalAlpha = 1;
    rect(f.snake[0], series);
    if (f.end === "wall" || f.end === "body") {
      ctx.strokeStyle = css("--crash"); ctx.lineWidth = gap * 2;
      ctx.strokeRect(f.snake[0][0] * cell + gap, f.snake[0][1] * cell + gap, cell - 2 * gap, cell - 2 * gap);
    }
    for (const b of ex.boxes) {
      ctx.strokeStyle = tone(b.tone); ctx.lineWidth = 2; ctx.setLineDash([]);
      ctx.strokeRect(X(b.x0) + 1, X(b.y0) + 1, X(b.x1 - b.x0) - 2, X(b.y1 - b.y0) - 2);
    }
    ctx.lineCap = "round";
    for (const l of ex.lines) {
      const [x0, y0] = l.from.map(X), [x1, y1] = l.to.map(X), c = tone(l.tone);
      ctx.strokeStyle = c; ctx.fillStyle = c;
      ctx.lineWidth = l.tone === "danger" ? 3 : 2; ctx.globalAlpha = l.tone === "info" ? 0.55 : 1;
      ctx.setLineDash(l.dash ? [cell * 0.12, cell * 0.12] : []);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      ctx.setLineDash([]);
      if (l.arrow) {
        const a = Math.atan2(y1 - y0, x1 - x0), s = cell * 0.2;
        ctx.beginPath(); ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - s * Math.cos(a - 0.5), y1 - s * Math.sin(a - 0.5));
        ctx.lineTo(x1 - s * Math.cos(a + 0.5), y1 - s * Math.sin(a + 0.5)); ctx.closePath(); ctx.fill();
      }
      if (l.tone === "danger") {  // a cross on deadly squares: never colour alone
        const s = cell * 0.16, cx = x1, cy = y1;
        ctx.lineWidth = 3; ctx.beginPath();
        ctx.moveTo(cx - s, cy - s); ctx.lineTo(cx + s, cy + s); ctx.moveTo(cx + s, cy - s); ctx.lineTo(cx - s, cy + s); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
    ctx.font = `600 ${Math.max(10, cell * 0.24)}px system-ui, sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    for (const m of ex.marks) {
      const [x, y] = m.at.map(X), r = Math.max(4, cell * 0.13);
      ctx.fillStyle = tone(m.tone); ctx.strokeStyle = css("--board-bg"); ctx.lineWidth = 2;
      ctx.beginPath();
      if (m.tone === "wall") ctx.rect(x - r, y - r, 2 * r, 2 * r); else ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill(); ctx.stroke();
      if (m.label) {
        ctx.lineWidth = 3; ctx.strokeText(m.label, x, y - r - 1);
        ctx.fillStyle = css("--ink"); ctx.fillText(m.label, x, y - r - 1);
      }
    }
    this.canvas.setAttribute("aria-label", `${f.explain.text} Snake head at ${f.snake[0].join(",")}, score ${f.score}.`);
  }
}
