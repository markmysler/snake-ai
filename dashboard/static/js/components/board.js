// One replay board: fetches a whole game from the server, animates it locally.
// Weights can only change between games (each game is fetched with the newest or the frozen
// version), so behaviour never changes mid-game.
import { h } from "../util.js";

const MAX_MOVES_PER_FRAME = 40; // "max" speed
const NEW_BEST_MS = 2500;

export class Board {
  // opts: {run, agent, title, color, small, waitingText, rank: () => int, flashNewBest}
  constructor(opts) {
    this.o = opts;
    this.lineEl = h("span", { class: "line" });
    this.overlay = h("div", { class: "overlay" });
    this.canvas = h("canvas");
    this.freezeBtn = h("button", { class: "freeze", "aria-pressed": "false",
      title: "Keep playing this policy instead of switching to newer weights", onclick: () => this.setFrozen(!this.frozen) },
      opts.small ? "❄" : "Freeze");
    this.el = h("div", { class: "board" + (opts.small ? " small" : ""), "--series": opts.color },
      h("div", { class: "board-head" }, h("span", { class: "swatch" }), h("span", { class: "board-title" }, opts.title || ""),
        h("span", { class: "badge" }, "new best")),
      h("div", { class: "canvas-wrap" }, this.canvas, this.overlay),
      h("div", { class: "board-foot" }, this.lineEl, h("span", { class: "buttons" },
        h("button", { title: "Start a new game now", onclick: () => this.restart() }, opts.small ? "↻" : "Restart"),
        this.freezeBtn)));
    this.ctx = this.canvas.getContext("2d");
    this.game = null; this.pending = null; this.loading = false; this.token = 0;
    this.version = null; this.meta = null; this.frozen = false; this.frozenRank = null;
    this.retryAt = 0; this.holdUntil = 0; this.acc = 0; this.bestReplay = 0;
    this.setOverlay(opts.waitingText || "Waiting for a model…");
  }

  dispose() { this.token++; clearTimeout(this.flashTimer); }

  setFrozen(on) {
    if (this.pending || this.loading) { this.token++; this.pending = null; this.loading = false; } // drop prefetch
    this.frozen = on && this.version !== null;
    this.frozenRank = this.frozen && this.meta ? this.meta.rank : null;
    this.freezeBtn.setAttribute("aria-pressed", String(this.frozen));
    this.el.classList.toggle("frozen", this.frozen);
  }

  restart() {
    this.token++; this.game = null; this.pending = null; this.loading = false;
    this.retryAt = 0; this.holdUntil = 0;
  }

  async load() {
    const token = ++this.token;
    this.loading = true;
    const pinned = this.frozen ? this.version : null;
    const p = new URLSearchParams({ run: this.o.run, agent: this.o.agent });
    if (this.o.rank) p.set("rank", this.frozen ? this.frozenRank : this.o.rank());
    if (pinned) p.set("version", pinned);
    let data = null;
    try {
      const res = await fetch("/api/game?" + p);
      if (res.ok) data = await res.json();
    } catch (e) { /* retried below */ }
    if (token !== this.token) return; // restarted or removed meanwhile
    this.loading = false;
    if (!data || data.waiting) {
      this.retryAt = performance.now() + 1500;
      if (!this.game) this.setOverlay(this.o.waitingText || "Waiting for a model…");
      return;
    }
    this.pending = { ...data, pinned };
  }

  start({ game, meta, pinned }) {
    if (pinned && meta.version !== pinned) this.setFrozen(false); // pinned version no longer on the server
    const isNew = this.o.flashNewBest && this.version !== null && meta.version !== this.version;
    this.version = meta.version; this.meta = meta;
    this.game = game; this.frame = 0; this.acc = 0;
    this.snake = game.init.slice(); this.food = game.food; this.score = 0;
    this.eats = new Map(game.eats.map(([f, c]) => [f, c]));
    this.setOverlay(null);
    if (isNew) {
      this.el.classList.remove("flash"); void this.el.offsetWidth; this.el.classList.add("flash");
      clearTimeout(this.flashTimer);
      this.flashTimer = setTimeout(() => this.el.classList.remove("flash"), NEW_BEST_MS);
    }
    this.draw();
  }

  // Called every animation frame while replays run. mps = moves/sec, or Infinity for "max".
  update(now, dt, mps) {
    const g = this.game;
    if (g && this.frame < g.heads.length) {
      let n;
      if (mps === Infinity) n = MAX_MOVES_PER_FRAME;
      else { this.acc += (dt / 1000) * mps; n = Math.floor(this.acc); this.acc -= n; }
      if (n > 0) {
        this.advance(Math.min(n, g.heads.length - this.frame));
        if (this.frame >= g.heads.length) this.finish(now, mps);
        this.draw();
      }
      return;
    }
    if (!this.pending && !this.loading && now >= this.retryAt) this.load();
    if (this.pending && now >= this.holdUntil) { this.start(this.pending); this.pending = null; }
  }

  advance(n) {
    const heads = this.game.heads;
    for (let i = 0; i < n; i++) {
      this.frame++;
      this.snake.unshift(heads[this.frame - 1]);
      if (this.eats.has(this.frame)) { this.food = this.eats.get(this.frame); this.score++; }
      else this.snake.pop();
    }
  }

  finish(now, mps) {
    this.bestReplay = Math.max(this.bestReplay, this.score);
    this.holdUntil = now + (mps >= 400 ? 150 : 800);
    const why = { crash: "crashed", stall: "starved", win: "board full!", limit: "step limit" }[this.game.end];
    this.setOverlay(`${why} · ${this.score}`, true);
  }

  setOverlay(text, translucent = false) {
    this.overlay.textContent = text || "";
    this.overlay.hidden = !text;
    this.overlay.classList.toggle("translucent", translucent);
    if (!this.game) this.lineEl.textContent = this.o.small ? "" : "–";
  }

  draw() {
    const g = this.game, ctx = this.ctx;
    const style = getComputedStyle(this.el), css = (k) => style.getPropertyValue(k).trim();
    const dpr = window.devicePixelRatio || 1;
    const px = Math.round((this.canvas.clientWidth || 200) * dpr);
    if (this.canvas.width !== px) { this.canvas.width = px; this.canvas.height = Math.round(px * g.h / g.w); }
    const cell = this.canvas.width / g.w, gap = Math.max(1, cell * 0.08);
    ctx.fillStyle = css("--board-bg");
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    const rect = (c, color, inset = gap) => {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect((c % g.w) * cell + inset, Math.floor(c / g.w) * cell + inset,
        cell - 2 * inset, cell - 2 * inset, cell * 0.2);
      ctx.fill();
    };
    if (this.food >= 0) rect(this.food, css("--food"), cell * 0.22);
    const series = css("--series");
    ctx.globalAlpha = 0.7; // body faded, head solid
    for (let i = this.snake.length - 1; i > 0; i--) rect(this.snake[i], series);
    ctx.globalAlpha = 1;
    rect(this.snake[0], series);
    if (this.frame >= g.heads.length && g.end === "crash") {
      ctx.strokeStyle = css("--crash"); ctx.lineWidth = gap * 2;
      const c = this.snake[0];
      ctx.strokeRect((c % g.w) * cell + gap, Math.floor(c / g.w) * cell + gap, cell - 2 * gap, cell - 2 * gap);
    }
    const m = this.meta;
    const it = m.iteration ? (m.iteration_label === "generation" ? `gen ${m.iteration}` : `ep ${m.iteration}`) : "saved";
    this.lineEl.textContent = this.o.small
      ? `#${m.rank + 1} · ${this.score}`
      : `score ${this.score} · best ${Math.max(this.bestReplay, this.score)} · ${it}`;
  }
}
