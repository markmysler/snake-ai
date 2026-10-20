// 16 small boards, each replaying a different genome of a GA agent's latest generation.
import { h } from "../util.js";
import { Board } from "./board.js";

const N_TILES = 16;

export class Population {
  constructor(run, agent, color) {
    this.size = 200; this.ranks = [];
    this.select = h("select", { "aria-label": "Which genomes",
      onchange: () => { this.pickRanks(); this.tiles.forEach((t) => !t.frozen && t.restart()); } },
      h("option", { value: "top" }, "Top 16"), h("option", { value: "random" }, "Random 16"),
      h("option", { value: "bottom" }, "Bottom 16"));
    this.title = h("span", { class: "muted small" }, "waiting for the first generation");
    this.tiles = Array.from({ length: N_TILES }, (_, i) => new Board({
      run, agent, small: true, color, waitingText: "Waiting…", rank: () => this.ranks[i] }));
    this.el = h("div", {},
      h("div", { class: "section-head" }, this.title, h("label", { class: "controls" }, "Show", this.select)),
      h("div", { class: "pop-grid" }, this.tiles.map((t) => t.el)));
    this.pickRanks();
  }

  pickRanks() {
    const n = Math.min(N_TILES, this.size), mode = this.select.value;
    if (mode === "top") this.ranks = Array.from({ length: n }, (_, i) => i);
    else if (mode === "bottom") this.ranks = Array.from({ length: n }, (_, i) => this.size - n + i);
    else {
      const all = Array.from({ length: this.size }, (_, i) => i);
      for (let i = all.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [all[i], all[j]] = [all[j], all[i]]; }
      this.ranks = all.slice(0, n).sort((a, b) => a - b); // fixed until picked again
    }
  }

  // A new generation was published: tiles switch to it at their next game.
  set(meta) {
    if (!meta) return;
    if (meta.size !== this.size) { this.size = meta.size; this.pickRanks(); }
    this.title.textContent = `generation ${meta.generation} · ${meta.size} genomes ranked by fitness`;
  }
}
