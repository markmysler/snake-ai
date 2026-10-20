// How training games ended over time: a 100% stacked area per agent (wall / body / starved / won shares
// of the last 100 training games at each stats row). Same x-axis modes as the learning-curve chart.
import { cssVar } from "../util.js";
import { X_MODES } from "./linechart.js";
import { ENDS } from "./spread.js";

const END_COL = 7;  // history columns 7..10 = end_wall, end_body, end_starved, end_won (see dashboard/store.py)

export const hasEndings = (rows) => (rows || []).some((r) => r[END_COL] !== null && r[END_COL] !== undefined);

export class EndChart {
  constructor(canvas) {
    this.chart = new Chart(canvas, {
      type: "line",
      data: { datasets: [] },
      options: {
        animation: false, responsive: true, maintainAspectRatio: false, parsing: false, normalized: true,
        interaction: { mode: "nearest", axis: "x", intersect: false },
        elements: { point: { radius: 0, hoverRadius: 3 }, line: { borderWidth: 1, tension: 0 } },
        scales: {
          x: { type: "linear", grid: { color: cssVar("--grid") }, ticks: { color: cssVar("--muted"), maxTicksLimit: 6 } },
          y: { stacked: true, min: 0, max: 100, grid: { color: cssVar("--grid") },
               ticks: { color: cssVar("--muted"), callback: (v) => `${v}%`, stepSize: 25 } },
        },
        plugins: {
          legend: { display: false },
          tooltip: { itemSort: (a, b) => a.datasetIndex - b.datasetIndex,
            callbacks: { label: (c) => ` ${c.dataset.label}: ${Math.round(c.parsed.y)}%` } },
        },
      },
    });
  }

  set(rows, x) {
    const mode = X_MODES[x];
    const pts = (rows || []).filter((r) => r[END_COL] !== null && r[END_COL] !== undefined);
    const colors = { wall: cssVar("--end-wall"), body: cssVar("--end-body"), starved: cssVar("--end-starved"), won: cssVar("--end-won") };
    this.chart.data.datasets = ENDS.map(([k, label], j) => ({
      label, borderColor: colors[k], backgroundColor: colors[k] + "cc", fill: j ? "-1" : "origin", stack: "ends",
      data: pts.map((r) => {
        const total = r[7] + r[8] + r[9] + r[10] || 1;
        return { x: r[mode.col], y: (r[END_COL + j] / total) * 100 };
      }),
    }));
    this.chart.options.scales.x.ticks.callback = (v) => mode.fmt(v);
    this.chart.options.plugins.tooltip.callbacks.title = (items) => items.length ? `${mode.title}: ${mode.fmt(items[0].parsed.x)}` : "";
    this.chart.update();
  }

  destroy() { this.chart.destroy(); }
}
