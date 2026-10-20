// Line chart (Chart.js) used by Watch and Compare: switchable x-axis, crosshair, one tooltip that
// lists every series at the pointer, and direct labels at line ends.
// History rows: [elapsed_s, total_steps, total_games, best_score, mean_score_100, steps_per_sec, iteration]
import { cssVar, fmtCompact, fmtDuration, fmtNum } from "../util.js";

export const X_MODES = {
  time: { col: 0, title: "Wall-clock time", fmt: fmtDuration },
  steps: { col: 1, title: "Env steps (experience)", fmt: fmtCompact },
  games: { col: 2, title: "Games played", fmt: fmtCompact },
};
export const METRICS = {
  mean: { col: 4, title: "Mean score, last 100 games", stepped: false },
  best: { col: 3, title: "Best score in a single game", stepped: true },
};

// For every line, the point closest to the pointer's x (if the pointer is within that line's range).
Chart.Interaction.modes.eachNearestX = function (chart, e) {
  const pos = Chart.helpers.getRelativePosition(e, chart);
  const items = [];
  chart.data.datasets.forEach((ds, di) => {
    const pts = chart.getDatasetMeta(di).data;
    if (!pts.length || chart.getDatasetMeta(di).hidden) return;
    if (pos.x < pts[0].x - 8 || pos.x > pts[pts.length - 1].x + 8) return;
    let lo = 0, hi = pts.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (pts[mid].x < pos.x) lo = mid; else hi = mid; }
    const i = Math.abs(pts[lo].x - pos.x) <= Math.abs(pts[hi].x - pos.x) ? lo : hi;
    items.push({ element: pts[i], datasetIndex: di, index: i });
  });
  return items;
};

const crosshair = {
  id: "crosshair",
  afterEvent(chart, args) {
    const { event } = args, area = chart.chartArea;
    chart.$cross = event.type !== "mouseout" && event.x >= area.left && event.x <= area.right ? event.x : null;
    args.changed = true;
  },
  afterDatasetsDraw(chart) {
    if (chart.$cross == null || !chart.tooltip?.getActiveElements().length) return;
    const { ctx, chartArea: a } = chart;
    ctx.save();
    ctx.strokeStyle = cssVar("--axis"); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(chart.$cross, a.top); ctx.lineTo(chart.$cross, a.bottom); ctx.stroke();
    ctx.restore();
  },
};

const endLabels = {
  id: "endLabels",
  afterDatasetsDraw(chart) {
    if (chart.data.datasets.length > 4) return; // past 4 series the legend does the labelling
    const ctx = chart.ctx, labels = [];
    chart.data.datasets.forEach((ds, i) => {
      const pts = chart.getDatasetMeta(i).data, pt = pts[pts.length - 1];
      if (pt) labels.push({ x: pt.x, y: pt.y, text: ds.short });
    });
    labels.sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 13);
    ctx.save();
    ctx.font = "600 11px system-ui, sans-serif"; ctx.textBaseline = "middle"; ctx.fillStyle = cssVar("--ink-2");
    for (const l of labels) ctx.fillText(l.text, Math.min(l.x + 6, chart.width - ctx.measureText(l.text).width - 2), l.y);
    ctx.restore();
  },
};

export class LineChart {
  constructor(canvas) {
    this.x = "time"; this.metric = "mean"; this.logX = false;
    const grid = cssVar("--grid"), muted = cssVar("--muted");
    this.chart = new Chart(canvas, {
      type: "line",
      data: { datasets: [] },
      options: {
        animation: false, parsing: false, normalized: true, maintainAspectRatio: false,
        layout: { padding: { right: 70, top: 6 } },
        interaction: { mode: "eachNearestX", intersect: false },
        scales: {
          x: { type: "linear", min: 0, grid: { color: grid }, border: { color: cssVar("--axis") },
               ticks: { color: muted, maxTicksLimit: 6, callback: (v) => X_MODES[this.x].fmt(v) },
               title: { display: true, color: muted, text: X_MODES[this.x].title } },
          y: { beginAtZero: true, grid: { color: grid }, border: { display: false }, ticks: { color: muted, precision: 0 } },
        },
        plugins: {
          legend: { display: false },
          decimation: { enabled: true, algorithm: "lttb", samples: 300 },
          tooltip: {
            backgroundColor: cssVar("--surface"), titleColor: cssVar("--ink-2"), bodyColor: cssVar("--ink"),
            borderColor: cssVar("--border"), borderWidth: 1, padding: 10, boxWidth: 14, boxHeight: 2,
            itemSort: (a, b) => b.parsed.y - a.parsed.y,
            callbacks: {
              title: (items) => items.length ? `${X_MODES[this.x].title}: ${X_MODES[this.x].fmt(items[0].parsed.x)}` : "",
              label: (it) => ` ${fmtNum(it.parsed.y, 2)}   ${it.dataset.label}`,
              labelColor: (it) => ({ borderColor: it.dataset.borderColor, backgroundColor: it.dataset.borderColor }),
            },
          },
        },
      },
      plugins: [crosshair, endLabels],
    });
  }

  // series: [{label, short, color, dash, rows}]; yMax: shared scale for small multiples (optional)
  set(series, { x, metric, logX, yMax } = {}) {
    if (x) this.x = x;
    if (metric) this.metric = metric;
    if (logX !== undefined) this.logX = logX;
    const xc = X_MODES[this.x].col, yc = METRICS[this.metric].col;
    this.chart.data.datasets = series.map((s) => ({
      label: s.label, short: s.short, borderColor: s.color, backgroundColor: s.color,
      borderWidth: 2, borderDash: s.dash ? [6, 4] : [], pointRadius: 0, pointHoverRadius: 4,
      stepped: METRICS[this.metric].stepped ? "before" : false,
      data: s.rows.filter((r) => r[xc] > 0).map((r) => ({ x: r[xc], y: r[yc] })),
    }));
    const xs = this.chart.options.scales.x, ys = this.chart.options.scales.y;
    Object.assign(xs, { type: this.logX ? "logarithmic" : "linear", min: this.logX ? undefined : 0 });
    xs.title.text = X_MODES[this.x].title + (this.logX ? " (log)" : "");
    ys.max = yMax || undefined;
    this.chart.update("none");
  }

  destroy() { this.chart.destroy(); }
}

// The highest value a metric reaches in some rows (for shared y-scales).
export function maxOf(rowsList, metric) {
  const yc = METRICS[metric].col;
  let m = 0;
  for (const rows of rowsList) for (const r of rows) if (r[yc] > m) m = r[yc];
  return m;
}
