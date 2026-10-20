// Small shared helpers: DOM building (text is always set with textContent), formatting, API calls.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("--")) el.style.setProperty(k, v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k in el && typeof v !== "string") el[k] = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c === undefined || c === null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export const cssVar = (k) => getComputedStyle(document.documentElement).getPropertyValue(k).trim();
export const seriesVar = (i) => `var(--c${(i % 8) + 1})`;
export const seriesColor = (i) => cssVar(`--c${(i % 8) + 1}`);

export function fmtDuration(s) {
  if (s === null || s === undefined || !isFinite(s)) return "–";
  s = Math.max(0, Math.round(s));
  const hr = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = String(s % 60).padStart(2, "0");
  return hr ? `${hr}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
}
export function fmtCompact(v) {
  if (v === null || v === undefined || !isFinite(v)) return "–";
  return Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(v);
}
export function fmtNum(v, digits = 1) {
  if (v === null || v === undefined || !isFinite(v)) return "–";
  return Number(v).toLocaleString("en", { maximumFractionDigits: digits, minimumFractionDigits: 0 });
}
export function fmtDate(s) {
  if (!s) return "–";
  const d = new Date(s.replace(" ", "T"));
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}
export function fmtMB(mb) {
  if (mb === null || mb === undefined) return "?";
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`;
}

export function statusPill(status) {
  const label = { running: "training", queued: "queued", finished: "finished", stopped: "stopped early",
                  interrupted: "interrupted", failed: "failed" }[status] || status;
  return h("span", { class: `pill ${status}` }, label);
}

export function runTitle(run) {
  if (run.name) return run.name;
  const d = new Date((run.created || run.started || "").replace(" ", "T"));
  const when = isNaN(d) ? run.run_id : d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  return `${run.lesson} · ${when}`;
}

export function agentLabel(agent, perAgentReward) {
  const alg = agent.algorithm, short = { ga: "GA", dqn: "DQN" }[alg] || alg;
  let name = agent.id === alg ? short
    : agent.id.startsWith(alg + "_") || agent.id.startsWith(alg + "-") ? `${short} ${agent.id.slice(alg.length + 1)}` : `${short} · ${agent.id}`;
  if (perAgentReward && !name.endsWith(agent.reward)) name += ` (${agent.reward})`;
  return name;
}

// Markdown written by the LLM: raw HTML is escaped and only http(s) links are kept.
export function markdown(text) {
  const el = h("div", { class: "markdown" });
  const safe = String(text || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  el.innerHTML = window.marked ? window.marked.parse(safe) : safe;
  for (const a of el.querySelectorAll("a")) {
    if (!/^https?:\/\//.test(a.getAttribute("href") || "")) a.removeAttribute("href");
    else { a.target = "_blank"; a.rel = "noopener"; }
  }
  return el;
}

let toastTimer;
export function toast(message) {
  document.querySelector(".toast")?.remove();
  const el = h("div", { class: "toast", role: "status" }, message);
  document.body.append(el);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.remove(), 3500);
}

export function openDialog(...content) {
  const dlg = document.getElementById("dialog");
  const body = document.getElementById("dialog-body");
  body.replaceChildren(...content);
  if (!dlg.open) dlg.showModal();
  return dlg;
}
export function closeDialog() { document.getElementById("dialog").close(); }

export function confirmDialog(title, text, okLabel = "Delete") {
  return new Promise((resolve) => {
    const done = (v) => { closeDialog(); resolve(v); };
    openDialog(h("h2", {}, title), h("p", { class: "ink2" }, text),
      h("div", { class: "btn-row" }, h("button", { class: "primary danger", onclick: () => done(true) }, okLabel),
        h("button", { onclick: () => done(false) }, "Cancel")));
    document.getElementById("dialog").addEventListener("close", () => resolve(false), { once: true });
  });
}

// ---- API ----
async function call(method, path, body) {
  const res = await fetch(path, {
    method, headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `${res.status} ${res.statusText}`);
  return data;
}
export const api = {
  get: (p) => call("GET", p), post: (p, b) => call("POST", p, b || {}),
  patch: (p, b) => call("PATCH", p, b), del: (p) => call("DELETE", p),
};
export const q = (id) => encodeURIComponent(id);
