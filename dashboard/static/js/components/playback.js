// One animation loop for every board on screen, plus the shared playback controls.
// Controls only change how replays are animated; training is never affected.
import { h } from "../util.js";

const SPEEDS = [["1×", 1], ["2×", 2], ["5×", 5], ["10×", 10], ["50×", 50], ["max", Infinity]];
const MOVES_PER_SEC_AT_1X = 8;

export const playback = { mult: 1, paused: false, boards: new Set() };

export function addBoards(...boards) { for (const b of boards.flat()) playback.boards.add(b); }
export function removeBoards(...boards) { for (const b of boards.flat()) { playback.boards.delete(b); b.dispose?.(); } }

// Boards that can't be seen (closed sections, hidden tabs) don't fetch or animate games.
const visible = (el) => el.isConnected && (el.checkVisibility ? el.checkVisibility() : el.offsetParent !== null);

let last = performance.now();
function frame(now) {
  const dt = Math.min(now - last, 250);
  last = now;
  if (!playback.paused) {
    const mps = playback.mult * MOVES_PER_SEC_AT_1X;
    for (const b of playback.boards) if (visible(b.el)) b.update(now, dt, mps);
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// Speed + pause controls (state is shared, so every view shows the same setting).
export function playbackControls() {
  const seg = h("div", { class: "segmented", role: "group", "aria-label": "Replay speed" });
  const render = () => {
    seg.replaceChildren(...SPEEDS.map(([label, mult]) => h("button", {
      "aria-pressed": String(playback.mult === mult), title: mult === 1 ? "about 8 moves per second" : undefined,
      onclick: () => { playback.mult = mult; render(); },
    }, label)));
  };
  render();
  const pause = h("button", {
    "aria-pressed": String(playback.paused),
    onclick: () => {
      playback.paused = !playback.paused;
      pause.textContent = playback.paused ? "Resume" : "Pause";
      pause.setAttribute("aria-pressed", String(playback.paused));
    },
  }, playback.paused ? "Resume" : "Pause");
  return h("div", { class: "controls" }, h("div", { class: "ctl", role: "group", "aria-label": "Replay speed" }, "Replay speed", seg), pause);
}
