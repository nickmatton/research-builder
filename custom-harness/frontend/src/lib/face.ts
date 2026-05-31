// Global face controller. Imperative API for chat hooks to call as
// streams start / end, plus a subscribe() so the <Face/> component
// re-renders when state flips.
//
// Multiple concurrent streams are supported via a counter: each
// assistant_start increments, each assistant_end decrements. The face
// "talks" whenever the counter is > 0.

type Listener = () => void;

let talkingCount = 0;
let winkUntil = 0;
const listeners = new Set<Listener>();

function notify() {
  listeners.forEach((fn) => fn());
}

export const FaceController = {
  startTalking() {
    talkingCount += 1;
    notify();
  },
  stopTalking() {
    talkingCount = Math.max(0, talkingCount - 1);
    notify();
  },
  wink(durationMs = 1000) {
    winkUntil = Date.now() + durationMs;
    notify();
  },
  isTalking() {
    return talkingCount > 0;
  },
  isWinking() {
    return Date.now() < winkUntil;
  },
  subscribe(fn: Listener) {
    listeners.add(fn);
    return () => {
      listeners.delete(fn);
    };
  },
};
