// ASCII face that talks while any chat stream is open, blinks at random
// idle intervals, and winks each time a stream finishes. State machine
// mirrors the Python `research_builder.face.Face` for parity.
//
// Drives re-renders at ~12fps via setInterval so the mouth animation
// and blink schedule stay smooth without needing each tick to be a
// controller event.

import { useEffect, useRef, useState } from "react";
import { FaceController } from "../lib/face";

const BLINK_DURATION_MS = 130;
const BLINK_MIN_MS = 2500;
const BLINK_MAX_MS = 5500;
const TALK_FRAME_HZ = 7;
const TALK_MOUTHS = ["o", "O", "o", "─"] as const;

function randBlinkInterval() {
  return BLINK_MIN_MS + Math.random() * (BLINK_MAX_MS - BLINK_MIN_MS);
}

export function Face() {
  const blinkUntilRef = useRef(0);
  const nextBlinkAtRef = useRef(Date.now() + randBlinkInterval());
  const [, setTick] = useState(0);

  useEffect(() => {
    const unsub = FaceController.subscribe(() => setTick((t) => t + 1));
    const id = window.setInterval(() => setTick((t) => t + 1), 80);
    return () => {
      unsub();
      window.clearInterval(id);
    };
  }, []);

  const now = Date.now();
  const winking = FaceController.isWinking();

  // Schedule a blink if it's time (skipped while winking — never overlap).
  if (
    !winking &&
    now >= nextBlinkAtRef.current &&
    now >= blinkUntilRef.current
  ) {
    blinkUntilRef.current = now + BLINK_DURATION_MS;
    nextBlinkAtRef.current = now + randBlinkInterval();
  }
  const blinking = !winking && now < blinkUntilRef.current;
  const talking = FaceController.isTalking() && !winking;

  let le = "o";
  let re = "o";
  let mouth: string = "─";
  let mouthColor = "var(--color-fg)";

  if (winking) {
    le = "o";
    re = "-";
    mouth = "‿";
    mouthColor = "var(--color-accent)";
  } else if (blinking) {
    le = "-";
    re = "-";
    if (talking) {
      mouth = TALK_MOUTHS[Math.floor((now / 1000) * TALK_FRAME_HZ) % TALK_MOUTHS.length];
      mouthColor = "var(--color-warn)";
    }
  } else if (talking) {
    mouth = TALK_MOUTHS[Math.floor((now / 1000) * TALK_FRAME_HZ) % TALK_MOUTHS.length];
    mouthColor = "var(--color-warn)";
  }

  const border = { color: "var(--color-accent)" };
  const eye = { color: "var(--color-fg)" };

  // 5-wide × 4-row face. Tight enough to live inside the TopBar.
  return (
    <div
      aria-hidden
      className="select-none font-mono text-[8px] leading-[9px]"
      title="research-builder"
    >
      <div style={border}>╭───╮</div>
      <div>
        <span style={border}>│</span>
        <span style={eye}>{le}</span>
        <span> </span>
        <span style={eye}>{re}</span>
        <span style={border}>│</span>
      </div>
      <div>
        <span style={border}>│ </span>
        <span style={{ color: mouthColor }}>{mouth}</span>
        <span style={border}> │</span>
      </div>
      <div style={border}>╰───╯</div>
    </div>
  );
}
