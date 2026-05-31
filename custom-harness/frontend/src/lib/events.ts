// WebSocket subscriber for /ws/events. Auto-reconnects with backoff.
// Exposes a tiny pub-sub so multiple components can listen without
// each opening its own socket.
//
// Maintains a ring buffer of recent events so components that mount
// after the stream connected (or remount on tab toggle) still see the
// transcript. Capped at MAX_BUFFER to bound memory on long runs.

import type { HarnessEvent } from "./types";

type Listener = (e: HarnessEvent) => void;

const MAX_BUFFER = 5000;

export class EventStream {
  private ws: WebSocket | null = null;
  private listeners = new Set<Listener>();
  private attempt = 0;
  private closed = false;
  private url: string;
  private buffer: HarnessEvent[] = [];

  constructor(url: string) {
    this.url = url;
  }

  connect(): void {
    if (this.ws) return;
    const ws = new WebSocket(this.url);
    this.ws = ws;

    ws.addEventListener("open", () => {
      this.attempt = 0;
    });
    ws.addEventListener("message", (msg) => {
      try {
        const data = JSON.parse(msg.data) as HarnessEvent;
        // Buffer first, then fan out — so a late subscriber's snapshot()
        // already includes the event that just arrived.
        this.buffer.push(data);
        if (this.buffer.length > MAX_BUFFER) {
          this.buffer.splice(0, this.buffer.length - MAX_BUFFER);
        }
        for (const fn of this.listeners) fn(data);
      } catch {
        // Ignore non-JSON frames — backend only ever sends JSON, but
        // hot-reloads / proxies can inject junk during dev.
      }
    });
    ws.addEventListener("close", () => {
      this.ws = null;
      if (this.closed) return;
      // On reconnect the backend replays the full events.jsonl, so
      // wipe the buffer to avoid double-counting historical events
      // when the WS comes back up.
      this.buffer = [];
      const delay = Math.min(10_000, 250 * 2 ** this.attempt);
      this.attempt += 1;
      setTimeout(() => this.connect(), delay);
    });
    ws.addEventListener("error", () => {
      // close handler runs after error — let it do the reconnect.
    });
  }

  /** Snapshot of all events seen so far. Cheap copy; caller owns it. */
  snapshot(): HarnessEvent[] {
    return this.buffer.slice();
  }

  /** Drop the in-memory ring (used by the Activity "clear" button). */
  clearBuffer(): void {
    this.buffer = [];
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  close(): void {
    this.closed = true;
    this.ws?.close();
    this.ws = null;
  }
}

let _stream: EventStream | null = null;

export function getEventStream(): EventStream {
  if (_stream) return _stream;
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  // Same host as the page — dev: 5173 via Vite proxy → :7777; prod: :7777.
  _stream = new EventStream(`${proto}://${window.location.host}/ws/events`);
  _stream.connect();
  return _stream;
}
