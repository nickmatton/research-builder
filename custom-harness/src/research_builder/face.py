"""ASCII face animation — the 'face' of the research builder.

A small stateful Rich renderable that draws an ASCII face which can:
  - talk (mouth cycles through shapes while ``_talking`` is True)
  - blink (eyes close briefly at random intervals while idle/talking)
  - wink (one eye closed + smile, held for ``WINK_DURATION`` seconds)

Drop it inside a ``rich.live.Live`` so frames refresh in place:

    from rich.live import Live
    from research_builder.face import Face

    face = Face()
    with Live(face, refresh_per_second=12):
        face.start_talking()
        ...  # stream tokens from the model
        face.stop_talking()
        face.wink()
        time.sleep(Face.WINK_DURATION)  # let the wink play out

Run a standalone demo to see it:

    python -m research_builder.face
"""

from __future__ import annotations

import asyncio
import random
import time

from rich.console import Console
from rich.live import Live
from rich.text import Text


class Face:
    """Stateful, time-aware ASCII face.

    Rich calls ``__rich__`` on every refresh; that method consults
    ``time.time()`` to pick the right frame, so the only thing callers
    need to drive is the high-level state (talking / wink).
    """

    BLINK_DURATION = 0.13
    BLINK_MIN_INTERVAL = 2.5
    BLINK_MAX_INTERVAL = 5.5
    WINK_DURATION = 1.0
    TALK_FRAME_HZ = 7

    # Mouth glyph cycle used while ``_talking`` is True.
    TALK_MOUTHS = ("o", "O", "o", "─")
    IDLE_MOUTH = "─"
    WINK_MOUTH = "‿"
    EYE_OPEN = "o"
    EYE_CLOSED = "-"

    def __init__(self) -> None:
        now = time.time()
        self._talking = False
        self._blink_until: float = 0.0
        self._next_blink_at: float = now + self._next_blink_interval()
        self._wink_until: float = 0.0

    # ---- state controls ---------------------------------------------------

    def start_talking(self) -> None:
        self._talking = True

    def stop_talking(self) -> None:
        self._talking = False

    def wink(self) -> None:
        """Trigger a wink that holds for ``WINK_DURATION`` seconds."""
        self._wink_until = time.time() + self.WINK_DURATION

    @property
    def is_winking(self) -> bool:
        return time.time() < self._wink_until

    # ---- rendering --------------------------------------------------------

    def __rich__(self) -> Text:
        now = time.time()
        winking = now < self._wink_until

        # Schedule a new blink if it's time.
        if not winking and now >= self._next_blink_at and now >= self._blink_until:
            self._blink_until = now + self.BLINK_DURATION
            self._next_blink_at = now + self._next_blink_interval()
        blinking = (not winking) and (now < self._blink_until)

        if winking:
            le, re_ = self.EYE_OPEN, self.EYE_CLOSED
            mouth = self.WINK_MOUTH
            mouth_style = "magenta"
        elif blinking:
            le = re_ = self.EYE_CLOSED
            mouth = self._mouth_glyph(now)
            mouth_style = "yellow" if self._talking else "white"
        else:
            le = re_ = self.EYE_OPEN
            mouth = self._mouth_glyph(now)
            mouth_style = "yellow" if self._talking else "white"

        return self._compose(le, re_, mouth, mouth_style)

    def _mouth_glyph(self, now: float) -> str:
        if not self._talking:
            return self.IDLE_MOUTH
        idx = int(now * self.TALK_FRAME_HZ) % len(self.TALK_MOUTHS)
        return self.TALK_MOUTHS[idx]

    def _compose(self, le: str, re_: str, mouth: str, mouth_style: str) -> Text:
        # 9-cell-wide, 4-row face. Layout is hand-counted; keep widths in sync.
        border = "cyan"
        eye_style = "bright_white"

        t = Text()
        t.append("╭───────╮\n", style=border)
        t.append("│", style=border)
        t.append("  ")
        t.append(le, style=eye_style)
        t.append(" ")
        t.append(re_, style=eye_style)
        t.append("  ")
        t.append("│\n", style=border)
        t.append("│", style=border)
        t.append("   ")
        t.append(mouth, style=mouth_style)
        t.append("   ")
        t.append("│\n", style=border)
        t.append("╰───────╯", style=border)
        return t

    def _next_blink_interval(self) -> float:
        return random.uniform(self.BLINK_MIN_INTERVAL, self.BLINK_MAX_INTERVAL)


# ----------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------

async def _demo() -> None:
    """Sit idle, then talk for a few seconds, then wink off."""
    console = Console()
    face = Face()
    console.print("[dim]demo: idle (watch for blinks) →[/dim]")
    with Live(face, console=console, refresh_per_second=12, transient=False):
        await asyncio.sleep(2.5)
        console.print("[dim]demo: talking →[/dim]")
        face.start_talking()
        await asyncio.sleep(4.0)
        face.stop_talking()
        await asyncio.sleep(0.4)
        console.print("[dim]demo: wink →[/dim]")
        face.wink()
        await asyncio.sleep(Face.WINK_DURATION + 0.3)


if __name__ == "__main__":
    asyncio.run(_demo())
