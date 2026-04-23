"""Per-run GPU spend budget ledger.

Tracks projected and actual spend across all phases in a single research-builder
run. Hard-caps spend at a configurable ceiling; when a provision request would
push projected spend over the cap, the provisioner asks an injected callback
(usually a click prompt that bubbles up from a sub-agent's request_compute tool)
whether to raise the cap.

This is deliberately deterministic — the cap check is plain arithmetic, never
an LLM call. The only LLM-touched part of GPU sizing is the per-phase
classifier in CloudProvisioner.needs_gpu; the ledger itself only ever does math.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_CAP_USD = 30.0
# Used when the LLM classifier doesn't (yet) pin an instance_type to a known
# hourly rate. Conservative — better to over-debit and refund on release.
FALLBACK_HOURLY_RATE_USD = 1.50


@dataclass
class BudgetEntry:
    entry_id: str
    phase_id: str
    instance_type: str
    hourly_rate_usd: float
    estimated_hours: float
    started_at: float  # monotonic seconds when provisioned
    ended_at: float | None = None  # monotonic seconds when released
    actual_hours: float | None = None  # filled in on release

    def estimated_cost(self) -> float:
        return self.hourly_rate_usd * self.estimated_hours

    def actual_cost(self) -> float | None:
        if self.actual_hours is None:
            return None
        return self.hourly_rate_usd * self.actual_hours


class BudgetLedger:
    """Tracks projected GPU spend against a hard cap.

    Use ``would_exceed`` before calling ``commit``. Once an entry is committed
    its estimated cost counts toward the projected total until ``release`` is
    called, at which point we substitute the actual elapsed-time cost.
    """

    def __init__(self, cap_usd: float = DEFAULT_CAP_USD) -> None:
        self.cap_usd: float = float(cap_usd)
        self._entries: dict[str, BudgetEntry] = {}
        self._next_id = 0

    # ---- query --------------------------------------------------------------

    def projected_total(self) -> float:
        """Closed entries contribute their actual cost; open entries contribute their estimate."""
        total = 0.0
        for e in self._entries.values():
            actual = e.actual_cost()
            total += actual if actual is not None else e.estimated_cost()
        return total

    def remaining(self) -> float:
        return max(0.0, self.cap_usd - self.projected_total())

    def would_exceed(self, additional_cost_usd: float) -> bool:
        return self.projected_total() + additional_cost_usd > self.cap_usd

    def active_entries(self) -> list[BudgetEntry]:
        return [e for e in self._entries.values() if e.ended_at is None]

    # ---- mutate -------------------------------------------------------------

    def commit(
        self,
        *,
        phase_id: str,
        instance_type: str,
        hourly_rate_usd: float,
        estimated_hours: float,
    ) -> str:
        """Reserve budget for a newly provisioned machine. Returns entry_id."""
        self._next_id += 1
        entry_id = f"e{self._next_id}"
        entry = BudgetEntry(
            entry_id=entry_id,
            phase_id=phase_id,
            instance_type=instance_type,
            hourly_rate_usd=hourly_rate_usd,
            estimated_hours=estimated_hours,
            started_at=time.monotonic(),
        )
        self._entries[entry_id] = entry
        logger.info(
            "BudgetLedger.commit: id=%s phase=%s type=%s rate=$%.2f/hr est_hrs=%.2f "
            "est_cost=$%.2f new_total=$%.2f cap=$%.2f",
            entry_id, phase_id, instance_type, hourly_rate_usd, estimated_hours,
            entry.estimated_cost(), self.projected_total(), self.cap_usd,
        )
        return entry_id

    def release(self, entry_id: str) -> None:
        """Mark a machine as torn down; substitute actual elapsed cost for estimate."""
        entry = self._entries.get(entry_id)
        if entry is None or entry.ended_at is not None:
            return
        entry.ended_at = time.monotonic()
        entry.actual_hours = (entry.ended_at - entry.started_at) / 3600.0
        logger.info(
            "BudgetLedger.release: id=%s phase=%s actual_hrs=%.3f actual_cost=$%.2f new_total=$%.2f",
            entry_id, entry.phase_id, entry.actual_hours,
            entry.actual_cost() or 0.0, self.projected_total(),
        )

    def raise_cap(self, new_cap_usd: float) -> None:
        """Raise the cap. No-op if new_cap_usd <= current cap (caps never lower mid-run)."""
        if new_cap_usd <= self.cap_usd:
            return
        logger.info("BudgetLedger.raise_cap: $%.2f -> $%.2f", self.cap_usd, new_cap_usd)
        self.cap_usd = float(new_cap_usd)

    # ---- display ------------------------------------------------------------

    def summary(self) -> str:
        return (
            f"GPU spend: ${self.projected_total():.2f} / ${self.cap_usd:.2f} cap "
            f"({len(self.active_entries())} active machine(s))"
        )
