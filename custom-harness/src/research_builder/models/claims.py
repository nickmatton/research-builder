"""Data models for the claims ledger — structured numerical claims from the paper.

The claims ledger is extracted during spec creation and stored in
``canonical_spec/claims.json``. Each phase's results are verified against
the relevant claims during acceptance review.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ClaimSource(BaseModel):
    """Where in the paper a claim originates."""
    table: str | None = None       # e.g. "Table 2"
    figure: str | None = None      # e.g. "Figure 3"
    section: str | None = None     # e.g. "Section 4.2"
    page: int | None = None
    verbatim: str = ""             # exact quote from paper


class Claim(BaseModel):
    """A single numerical claim extracted from the paper.

    Example: "CIFAR-10 test accuracy = 95.2 ± 0.3 (Table 2, row 3)"
    """
    claim_id: str                    # stable ID, e.g. "table2_cifar10_accuracy"
    metric: str                      # e.g. "accuracy", "F1", "BLEU", "loss"
    value: float                     # the reported number
    tolerance: float = 0.0           # ± range (0 if not stated)
    unit: str = ""                   # e.g. "%", "ms", "perplexity"
    dataset: str = ""                # e.g. "CIFAR-10 test set"
    condition: str = ""              # e.g. "ResNet-50, batch size 256"
    source: ClaimSource = Field(default_factory=ClaimSource)
    phase_id: str = ""               # which phase should produce this result
    notes: str = ""                  # any caveats flagged during extraction


class VerificationStatus(str, Enum):
    verified = "verified"            # within tolerance
    close = "close"                  # within 2x tolerance or 5% relative
    missed = "missed"                # outside tolerance
    exceeded = "exceeded"            # suspiciously better than paper
    not_checked = "not_checked"      # no matching result found


class ClaimVerification(BaseModel):
    """Result of checking one claim against actual outputs."""
    claim_id: str
    status: VerificationStatus
    expected: float
    actual: float | None = None
    delta: float | None = None       # actual - expected
    delta_pct: float | None = None   # relative delta as percentage
    note: str = ""


class ClaimsLedger(BaseModel):
    """The full ledger of numerical claims from a paper."""
    claims: list[Claim] = Field(default_factory=list)

    def for_phase(self, phase_id: str) -> list[Claim]:
        return [c for c in self.claims if c.phase_id == phase_id]

    def get_claim(self, claim_id: str) -> Claim | None:
        for c in self.claims:
            if c.claim_id == claim_id:
                return c
        return None


class ClaimsReport(BaseModel):
    """Aggregated verification results for a set of claims."""
    verifications: list[ClaimVerification] = Field(default_factory=list)

    @property
    def verified_count(self) -> int:
        return sum(1 for v in self.verifications if v.status == VerificationStatus.verified)

    @property
    def close_count(self) -> int:
        return sum(1 for v in self.verifications if v.status == VerificationStatus.close)

    @property
    def missed_count(self) -> int:
        return sum(1 for v in self.verifications if v.status == VerificationStatus.missed)

    @property
    def exceeded_count(self) -> int:
        return sum(1 for v in self.verifications if v.status == VerificationStatus.exceeded)

    @property
    def not_checked_count(self) -> int:
        return sum(1 for v in self.verifications if v.status == VerificationStatus.not_checked)

    def to_markdown(self) -> str:
        if not self.verifications:
            return "_No claims to verify._"
        lines = [
            "| Status | Claim | Expected | Actual | Delta |",
            "|--------|-------|----------|--------|-------|",
        ]
        status_icon = {
            VerificationStatus.verified: "✓",
            VerificationStatus.close: "~",
            VerificationStatus.missed: "✗",
            VerificationStatus.exceeded: "⚠",
            VerificationStatus.not_checked: "—",
        }
        for v in self.verifications:
            icon = status_icon.get(v.status, "?")
            actual = f"{v.actual:.4g}" if v.actual is not None else "—"
            delta = ""
            if v.delta is not None:
                sign = "+" if v.delta >= 0 else ""
                delta = f"{sign}{v.delta:.4g}"
                if v.delta_pct is not None:
                    delta += f" ({sign}{v.delta_pct:.1f}%)"
            lines.append(f"| {icon} | {v.claim_id} | {v.expected:.4g} | {actual} | {delta} |")

        summary = (
            f"\n**Summary:** {self.verified_count} verified, "
            f"{self.close_count} close, {self.missed_count} missed, "
            f"{self.exceeded_count} suspicious, {self.not_checked_count} unchecked"
        )
        lines.append(summary)
        return "\n".join(lines)
