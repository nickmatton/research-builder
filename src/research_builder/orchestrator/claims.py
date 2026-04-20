"""Claims verification: compare sub-agent results against the claims ledger.

The acceptance review calls ``verify_phase_claims`` after a phase reports
success. The result is a ``ClaimsReport`` that feeds into the semantic
acceptance review and the final reproduction report.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ..models.claims import (
    Claim,
    ClaimsLedger,
    ClaimsReport,
    ClaimVerification,
    VerificationStatus,
)
from ..models.results import SubAgentResult

logger = logging.getLogger(__name__)


def verify_phase_claims(
    phase_id: str,
    result: SubAgentResult,
    ledger: ClaimsLedger,
    work_dir: Path | None = None,
) -> ClaimsReport:
    """Check a phase's results against its claims from the ledger.

    Strategy:
    1. Get all claims assigned to this phase.
    2. For each claim, try to find the actual value in:
       a. The sub-agent's test report (test names/descriptions/messages that
          mention the metric).
       b. The sub-agent's diagnostics dict.
       c. Output artifact JSON files in work_dir/outputs/.
    3. Compare actual vs. expected with tolerance.
    """
    phase_claims = ledger.for_phase(phase_id)
    if not phase_claims:
        return ClaimsReport()

    verifications: list[ClaimVerification] = []
    for claim in phase_claims:
        actual = _find_actual_value(claim, result, work_dir)
        verification = _compare(claim, actual)
        verifications.append(verification)

    report = ClaimsReport(verifications=verifications)
    logger.info(
        "Claims verification for phase=%s: %d verified, %d close, %d missed, "
        "%d exceeded, %d unchecked",
        phase_id,
        report.verified_count,
        report.close_count,
        report.missed_count,
        report.exceeded_count,
        report.not_checked_count,
    )
    return report


def _find_actual_value(
    claim: Claim,
    result: SubAgentResult,
    work_dir: Path | None,
) -> float | None:
    """Best-effort search for the actual value of a claim in phase outputs."""
    metric_lower = claim.metric.lower()
    claim_id_lower = claim.claim_id.lower()

    # 1. Check sub-agent diagnostics
    if result.diagnostics:
        val = _search_dict_for_metric(result.diagnostics, metric_lower, claim_id_lower)
        if val is not None:
            return val

    # 2. Check test report messages — tests often print metric values
    for t in result.test_report.test_details:
        val = _extract_number_near_keyword(
            f"{t.test_name} {t.description or ''} {t.message or ''}",
            metric_lower,
        )
        if val is not None:
            return val

    # 3. Check sub-agent summary
    val = _extract_number_near_keyword(result.summary, metric_lower)
    if val is not None:
        return val

    # 4. Scan output JSON files in work_dir/outputs/
    if work_dir is not None:
        outputs_dir = work_dir / "outputs"
        if outputs_dir.exists():
            for f in sorted(outputs_dir.iterdir()):
                if f.suffix == ".json" and f.name != "_result.json":
                    val = _search_json_file(f, metric_lower, claim_id_lower)
                    if val is not None:
                        return val

    return None


def _search_dict_for_metric(
    d: dict,
    metric_lower: str,
    claim_id_lower: str,
) -> float | None:
    """Recursively search a dict for keys matching the metric."""
    for key, val in d.items():
        key_lower = key.lower()
        if metric_lower in key_lower or claim_id_lower in key_lower:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
        if isinstance(val, dict):
            result = _search_dict_for_metric(val, metric_lower, claim_id_lower)
            if result is not None:
                return result
    return None


def _search_json_file(
    path: Path,
    metric_lower: str,
    claim_id_lower: str,
) -> float | None:
    """Try to find a metric value in a JSON file."""
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return _search_dict_for_metric(data, metric_lower, claim_id_lower)
    except Exception:
        pass
    return None


def _extract_number_near_keyword(text: str, keyword: str) -> float | None:
    """Find a number in text that appears near a keyword.

    Looks for patterns like "accuracy: 95.2", "accuracy = 0.952",
    "accuracy 95.2%", etc.
    """
    if not text or keyword not in text.lower():
        return None

    # Find keyword positions and look for nearby numbers
    text_lower = text.lower()
    idx = text_lower.find(keyword)
    if idx < 0:
        return None

    # Search in a window around the keyword
    window_start = max(0, idx - 20)
    window_end = min(len(text), idx + len(keyword) + 50)
    window = text[window_start:window_end]

    # Look for number patterns: 95.2, 0.952, 1e-3, etc.
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", window)
    if not numbers:
        return None

    # Pick the number closest to (but after) the keyword
    keyword_pos_in_window = idx - window_start
    best = None
    best_dist = float("inf")
    for m in re.finditer(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", window):
        dist = abs(m.start() - keyword_pos_in_window - len(keyword))
        if dist < best_dist:
            best_dist = dist
            try:
                best = float(m.group())
            except ValueError:
                pass

    return best


def _compare(claim: Claim, actual: float | None) -> ClaimVerification:
    """Compare actual value against claim, respecting tolerance."""
    if actual is None:
        return ClaimVerification(
            claim_id=claim.claim_id,
            status=VerificationStatus.not_checked,
            expected=claim.value,
            note="No matching result found in phase outputs",
        )

    delta = actual - claim.value
    delta_pct = (delta / claim.value * 100) if claim.value != 0 else None

    # Determine tolerance band
    if claim.tolerance > 0:
        tol = claim.tolerance
    else:
        # Default: 5% relative tolerance or 0.5 absolute, whichever is larger
        tol = max(abs(claim.value * 0.05), 0.5)

    abs_delta = abs(delta)

    if abs_delta <= tol:
        status = VerificationStatus.verified
    elif abs_delta <= tol * 2:
        status = VerificationStatus.close
    elif delta > 0 and abs_delta > tol * 2:
        # Suspiciously better than paper claims
        status = VerificationStatus.exceeded
    else:
        status = VerificationStatus.missed

    note = ""
    if status == VerificationStatus.exceeded:
        note = (
            f"Result ({actual:.4g}) exceeds paper claim ({claim.value:.4g}) by "
            f"more than 2x tolerance — possible data leak or evaluation mismatch"
        )

    return ClaimVerification(
        claim_id=claim.claim_id,
        status=status,
        expected=claim.value,
        actual=actual,
        delta=delta,
        delta_pct=delta_pct,
        note=note,
    )
