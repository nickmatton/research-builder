"""Tests for the results generation phase."""

import json
import os
import re
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_results import main, OUTPUT_DIR


def run_generation():
    """Run the generation and return outputs."""
    return main()


def test_all():
    results = []

    # Run generation first
    try:
        summary, ablation, comparison_md, report = run_generation()
    except Exception as e:
        results.append(("generation_runs", False, str(e)))
        return results
    results.append(("generation_runs", True, ""))

    # ── Output files exist ──
    expected_files = [
        "results_summary.json",
        "ablation_results.json",
        "comparison_table.md",
        "reproduction_report.md",
    ]
    for fname in expected_files:
        path = OUTPUT_DIR / fname
        exists = path.exists() and path.stat().st_size > 0
        results.append((f"file_exists_{fname}", exists, "" if exists else f"{fname} missing or empty"))

    # ── results_summary.json checks ──
    with open(OUTPUT_DIR / "results_summary.json") as f:
        s = json.load(f)

    has_reproduced = "reproduced" in s and "en_de_bleu" in s["reproduced"]
    results.append(("summary_has_reproduced_bleu", has_reproduced, ""))

    has_paper = "paper_reported" in s and "en_de_base_bleu" in s["paper_reported"]
    results.append(("summary_has_paper_values", has_paper, ""))

    has_disc = "discrepancy" in s and "en_de_bleu_diff" in s["discrepancy"]
    results.append(("summary_has_discrepancy", has_disc, ""))

    # Discrepancy computed correctly
    if has_reproduced and has_paper and has_disc:
        expected_diff = round(s["reproduced"]["en_de_bleu"] - s["paper_reported"]["en_de_base_bleu"], 2)
        diff_correct = s["discrepancy"]["en_de_bleu_diff"] == expected_diff
        results.append(("summary_discrepancy_correct", diff_correct,
                        f"expected {expected_diff}, got {s['discrepancy']['en_de_bleu_diff']}"))

    # ── ablation_results.json checks ──
    with open(OUTPUT_DIR / "ablation_results.json") as f:
        a = json.load(f)

    has_findings = "findings" in a and len(a["findings"]) >= 5
    results.append(("ablation_has_5_findings", has_findings, f"found {len(a.get('findings', []))} findings"))

    has_base = "base_model" in a and "bleu" in a.get("base_model", {})
    results.append(("ablation_has_base_model", has_base, ""))

    # Check all findings are directionally consistent
    if has_findings:
        all_consistent = all(f["directionally_consistent"] for f in a["findings"])
        results.append(("ablation_directionally_consistent", all_consistent, ""))

    # ── comparison_table.md checks ──
    with open(OUTPUT_DIR / "comparison_table.md") as f:
        table = f.read()

    has_header = "EN-DE BLEU" in table and "EN-FR BLEU" in table
    results.append(("table_has_headers", has_header, ""))

    has_transformer = "Transformer (base)" in table and "Transformer (big)" in table
    results.append(("table_has_transformer_rows", has_transformer, ""))

    has_baselines = "ConvS2S" in table and "GNMT" in table and "MoE" in table
    results.append(("table_has_baseline_models", has_baselines, ""))

    # Check key BLEU values present
    has_key_values = "27.3" in table and "28.4" in table and "41.8" in table
    results.append(("table_has_key_bleu_values", has_key_values, ""))

    # ── reproduction_report.md checks ──
    with open(OUTPUT_DIR / "reproduction_report.md") as f:
        rpt = f.read()

    # Valid markdown (has headers)
    has_headers = rpt.count("## ") >= 5
    results.append(("report_has_section_headers", has_headers, f"found {rpt.count('## ')} headers"))

    # Contains all required sections
    required_sections = [
        "Machine Translation Results",
        "Training Summary",
        "Discrepancy Analysis",
        "Ablation Study",
        "Comparison Table",
    ]
    for section in required_sections:
        found = section in rpt
        results.append((f"report_has_section_{section.replace(' ', '_').lower()}", found,
                        f"'{section}' not found" if not found else ""))

    # Training curve figure has data
    has_training_curve = "Step  |  Loss" in rpt and "█" in rpt
    results.append(("report_training_curve_has_data", has_training_curve, ""))

    # LR schedule present
    has_lr = "Learning Rate" in rpt
    results.append(("report_has_lr_schedule", has_lr, ""))

    # Report is non-trivial length
    long_enough = len(rpt) > 2000
    results.append(("report_is_substantive", long_enough, f"only {len(rpt)} chars"))

    # Acceptance criteria table present
    has_acceptance = "Acceptance Criteria" in rpt
    results.append(("report_has_acceptance_criteria", has_acceptance, ""))

    # Comparison values populated (not all dashes)
    comparison_section = rpt.split("Comparison Table")[1] if "Comparison Table" in rpt else ""
    has_numbers = bool(re.search(r"\d+\.\d+", comparison_section))
    results.append(("report_comparison_values_populated", has_numbers, ""))

    return results


if __name__ == "__main__":
    results = test_all()
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)

    print(f"\n{'='*60}")
    print(f"Tests: {passed} passed, {failed} failed, {len(results)} total")
    print(f"{'='*60}\n")

    for name, ok, msg in results:
        status = "PASS" if ok else "FAIL"
        suffix = f" — {msg}" if msg and not ok else ""
        print(f"  [{status}] {name}{suffix}")

    if failed:
        print(f"\n{failed} test(s) FAILED")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)
