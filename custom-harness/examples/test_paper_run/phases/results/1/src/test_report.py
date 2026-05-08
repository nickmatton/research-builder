#!/usr/bin/env python3
"""Tests for the reproduction report."""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from generate_report import generate_report, _ascii_chart

BASE = "/private/tmp/research-builder-test"
METRICS = f"{BASE}/phases/eval/1/outputs/metrics.json"
TRAINING_LOG = f"{BASE}/phases/training/1/outputs/training_log.json"
OUTPUT = f"{BASE}/phases/results/1/outputs/reproduction_report.md"

results = []

def run_test(name, description):
    def decorator(fn):
        try:
            fn()
            results.append({"name": name, "desc": description, "status": "passed", "msg": ""})
        except Exception as e:
            results.append({"name": name, "desc": description, "status": "failed", "msg": str(e)})
    return decorator

# Generate the report first
generate_report(METRICS, TRAINING_LOG, OUTPUT)

with open(OUTPUT) as f:
    report = f.read()

@run_test("test_report_exists", "Output file exists")
def _():
    assert os.path.exists(OUTPUT), "reproduction_report.md not found"

@run_test("test_valid_markdown_headers", "Report has valid markdown headers")
def _():
    headers = re.findall(r'^#{1,3} .+', report, re.MULTILINE)
    assert len(headers) >= 5, f"Expected >=5 headers, got {len(headers)}: {headers}"

@run_test("test_contains_table1", "Contains Table 1: Test Set Performance")
def _():
    assert "Table 1" in report, "Table 1 not found"
    assert "Test Accuracy" in report
    assert "95.5%" in report
    assert "95.2%" in report

@run_test("test_contains_table2", "Contains Table 2: Per-Class Accuracy")
def _():
    assert "Table 2" in report, "Table 2 not found"
    assert "Class 0" in report
    assert "Class 1" in report

@run_test("test_contains_figure1", "Contains Figure 1: Training Loss")
def _():
    assert "Figure 1" in report, "Figure 1 not found"
    assert "Training Loss" in report

@run_test("test_contains_figure2", "Contains Figure 2: Training Accuracy")
def _():
    assert "Figure 2" in report, "Figure 2 not found"
    assert "Training Accuracy" in report

@run_test("test_figures_have_data", "Figures contain actual data (not blank)")
def _():
    # Check that ascii charts contain asterisks (data points)
    assert report.count("*") >= 10, "Figures appear to be blank (too few data points)"

@run_test("test_comparison_values_populated", "Comparison table has paper and reproduced values")
def _():
    assert "95.2%" in report, "Paper target accuracy not in report"
    assert "95.5%" in report, "Reproduced accuracy not in report"
    assert "+0.3%" in report, "Difference not in report"

@run_test("test_within_tolerance", "Reproduced accuracy within ±0.5% tolerance")
def _():
    assert "within" in report.lower() or "✓" in report, "Tolerance check not found"
    assert "✓ Yes" in report, "Should be within tolerance"

@run_test("test_discrepancy_analysis", "Discrepancy analysis section present")
def _():
    assert "Discrepancy Analysis" in report
    assert "Possible sources" in report or "possible sources" in report

@run_test("test_training_summary", "Training summary table present with correct values")
def _():
    assert "Total Epochs" in report
    assert "100" in report  # 100 epochs
    assert "Final Training" in report

@run_test("test_conclusion_present", "Conclusion section present")
def _():
    assert "Conclusion" in report
    assert "succeeded" in report.lower()

@run_test("test_ascii_chart_function", "ASCII chart renders non-empty output")
def _():
    chart = _ascii_chart([0.5, 0.3, 0.1], title="Test", width=20, height=5)
    assert len(chart) > 0
    assert "*" in chart
    assert "Test" in chart

# Print results
print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)
passed = sum(1 for r in results if r["status"] == "passed")
failed = sum(1 for r in results if r["status"] == "failed")
for r in results:
    icon = "✓" if r["status"] == "passed" else "✗"
    print(f"  {icon} {r['name']}: {r['desc']}")
    if r["msg"]:
        print(f"    → {r['msg']}")
print(f"\n{passed}/{len(results)} passed, {failed} failed")
print("=" * 60)

if failed > 0:
    sys.exit(1)
