#!/usr/bin/env python3
"""Generate reproduction report from training logs and eval metrics."""

import json
import os

def load_json(path):
    with open(path) as f:
        return json.load(f)

def generate_report(metrics_path, training_log_path, output_path):
    metrics = load_json(metrics_path)
    training_log = load_json(training_log_path)

    # Extract key values
    test_acc = metrics["test_accuracy"]
    test_loss = metrics["test_loss"]
    num_test = metrics["num_test_samples"]
    num_correct = metrics["num_correct"]
    paper_target = metrics["paper_target_accuracy"]
    class_0_acc = metrics["class_0_accuracy"]
    class_1_acc = metrics["class_1_accuracy"]

    diff = test_acc - paper_target
    within_tolerance = abs(diff) <= 0.5

    # Training summary
    epochs = [e["epoch"] for e in training_log]
    losses = [e["train_loss"] for e in training_log]
    accs = [e["train_accuracy"] for e in training_log]
    final_train_acc = accs[-1]
    final_train_loss = losses[-1]
    best_train_acc = max(accs)
    min_train_loss = min(losses)

    # Generate ASCII training curves
    loss_curve = _ascii_chart(losses, title="Training Loss", width=60, height=15)
    acc_curve = _ascii_chart(accs, title="Training Accuracy", width=60, height=15)

    report = f"""# Reproduction Report

## 1. Overview

This report summarizes the reproduction of the paper's results. The paper reports a headline
test accuracy of **{paper_target}%**. Our reproduction achieved **{test_acc}%** test accuracy.

## 2. Training Summary

| Metric | Value |
|---|---|
| Total Epochs | {len(training_log)} |
| Final Training Loss | {final_train_loss:.6f} |
| Final Training Accuracy | {final_train_acc * 100:.1f}% |
| Best Training Accuracy | {best_train_acc * 100:.1f}% |
| Minimum Training Loss | {min_train_loss:.6f} |
| Learning Rate | {training_log[0]['lr']} |

## 3. Training Curves

### Figure 1: Training Loss over Epochs

```
{loss_curve}
```

### Figure 2: Training Accuracy over Epochs

```
{acc_curve}
```

## 4. Evaluation Results

### Table 1: Test Set Performance

| Metric | Reproduced | Paper-Reported |
|---|---|---|
| Test Accuracy | {test_acc}% | {paper_target}% |
| Test Loss | {test_loss:.4f} | — |
| Num Test Samples | {num_test} | — |
| Num Correct | {num_correct} | — |

### Table 2: Per-Class Accuracy

| Class | Accuracy |
|---|---|
| Class 0 | {class_0_acc * 100:.1f}% |
| Class 1 | {class_1_acc * 100:.1f}% |

## 5. Comparison with Paper

| Metric | Paper | Reproduced | Difference | Within ±0.5%? |
|---|---|---|---|---|
| Test Accuracy | {paper_target}% | {test_acc}% | {diff:+.1f}% | {"✓ Yes" if within_tolerance else "✗ No"} |

### Discrepancy Analysis

The reproduced test accuracy of **{test_acc}%** differs from the paper-reported value of
**{paper_target}%** by **{diff:+.1f}%**.

{"This difference is **within** the acceptable tolerance of ±0.5%, indicating a successful reproduction." if within_tolerance else "This difference **exceeds** the acceptable tolerance of ±0.5%, indicating a potential reproduction gap."}

Possible sources of minor variation:
- Random seed differences in data splitting and weight initialization
- Minor numerical differences across hardware/software environments
- The paper does not provide full hyperparameter details or exact data splits

### Limitations and Gaps

- **No ablation studies**: The paper does not report ablation studies, so we cannot assess
  individual component contributions.
- **No baseline comparisons**: The paper does not provide baseline results for comparison.
- **Table 1 data missing**: The paper references Table 1 with full benchmark results, but its
  contents were not available from the provided text.

## 6. Conclusion

The reproduction {"**succeeded**" if within_tolerance else "**partially succeeded**"}. The reproduced
test accuracy of {test_acc}% is {f"within the ±0.5% tolerance of the paper-reported {paper_target}%" if within_tolerance else f"outside the ±0.5% tolerance of the paper-reported {paper_target}%"}.
The model trained for {len(training_log)} epochs, converging to a final training accuracy of
{final_train_acc * 100:.1f}% with a loss of {final_train_loss:.6f}.
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    return report


def _ascii_chart(values, title="", width=60, height=15):
    """Render a simple ASCII line chart."""
    if not values:
        return "(no data)"

    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0

    lines = []
    lines.append(title)
    lines.append("")

    for row in range(height):
        threshold = mx - (row / (height - 1)) * rng
        label = f"{threshold:8.4f} |"
        chars = []
        # Sample `width` points from values
        for col in range(width):
            idx = int(col * (len(values) - 1) / (width - 1))
            val = values[idx]
            if row == height - 1:
                chars.append("_")
            elif abs(val - threshold) <= rng / (height - 1) / 2:
                chars.append("*")
            elif val >= threshold:
                chars.append("*" if row == 0 else " ")
            else:
                chars.append(" ")
        lines.append(label + "".join(chars))

    # X-axis
    lines.append("         |" + "_" * width)
    lines.append(f"          Epoch 1{' ' * (width - 14)}Epoch {len(values)}")

    return "\n".join(lines)


if __name__ == "__main__":
    base = "/private/tmp/research-builder-test"
    generate_report(
        metrics_path=f"{base}/phases/eval/1/outputs/metrics.json",
        training_log_path=f"{base}/phases/training/1/outputs/training_log.json",
        output_path=f"{base}/phases/results/1/outputs/reproduction_report.md",
    )
    # Also copy to results_summary.md as expected output
    import shutil
    shutil.copy(
        f"{base}/phases/results/1/outputs/reproduction_report.md",
        f"{base}/phases/results/1/outputs/results_summary.md",
    )
    print("Report generated successfully.")
