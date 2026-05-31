"""Regenerate the bundled test_paper.pdf fixtures.

Run with:
    uv run --with reportlab python tools/make_test_paper.py

Writes identical PDFs to both:
    - custom-harness/paper/test_paper.pdf           (used by --test smoke run)
    - custom-harness/tests/fixtures/test_paper.pdf  (used by unit tests)
"""

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, PageBreak


PAGES: list[list[str]] = [
    [
        "<b>Test Paper: A Study of Widget Performance</b>",
        "<b>Abstract:</b> This paper studies widget performance under various "
        "conditions. We find that widgets perform best when properly configured.",
    ],
    [
        "<b>Section 2: Methods</b>",
        "We used a batch size of 32 and learning rate of 0.001. The model was "
        "trained for 100 epochs using AdamW optimizer with weight decay 0.01.",
    ],
    [
        "<b>Section 3: Compute Setup</b>",
        "All experiments were run on a single NVIDIA GPU. Before training, we "
        "allocate GPU memory by moving the model and a warmup batch to the "
        "device, then run a short CUDA availability check script that verifies "
        "<i>torch.cuda.is_available()</i> returns True, prints the device name "
        "via <i>torch.cuda.get_device_name(0)</i>, and confirms the active "
        "device index. After training and evaluation finish, we deallocate the "
        "GPU memory by deleting model and optimizer references and calling "
        "<i>torch.cuda.empty_cache()</i> to release cached blocks back to the "
        "driver.",
    ],
    [
        "<b>Section 4: Results</b>",
        "The model achieved 95.2% accuracy on the test set. Table 1 shows the "
        "full results across all benchmarks.",
    ],
]


def build(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=LETTER,
        leftMargin=72,
        rightMargin=72,
        topMargin=72,
        bottomMargin=72,
    )
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    flow: list = []
    for i, paragraphs in enumerate(PAGES):
        for j, text in enumerate(paragraphs):
            flow.append(Paragraph(text, body))
            if j < len(paragraphs) - 1:
                flow.append(Spacer(1, 12))
        if i < len(PAGES) - 1:
            flow.append(PageBreak())
    doc.build(flow)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    targets = [
        root / "paper" / "test_paper.pdf",
        root / "tests" / "fixtures" / "test_paper.pdf",
    ]
    for target in targets:
        build(target)
        print(f"wrote {target}")
