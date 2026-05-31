"""Tests for the LLM-response JSON extractor in orchestrator.agent.

The refiner / researcher / verifier all funnel through _extract_json.
Three failure modes recur in practice and each is exercised below:

  - Nested code fences inside a string value's markdown (e.g. refined_spec_md
    contains a python code block). The fenced-block regex must NOT truncate
    at the first inner fence.
  - Raw newlines / tabs / carriage returns inside string values. LLMs emit
    these instead of escape sequences constantly — JSON-invalid but the
    repair pass converts them.
  - Trailing commas in objects/arrays — common LLM typo.
"""

from research_builder.orchestrator.agent import (
    _balanced_array_span,
    _balanced_object_span,
    _extract_json,
    _extract_json_array,
    _repair_json_strings,
    _strip_trailing_commas,
)


def test_nested_code_fence_inside_string():
    src = (
        "Here is the plan:\n\n"
        "```json\n"
        '{\n  "refined_spec_md": "## §3.2\\n\\nExample:\\n\\n```python\\nx=1\\n```",\n'
        '  "summary": "ok",\n  "research_questions": []\n}\n'
        "```\n"
    )
    out = _extract_json(src)
    assert out["summary"] == "ok"
    assert "```python" in out["refined_spec_md"]


def test_raw_newlines_inside_string_value():
    src = '{"text": "line1\nline2\nline3", "n": 3}'
    out = _extract_json(src)
    assert out["n"] == 3
    assert out["text"] == "line1\nline2\nline3"


def test_trailing_commas_object():
    out = _extract_json('{"a": 1, "b": [1, 2,], }')
    assert out == {"a": 1, "b": [1, 2]}


def test_trailing_commas_array():
    out = _extract_json_array('[{"a":1}, {"b":2},]')
    assert out == [{"a": 1}, {"b": 2}]


def test_prose_wrapped_json():
    src = 'Prelude. {"k": "v", "i": 7} Postlude.'
    assert _extract_json(src) == {"k": "v", "i": 7}


def test_well_formed_fenced_block():
    out = _extract_json('```json\n{"x": 1}\n```')
    assert out == {"x": 1}


def test_no_json_returns_empty():
    assert _extract_json("I refuse to help.") == {}
    assert _extract_json_array("No array here.") == []


def test_braces_inside_string_dont_break_balanced_scan():
    src = 'noise {"text": "this has { and } inside", "k": 42} more noise'
    out = _extract_json(src)
    assert out["k"] == 42


def test_balanced_object_span_returns_none_when_no_object():
    assert _balanced_object_span("hello world") is None


def test_balanced_array_span_returns_none_when_no_array():
    assert _balanced_array_span("hello world") is None


def test_repair_strings_leaves_structure_intact():
    """The repair pass must NOT touch newlines outside string values."""
    src = '{\n  "k": "v",\n  "n": 1\n}'
    repaired = _repair_json_strings(src)
    # Structural newlines unchanged; no string contains a raw newline so
    # nothing should be modified.
    assert repaired == src


def test_repair_strings_escapes_only_inside_strings():
    src = '{"a": "x\ny", "b":\n2}'
    repaired = _repair_json_strings(src)
    # The newline inside "x\ny" becomes \\n; the structural newline between
    # the colon and the value stays as-is.
    assert '"x\\ny"' in repaired
    assert '":\n2' in repaired


def test_strip_trailing_commas_idempotent():
    src = '{"a":1,"b":[1,2]}'
    assert _strip_trailing_commas(src) == src
