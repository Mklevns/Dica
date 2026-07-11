"""Tests for hardened code extraction (C1-critical path)."""

from __future__ import annotations

from dica.extraction import extract_python_code
from dica.pipeline import unwrap_extraction
from tests.conftest import FENCE, fenced_python


def test_valid_fence_ok() -> None:
    result = extract_python_code(fenced_python("x = 1\nprint(x)"))
    assert result.ok is True
    assert result.code is not None
    assert "x = 1" in result.code
    assert unwrap_extraction(result) == result.code


def test_broken_fence_ok_false_but_code_present() -> None:
    """Unparseable fence must fail the gate while keeping the snippet."""
    resp = f"Here:\n{FENCE}python\ndef foo(\n{FENCE}\n"
    result = extract_python_code(resp)
    assert result.ok is False
    assert result.code == "def foo("
    assert result.error is not None
    assert "SyntaxError" in result.error or "parse" in result.error.lower()
    # C1: unwrap must NOT treat this as success
    assert unwrap_extraction(result) is None


def test_no_fence() -> None:
    result = extract_python_code("sorry, I cannot help with that")
    assert result.ok is False
    assert result.code is None
    assert unwrap_extraction(result) is None


def test_salvage_leading_prose() -> None:
    body = "Here is the code:\nx = 1\n"
    result = extract_python_code(fenced_python(body))
    assert result.ok is True
    assert result.code is not None
    assert "x = 1" in result.code
    assert "Here is the code" not in (result.code or "")


def test_salvage_trailing_prose() -> None:
    body = "x = 1\nLet me know if you need anything else.\n"
    result = extract_python_code(fenced_python(body))
    assert result.ok is True
    assert result.code is not None
    assert "x = 1" in result.code
    assert "Let me know" not in (result.code or "")


def test_salvage_both_edges() -> None:
    body = "Sure, here you go:\nx = 1\nHope that helps!\n"
    result = extract_python_code(fenced_python(body))
    assert result.ok is True
    assert result.code is not None
    assert result.code.strip() == "x = 1"
    assert "Sure" not in (result.code or "")
    assert "Hope" not in (result.code or "")


def test_unclosed_fence_truncated() -> None:
    resp = f"prefix\n{FENCE}python\nx = 42\n"
    result = extract_python_code(resp)
    assert result.ok is True
    assert result.code is not None
    assert "x = 42" in result.code


def test_largest_parseable_block_wins() -> None:
    small = fenced_python("a = 1")
    large = fenced_python("def ok():\n    return 1\n")
    broken = f"{FENCE}python\ndef broken(\n{FENCE}"
    result = extract_python_code(small + "\n" + large + "\n" + broken)
    assert result.ok is True
    assert result.code is not None
    assert "def ok" in result.code


def test_unwrap_none() -> None:
    assert unwrap_extraction(None) is None
