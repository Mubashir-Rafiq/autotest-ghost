"""Golden-file and contract tests for pure prompt templates."""

from __future__ import annotations

from ghost.prompts import (
    HEADER_PREFIX,
    SYS_PATH_SNIPPET,
    build_generation_prompt,
    build_healing_prompt,
    build_judge_prompt,
    format_header,
)


def test_format_header_fixed_timestamp() -> None:
    header = format_header("src/core/app.py", timestamp="24-09-2026 12:00:00")
    assert header == f"{HEADER_PREFIX} 24-09-2026 12:00:00 | Source: src/core/app.py"


def test_generation_prompt_golden_contract() -> None:
    """Golden-file contract test: generation prompt format must be strictly deterministic."""
    source_code = "def add(x: int, y: int) -> int:\n    return x + y\n"
    source_path = "src/math.py"
    project_tree = "src/\n└── math.py"
    context = {"src/math.py": "Functions: add(x: int, y: int) -> int; Classes: None"}
    fixed_ts = "01-01-2026 00:00:00"

    prompt = build_generation_prompt(
        source_code=source_code,
        source_path=source_path,
        project_tree=project_tree,
        context=context,
        framework="pytest",
        timestamp=fixed_ts,
    )

    # Core non-negotiable assertions from SPEC §7.1
    assert "Output RAW PYTHON CODE ONLY" in prompt
    assert f"# Generated at: {fixed_ts} | Source: {source_path}" in prompt
    assert SYS_PATH_SNIPPET in prompt
    assert "Explicitly import pytest." in prompt
    assert "PROJECT TREE:\nsrc/\n└── math.py" in prompt
    assert '"src/math.py": "Functions: add(x: int, y: int) -> int; Classes: None"' in prompt
    assert "SOURCE CODE UNDER TEST (`src/math.py`):\ndef add(x: int, y: int) -> int:" in prompt

    # Verify byte-for-byte exact equality on repeated calls with same inputs (pure function)
    prompt_repeat = build_generation_prompt(
        source_code=source_code,
        source_path=source_path,
        project_tree=project_tree,
        context=context,
        framework="pytest",
        timestamp=fixed_ts,
    )
    assert prompt == prompt_repeat


def test_healing_prompt_golden_contract() -> None:
    """Golden-file contract test: healing prompt format must be strictly deterministic."""
    source_code = "def divide(a: float, b: float) -> float:\n    return a / b\n"
    source_path = "src/calc.py"
    test_code = "def test_divide():\n    assert divide(10, 0) == 0\n"
    error_output = "ZeroDivisionError: division by zero"
    project_tree = "src/\n└── calc.py"
    context = {"src/calc.py": "Functions: divide(a: float, b: float) -> float; Classes: None"}
    fixed_ts = "01-01-2026 00:00:00"

    prompt = build_healing_prompt(
        source_code=source_code,
        source_path=source_path,
        test_code=test_code,
        error_output=error_output,
        project_tree=project_tree,
        context=context,
        framework="pytest",
        return_code=1,
        timestamp=fixed_ts,
    )

    # SPEC §7.2 requirements
    assert "Fix ONLY what is broken" in prompt
    assert "Preserve all existing valid test coverage" in prompt
    assert f"# Generated at: {fixed_ts} | Source: {source_path}" in prompt
    assert SYS_PATH_SNIPPET in prompt
    assert "FAILING TEST CODE:\ndef test_divide():" in prompt
    assert "TEST RUN FAILURE OUTPUT (exit code 1):\nZeroDivisionError: division by zero" in prompt

    # Determinism
    prompt_repeat = build_healing_prompt(
        source_code=source_code,
        source_path=source_path,
        test_code=test_code,
        error_output=error_output,
        project_tree=project_tree,
        context=context,
        framework="pytest",
        return_code=1,
        timestamp=fixed_ts,
    )
    assert prompt == prompt_repeat


def test_judge_prompt_golden_contract() -> None:
    """Golden-file contract test: judge prompt format must be strictly deterministic."""
    source_code = "def broken(): raise RuntimeError('bug')\n"
    test_code = "def test_broken(): assert broken() == 1\n"
    error_output = "RuntimeError: bug"

    prompt = build_judge_prompt(
        source_code=source_code,
        test_code=test_code,
        error_output=error_output,
    )

    # SPEC §7.3 requirements
    assert "BUG_IN_CODE" in prompt
    assert "FIX_TEST" in prompt
    assert "Output EXACTLY one of these two tokens:" in prompt
    assert "SOURCE CODE UNDER TEST:\ndef broken(): raise RuntimeError('bug')" in prompt
    assert "FAILING TEST CODE:\ndef test_broken(): assert broken() == 1" in prompt
    assert "FAILURE ERROR OUTPUT:\nRuntimeError: bug" in prompt

    # Determinism
    prompt_repeat = build_judge_prompt(
        source_code=source_code,
        test_code=test_code,
        error_output=error_output,
    )
    assert prompt == prompt_repeat
