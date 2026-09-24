"""Tests for LLM client, response validation, judge parsing, and overwrite protection."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import pytest

from ghost.client import (
    JudgeOutcome,
    LLMClient,
    clean_llm_response,
    ensure_can_overwrite_test,
    is_ghost_managed_test,
    parse_judge_response,
    validate_test_code,
)
from ghost.config import GhostConfig
from ghost.errors import HandwrittenTestOverwriteError, UnparseableCodeError
from ghost.providers import BaseProvider


class DummyProvider(BaseProvider):
    """Simple mock provider returning programmed responses."""

    def __init__(self, response: str = "") -> None:
        super().__init__()
        self.response = response
        self.calls: list[dict[str, Any]] = []

    @property
    @override
    def name(self) -> str:
        return "dummy"

    @override
    async def _call_api(
        self, messages: list[dict[str, str]], model: str, temperature: float
    ) -> str:
        self.calls.append({"messages": messages, "model": model, "temperature": temperature})
        return self.response

    @override
    async def list_models(self) -> list[str]:
        return ["dummy-model"]

    @override
    async def is_available(self) -> bool:
        return True


def test_clean_llm_response() -> None:
    fenced_python = "```python\ndef test_fn():\n    assert True\n```"
    assert clean_llm_response(fenced_python) == "def test_fn():\n    assert True"

    fenced_generic = "```\ndef test_fn():\n    assert True\n```"
    assert clean_llm_response(fenced_generic) == "def test_fn():\n    assert True"

    raw = "   def test_fn():\n    assert True\n   "
    assert clean_llm_response(raw) == "def test_fn():\n    assert True"


def test_validate_test_code_valid_and_unparseable() -> None:
    valid_code = "def test_pass():\n    assert 1 == 1\n"
    ast_tree = validate_test_code(valid_code, "test_pass.py")
    assert ast_tree is not None

    invalid_code = "def broken( incomplete syntax"
    with pytest.raises(UnparseableCodeError) as exc_info:
        validate_test_code(invalid_code, "test_broken.py")
    assert "LLM generated invalid Python syntax" in str(exc_info.value)
    assert "test_broken.py" in str(exc_info.value)


def test_handwritten_test_overwrite_protection(tmp_path: Path) -> None:
    """Regression test: Ghost must never overwrite hand-written test files without asking."""
    ghost_test = tmp_path / "test_ghost.py"
    ghost_test.write_text(
        "# Generated at: 24-09-2026 12:00:00 | Source: app.py\ndef test_a(): pass\n",
        encoding="utf-8",
    )
    assert is_ghost_managed_test(ghost_test)
    # Overwrite permitted for Ghost-managed test
    ensure_can_overwrite_test(ghost_test, force=False)

    human_test = tmp_path / "test_human.py"
    human_test.write_text(
        "import pytest\n\ndef test_manual():\n    assert True\n",
        encoding="utf-8",
    )
    assert not is_ghost_managed_test(human_test)

    # Refuses to overwrite hand-written test without --force
    with pytest.raises(HandwrittenTestOverwriteError) as exc_info:
        ensure_can_overwrite_test(human_test, force=False)
    assert "refusing to overwrite hand-written test file" in str(exc_info.value)

    # Permitted when force=True
    ensure_can_overwrite_test(human_test, force=True)


def test_parse_judge_response_tripartite_classification() -> None:
    """Regression test: Judge response classification must handle UNCLEAR safely."""
    assert parse_judge_response("BUG_IN_CODE") == JudgeOutcome.BUG_IN_CODE
    assert parse_judge_response("   bug_in_code   ") == JudgeOutcome.BUG_IN_CODE
    assert parse_judge_response("The answer is BUG_IN_CODE.") == JudgeOutcome.BUG_IN_CODE

    assert parse_judge_response("FIX_TEST") == JudgeOutcome.FIX_TEST
    assert parse_judge_response("   fix_test\n") == JudgeOutcome.FIX_TEST

    # Third outcome: UNCLEAR for contradictory or unclassifiable answers
    assert (
        parse_judge_response("I think it is BUG_IN_CODE but also FIX_TEST") == JudgeOutcome.UNCLEAR
    )
    assert parse_judge_response("Maybe try re-running") == JudgeOutcome.UNCLEAR
    assert parse_judge_response("") == JudgeOutcome.UNCLEAR
    assert parse_judge_response("UNCLEAR") == JudgeOutcome.UNCLEAR


@pytest.mark.asyncio
async def test_llm_client_generate_test_success(tmp_path: Path) -> None:
    src_file = tmp_path / "app.py"
    src_file.write_text("def hello() -> str: return 'hi'\n", encoding="utf-8")

    generated_code = (
        "```python\n"
        "# Generated at: 24-09-2026 12:00:00 | Source: app.py\n"
        "import pytest\n"
        "def test_hello(): assert True\n"
        "```"
    )
    provider = DummyProvider(response=generated_code)
    client = LLMClient(provider=provider, config=GhostConfig())

    code = await client.generate_test(source_path=src_file, project_root=tmp_path)
    assert "def test_hello():" in code
    assert not code.startswith("```")
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_llm_client_generate_test_unparseable_raises(tmp_path: Path) -> None:
    src_file = tmp_path / "app.py"
    src_file.write_text("def hello() -> str: return 'hi'\n", encoding="utf-8")

    provider = DummyProvider(response="def broken_syntax( pass")
    client = LLMClient(provider=provider, config=GhostConfig())

    with pytest.raises(UnparseableCodeError):
        await client.generate_test(source_path=src_file, project_root=tmp_path)


@pytest.mark.asyncio
async def test_llm_client_consult_the_judge(tmp_path: Path) -> None:
    src_file = tmp_path / "app.py"
    src_file.write_text("def buggy(): raise ValueError()\n", encoding="utf-8")
    test_file = tmp_path / "test_app.py"
    test_file.write_text("def test_buggy(): buggy()\n", encoding="utf-8")

    provider = DummyProvider(response="BUG_IN_CODE")
    client = LLMClient(provider=provider, config=GhostConfig())

    outcome = await client.consult_the_judge(
        source_path=src_file,
        test_file=test_file,
        error_output="ValueError",
    )
    assert outcome == JudgeOutcome.BUG_IN_CODE


@pytest.mark.asyncio
async def test_llm_client_heal_test_success(tmp_path: Path) -> None:
    src_file = tmp_path / "app.py"
    src_file.write_text("def hello() -> str: return 'hi'\n", encoding="utf-8")
    test_file = tmp_path / "test_app.py"
    test_file.write_text("def test_hello(): assert False\n", encoding="utf-8")

    healed_code = (
        "# Generated at: 24-09-2026 12:00:00 | Source: app.py\n"
        "import pytest\n"
        "def test_hello(): assert True\n"
    )
    provider = DummyProvider(response=healed_code)
    client = LLMClient(provider=provider, config=GhostConfig())

    code = await client.heal_test(
        source_path=src_file,
        test_file=test_file,
        error_output="AssertionError: assert False",
        project_root=tmp_path,
    )
    assert "def test_hello(): assert True" in code
    assert len(provider.calls) == 1
