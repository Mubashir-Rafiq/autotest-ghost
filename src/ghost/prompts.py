"""Pure prompt templates and prompt construction for test generation, healing, and judging.

Owns:
- Constructing test generation prompts with strict adherence to system constraints.
- Constructing healing prompts containing failure context and failing test code.
- Constructing judge prompts to evaluate bug-in-code vs fix-test.
- Fixed-format timestamp headers and sys.path injection blocks.
- Deterministic output suitable for golden-file testing.

Does NOT:
- Call LLM APIs or handle networking (that belongs to :class:`~ghost.client.LLMClient`).
- Read or write files from disk.
- Parse or clean responses.

Guarantees:
- Pure functions: identical inputs always yield identical prompt strings.
- Enforces raw-code-only output instructions, correct sys.path prepend, explicit imports,
  and context isolation.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

__all__ = [
    "HEADER_PREFIX",
    "SYS_PATH_SNIPPET",
    "build_generation_prompt",
    "build_healing_prompt",
    "build_judge_prompt",
    "format_header",
]

HEADER_PREFIX = "# Generated at:"

SYS_PATH_SNIPPET = (
    "import sys\n"
    "import os\n"
    "sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))"
)


def format_header(source_path: str, timestamp: str | None = None) -> str:
    """Format the mandatory Ghost test file header comment."""
    ts = timestamp or datetime.now(tz=UTC).strftime("%d-%m-%Y %H:%M:%S")
    return f"{HEADER_PREFIX} {ts} | Source: {source_path}"


def build_generation_prompt(
    source_code: str,
    source_path: str,
    project_tree: str,
    context: dict[str, str],
    *,
    framework: str = "pytest",
    timestamp: str | None = None,
) -> str:
    """Build a deterministic prompt requesting the LLM to generate tests for *source_code*."""
    header = format_header(source_path, timestamp=timestamp)
    context_str = json.dumps(context, indent=2, sort_keys=True)

    return f"""You are an expert Python test engineer. Write a complete, production-ready \
{framework} test suite for `{source_path}`.

CRITICAL INSTRUCTIONS -- YOU MUST ADHERE TO EVERY SINGLE RULE:
1. Output RAW PYTHON CODE ONLY. Do NOT use markdown code blocks (no ```python or ```).
   No introduction, no markdown formatting, no explanations, no outro.
2. The file MUST begin with this exact header comment on line 1:
{header}
3. Immediately below the header, include this exact sys.path prepend before any other imports:
{SYS_PATH_SNIPPET}
4. Explicitly import {framework}.
5. Import only from: Python standard library, `{framework}`, or modules listed in GLOBAL CONTEXT \
below. Never use wildcard imports (no `from ... import *`). Never reference files or symbols \
absent from GLOBAL CONTEXT.
6. Write one test function or class per public function/class. Cover standard behavior, edge \
cases, and error/failure paths.
7. Use explicit assertions (e.g. `assert result == expected`), not bare truthiness assertions.
8. Mock ALL external dependencies: filesystem, network, environment variables, system time, \
randomness, UUIDs, subprocesses.
9. Do NOT modify, rewrite, or inline the source code under test.

PROJECT TREE:
{project_tree}

GLOBAL CONTEXT:
{context_str}

SOURCE CODE UNDER TEST (`{source_path}`):
{source_code}
"""


def build_healing_prompt(
    source_code: str,
    source_path: str,
    test_code: str,
    error_output: str,
    *,
    project_tree: str,
    context: dict[str, str],
    framework: str = "pytest",
    return_code: int = 1,
    timestamp: str | None = None,
) -> str:
    """Build a deterministic prompt requesting the LLM to heal failing test code."""
    header = format_header(source_path, timestamp=timestamp)
    context_str = json.dumps(context, indent=2, sort_keys=True)

    return f"""You are an expert Python test engineer. A previously generated {framework} test for \
`{source_path}` failed. Fix the test so that it passes completely.

CRITICAL INSTRUCTIONS -- YOU MUST ADHERE TO EVERY SINGLE RULE:
1. Output RAW PYTHON CODE ONLY. Do NOT use markdown code blocks (no ```python or ```).
   No introduction, no markdown formatting, no explanations, no outro.
2. Fix ONLY what is broken: correct syntax errors, missing imports, bad mocks, or incorrect \
expectations.
3. Preserve all existing valid test coverage, assertions, and test structure.
4. The file MUST begin with this exact header comment on line 1:
{header}
5. Immediately below the header, include this exact sys.path prepend before any other imports:
{SYS_PATH_SNIPPET}
6. Explicitly import {framework}.
7. Import only from: Python standard library, `{framework}`, or modules listed in GLOBAL CONTEXT \
below.
8. Mock ALL external dependencies: filesystem, network, environment variables, time, randomness, \
UUIDs, subprocesses.
9. Do NOT modify or inline the source code under test.

SOURCE CODE UNDER TEST (`{source_path}`):
{source_code}

FAILING TEST CODE:
{test_code}

TEST RUN FAILURE OUTPUT (exit code {return_code}):
{error_output}

PROJECT TREE:
{project_tree}

GLOBAL CONTEXT:
{context_str}
"""


def build_judge_prompt(
    source_code: str,
    test_code: str,
    error_output: str,
) -> str:
    """Build a deterministic prompt for the Judge step to decide BUG_IN_CODE vs FIX_TEST."""
    return f"""You are an impartial, expert Python code judge evaluating a test failure.

Determine whether the failure is caused by a real defect in the source code (`BUG_IN_CODE`) or by \
a flaw/mistake in the test file (`FIX_TEST`).

OUTPUT REQUIREMENTS:
- Output EXACTLY one of these two tokens:
  BUG_IN_CODE
  FIX_TEST
- Output NOTHING ELSE. No explanation, no punctuation, no markdown fences, no thinking process.
- If the source code has a legitimate bug, syntax error, or unhandled exception that causes the \
test to fail, output: BUG_IN_CODE
- If the test makes incorrect assertions, mocks improperly, has bad imports, or assumes wrong \
behavior, output: FIX_TEST

SOURCE CODE UNDER TEST:
{source_code}

FAILING TEST CODE:
{test_code}

FAILURE ERROR OUTPUT:
{error_output}
"""
