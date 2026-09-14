# Ghost — Rebuild Specification

This is a complete technical blueprint of Ghost, extracted from the working
codebase in this repo, written so you can rebuild the tool from scratch
**without** needing to re-read the original source line by line. Use it
together with [`IMPROVEMENTS.md`](IMPROVEMENTS.md):

- **This file** = what to build (the product's behavior, contracts, schemas,
  algorithms — the "current, working baseline").
- **`IMPROVEMENTS.md`** = how to build it *better* (concrete upgrades —
  architecture fixes, modernization, new features — organized by effort).

The intended workflow is to implement against this spec while applying the
relevant `IMPROVEMENTS.md` item at the point it's relevant, rather than
building the baseline first and retrofitting improvements after. Where this
spec and `IMPROVEMENTS.md` disagree (e.g. this spec describes today's
`GhostConfig` dataclass, `IMPROVEMENTS.md` §2.1 recommends `pydantic-settings`
instead), treat `IMPROVEMENTS.md` as the deciding vote — it was written by
reviewing this exact spec's baseline and represents "do it this way instead."

(This document reflects the codebase *after* the "quick wins" batch from
`IMPROVEMENTS.md` §1 has already been applied — framework-aware test runner,
the `auto_heal`/`use_judge` config-drift fix, the content-hash cache, the
`rich`-based spinner, CI/pre-commit, and a corrected Anthropic model
registry. It does **not** yet reflect §2/§3's larger architectural changes —
those are still improvements to make, not baseline behavior.)

---

## 1. Product Summary

Ghost is a **local-first CLI tool** that watches a Python project's source
files and, on every save, uses an LLM to generate a `pytest` (or
`unittest`-compatible) test file for the changed file, runs it in a real
subprocess, and — if it fails — attempts to autonomously fix it. A distinct
"Judge" step prevents the tool from silently rewriting a test to match
genuinely buggy source code: on an assertion failure, a second LLM call
decides whether the *source* or the *test* is wrong, and only rewrites the
test when the test itself is at fault.

Core design commitments to preserve in a rebuild:
1. **Multi-provider** — Groq, OpenAI, Anthropic, Ollama, LM Studio,
   OpenRouter, and generic OpenAI-compatible endpoints, behind one interface.
2. **Context-aware generation** — an AST-derived map of the project's
   functions/classes is injected into every prompt so generated imports and
   call signatures are grounded in what actually exists.
3. **Self-healing, with a safety valve** — syntax/runtime errors are
   auto-repaired; assertion failures go through the Judge instead of being
   blindly "fixed."
4. **Runs as a daemon or in the foreground** — `ghost start` detaches into a
   background process with its own PID file and rotating log; `ghost watch`
   runs attached to the terminal with live spinner output.
5. **Privacy-respecting** — Ollama/LM Studio need no API key and nothing
   leaves the machine.

---

## 2. Tech Stack & Packaging

- **Language:** Python, `requires-python = ">=3.11"` (the code uses stdlib
  `tomllib`, which doesn't exist before 3.11 — don't advertise 3.10 support).
- **Packaging:** `hatchling` build backend, distributed on PyPI, installed
  via `uv tool install` or `pip install`. Entry point: `ghost = "ghost.cli:main"`.
- **CLI framework:** `click` (command groups, one function per subcommand).
- **File watching:** `watchdog` (`Observer` + `FileSystemEventHandler`).
- **Terminal output:** `rich` for live-updating regions (spinners, countdown
  timers) — auto-detects non-TTY output (redirected files, CI logs, test
  capture) and degrades cleanly instead of emitting raw cursor-control
  sequences unconditionally.
- **LLM SDKs:** `groq`, `openai` (also used for Ollama/LM Studio/OpenRouter/
  custom endpoints, since they're all OpenAI-compatible), `anthropic`.
- **Config parsing:** stdlib `tomllib` (read-only; config files are written
  as raw formatted strings, not serialized back out).
- **Env loading:** `python-dotenv`.
- **Test runner (of the tool itself):** `pytest`, run via `uv run pytest`.
- **Dev tooling:** `black`, `isort`, `ruff`, `mypy`, `pre-commit`.
- **Lockfile/venv manager:** `uv`.

---

## 3. Repository Layout

```
ghost/
├── __init__.py        # Public API surface + __version__
├── cli.py              # click CLI: every `ghost <command>`
├── main.py             # Foreground watcher (`ghost watch`) + check_test/make_tests pipeline
├── daemon.py            # Background daemon (`ghost start`): PID file, signals, rotating log
├── change_tracker.py    # Content-hash cache — skip the pipeline for byte-identical re-saves
├── job_queue.py          # Debounce + single-worker FIFO queue for file events
├── init.py               # AST scanner: builds/maintains .ghost/context.json
├── config.py             # ghost.toml + .env + env-var loading into GhostConfig
├── chat.py               # Prompt construction, LLM call, response cleanup, the Judge
├── providers.py          # BaseProvider + 7 concrete providers + model registry
├── runner.py             # Runs the generated test in a subprocess; classifies failures
├── rate_limiter.py       # Global rate limiter + exponential-backoff retry decorator
└── console.py            # Terminal output: colors, icons, spinners, countdown, banners
tests/                    # pytest suite for Ghost itself, one file per module above
```

Module dependency direction (no cycles): `cli.py` → `main.py`/`daemon.py` →
`chat.py` + `runner.py` + `change_tracker.py` + `job_queue.py` → `config.py`
+ `providers.py` + `console.py`. `init.py` is called from `main.py`/
`daemon.py`/`cli.py` but doesn't depend on them.

---

## 4. Configuration System

### 4.1 `ghost.toml` — full schema

Written by `ghost init` into the project root; hand-editable afterward.

```toml
[project]
name = "my-app"              # str, used only for display
language = "python"          # str, currently always "python"

[ai]
provider = "groq"            # one of: groq | openai | ollama | anthropic | openrouter | lmstudio | custom
model = "llama-3.3-70b-versatile"   # provider-specific model id — SEE §11.4, THIS EXACT DEFAULT IS STALE
rate_limit_rpm = 30           # int, requests/minute — drives RateLimiter's minimum interval
# base_url = "http://localhost:11434/v1"  # optional, for local/custom endpoints — commented example only

[scanner]
ignore_dirs = [".venv", "venv", "node_modules", ".git", "__pycache__", "dist", "build", ".ghost", "tests", ".tox", ".pytest_cache", ".mypy_cache"]
ignore_files = ["setup.py", "conftest.py", "__init__.py"]

[tests]
framework = "pytest"          # "pytest" | "unittest" — validated by runner.run_test
output_dir = "tests"           # str, relative to project root
auto_heal = true                # bool — gates the regenerate-and-retry loop, NOT the first test run (see §7)
max_heal_attempts = 3            # int — upper bound on heal/retry iterations
use_judge = true                  # bool — gates the Judge consultation on assertion failures

[watcher]
debounce_seconds = 15              # int — how long a file must be quiet before the pipeline fires
patterns = ["*.py"]                 # list[str], declared but not actually consulted by the watchdog handler (it hardcodes .py filtering via CheckPath/_check_path instead — see §12)
```

This is what `ghost init`'s CLI command (`cli.py::_create_ghost_config`)
actually scaffolds — treat it as the canonical template. `AIConfig` (in
`config.py`) additionally defines `api_key`, `base_url`, `temperature`
(default `0.1`) and `max_retries` (default `5`) fields that **can** be set
by hand in `ghost.toml` and are read by `GhostConfig.from_dict`, but neither
`ghost init` code path currently scaffolds them into the generated file —
worth deciding deliberately in a rebuild whether the wizard/template should
expose them rather than leaving them as an undocumented manual-only escape
hatch.

Note there are **two** independent "write a default ghost.toml" code paths
that have drifted from each other — `ghost/init.py`'s standalone
`ghost_init()` function (not wired to the actual `ghost init` CLI command;
its template omits `use_judge`, `patterns`, and `__init__.py` from
`ignore_files`) versus `ghost/cli.py`'s `_create_ghost_config` (used by the
real `ghost init` command, shown above). A rebuild should have exactly
**one** template-writing function, not two.

### 4.2 Loading precedence (highest wins)

1. Environment variables (`GROQ_API_KEY`, `OPENAI_API_KEY`,
   `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, generic `GHOST_API_KEY`,
   `GHOST_BASE_URL`, `OLLAMA_HOST`) — loaded from a `.env` file in the
   project root first (via `python-dotenv`, `override=True`), then real
   process env vars.
2. `ghost.toml` values.
3. Dataclass defaults (`AIConfig`, `ScannerConfig`, `TestConfig`,
   `WatcherConfig` — see `ghost/config.py`).

`get_config(project_path=None)`: if no path given, searches upward from CWD
for the nearest `ghost.toml` (`find_project_root`); if none found anywhere,
returns all-defaults `GhostConfig()` rather than erroring.

### 4.3 API key resolution order

`get_api_key(provider)` checks, per provider, a small ordered list of env
var names (Groq additionally checks a legacy `GROQ_API_KEY3` name for
backward compatibility). `TestGenerator.provider` resolves the key as:
explicit constructor arg → `config.ai.api_key` (from `ghost.toml`) →
`get_api_key(provider_name)` (environment).

---

## 5. On-Disk State (`.ghost/`)

All of this lives under `<project_root>/.ghost/`, which is gitignored by the
template `ghost init` writes.

| Path | Written by | Purpose |
|---|---|---|
| `context.json` | `init.py` | `{filename: "Functions: f(a, b), g(); Classes: C [Methods: m1, m2]"}` — a flat per-file summary string, injected **wholesale** (`json.dumps(..., indent=2)`) into every LLM prompt. Rebuilt fully on `ghost init`; patched incrementally per-file on modify/delete during watch. |
| `hashes.json` | `change_tracker.py` | `{filename: sha256_hex}` — last-processed content hash per file, used to skip the pipeline on byte-identical re-saves. Deliberately **separate** from `context.json` so hashes never leak into a prompt. Written only *after* the pipeline completes (success or a handled failure) — never at the point of the change-check itself, so a crash mid-pipeline doesn't get mistaken for "already done." |
| `pid` | `daemon.py` | Daemon's PID, written atomically (`os.open` + `fsync`) so `ghost stop`/`ghost status` can find and signal it. Stale files (process no longer alive) are cleaned up automatically on next check. |
| `logs/ghost.log` | `daemon.py` | Rotating log (10 MB × 5 backups) — the daemon has no attached terminal, so all its output goes here (`stdout`/`stderr` are redirected to `/dev/null`). WARNING+ is also mirrored to stderr in case something is watching the raw process. |

Both `context.json` and `hashes.json` use the file's **basename only** as
the key (e.g. `"app.py"`, not the full path) — a known limitation: two files
with the same basename in different subdirectories collide. Preserve this
scheme for compatibility unless you deliberately fix it (worth doing in a
rebuild — see `IMPROVEMENTS.md` if it doesn't already call this out).

---

## 6. CLI Command Reference

All commands are under one `click.Group` (`ghost`). Global flag:
`--version`/`-v` prints version and exits before dispatching to any
subcommand.

| Command | Args/Options | Behavior |
|---|---|---|
| `ghost init [PATH]` | `--provider/-p {groq,openai,ollama,anthropic,openrouter,auto}` (default `auto`), `--model/-m TEXT`, `--framework/-f {pytest,unittest}` (default `pytest`) | Interactive wizard: confirms overwrite if `ghost.toml` exists; **always** prompts to pick/confirm a provider even if `-p` was passed (only the *default* changes); offers to reuse an existing env-var API key or prompts for one and offers to save it to `.env`; prompts for a model if `-m` wasn't given (pre-filled with a per-provider default); writes `ghost.toml`, creates `.ghost/`, and runs a full project scan (`walk_and_generate_json`) to seed `context.json`. |
| `ghost watch [PATH]` | `--verbose/-V` | Foreground watcher. Auto-runs `ghost init` first if no `ghost.toml` is found. Blocks on `Ctrl+C`. |
| `ghost start [PATH]` | `--detach/--foreground` (default `--detach`) | Detached mode: refuses to start a second daemon if one's already running (checked via PID file); launches `python -m ghost.daemon <path>` via `subprocess.Popen(start_new_session=True, stdin/stdout/stderr=DEVNULL)`; waits 0.5s and checks `proc.poll()` to catch immediate startup failure. Foreground mode: same as `ghost watch` in-process. |
| `ghost stop [PATH]` | — | Reads the PID file; sends `SIGTERM`; polls every 0.5s for up to 10s for the process to exit; sends `SIGKILL` if the grace period expires; removes the PID file. |
| `ghost status [PATH]` | — | Reports running/not-running + PID, and tails the last 10 log lines. |
| `ghost logs [PATH]` | `--follow/-f`, `--lines/-n INT` (default 20) | Without `--follow`: prints the last N lines. With `--follow`: seeks to EOF and polls for new lines every 0.1s (like `tail -f`), stopping automatically if the daemon process dies mid-follow. |
| `ghost generate FILE` | `--output/-o PATH`, `--force/-f` | One-shot: generate a test for exactly one file (not the file-watching pipeline). Finds the project root by walking up for `ghost.toml`. Prompts before overwriting an existing test file unless `--force`. Generates, writes, then immediately runs the test once via `runner.run_test` and reports pass/fail — **does not** invoke the heal/Judge loop (that only exists in the watch/daemon pipeline, `check_test`). |
| `ghost config` | `--show/-s`, `--set-provider {...}`, `--set-model TEXT`, `--set-api-key TEXT` | With no flags (or `--show`): pretty-prints the current `ghost.toml`. `--set-provider`/`--set-model` regex-replace the relevant line in `ghost.toml` in place (not a full TOML re-serialization — preserves comments/formatting, but is fragile against unusual formatting). `--set-api-key` doesn't write anything — it just prints the `export` command the user should run themselves (keys are meant to live in the environment, not the file). |
| `ghost providers` | `--check/-c` | Lists all 6 providers grouped as Local/Cloud, with availability (env var present for cloud; a live `is_available()` HTTP check for Ollama/LM Studio), plus a "Popular Models" table from the model registry. |
| `ghost version` | — | Prints version, Python version, platform. |
| `ghost doctor` | — | Health check: Python version, whether `groq`/`openai`/`watchdog`/`pytest`/`click` import successfully, whether `ghost.toml` exists in CWD, and which AI providers are currently available. |

`ghost` with no subcommand prints the mini-banner + `--help`.

---

## 7. Core Pipeline — Generate / Run / Classify / Heal / Judge

This is the heart of the tool and exists in **two** call sites that must
behave identically (a real bug in the original code let them drift — see
§7.4): `ghost/main.py::check_test` (foreground) and
`ghost/daemon.py`'s `_process_file` (background). Both follow this state
machine:

```
                         ┌─────────────────┐
                         │  Generate test   │  (only once, on first sight of the file)
                         └────────┬─────────┘
                                  │
                    ┌─────────────▼─────────────┐
        ┌──────────▶│   Run test (subprocess)    │◀───────────────┐
        │            └─────────────┬─────────────┘                │
        │                          │                               │
        │              return_code == 0?                          │
        │                    │           │                         │
        │                   yes          no                        │
        │                    │           │                         │
        │              ┌─────▼───┐  ┌────▼─────────────┐          │
        │              │ SUCCESS │  │ classify_error()   │          │
        │              └─────────┘  └────┬───────────────┘          │
        │                                │                          │
        │              ┌─────────────────┼──────────────────┐       │
        │       SYNTAX/RUNTIME/UNKNOWN   │            LOGIC   │       │
        │                │                │                   │       │
        │   attempt >= max_heal_attempts? │        use_judge?  │       │
        │        │            │           │          │      │        │
        │       yes           no          │         no     yes       │
        │        │            │           │          │      │        │
        │   ┌────▼───┐  ┌─────▼──────┐    │    ┌─────▼──┐ ┌─▼───────────┐
        │   │ GIVE UP│  │ Heal (LLM) │    │    │ GIVE UP│ │ Consult Judge│
        │   └────────┘  └─────┬──────┘    │    └────────┘ └──────┬───────┘
        │                     │           │                       │
        └─────────────────────┘           │           BUG_IN_CODE │ FIX_TEST
                                            │              │         │
                                            │         ┌────▼───┐    (same heal
                                            │         │ GIVE UP│     gate as
                                            │         │ (never │     above,
                                            │         │ touch  │     shares the
                                            │         │ the    │     attempt
                                            │         │ test)  │     counter)
                                            │         └────────┘        │
                                            └────────────────────────────┘
                                                    loop back to "Run test"
```

Precise rules (this is the **corrected** version — see §7.4 for what was
wrong before):

1. **Load config once**, before anything else — `framework`,
   `auto_heal`, `max_heal_attempts`, `use_judge` all come from it.
2. **The first test run always happens**, unconditionally, regardless of
   `auto_heal`. This matters: `auto_heal=False` should still tell you
   whether the freshly-generated test passes or fails, not silently skip
   running it.
3. `run_test(test_file_path, source_path, framework=...)` — runs
   `python -m pytest <test_file>` with `PYTHONPATH` set to the source root.
   (`framework` is validated against `{"pytest", "unittest"}` — pytest
   natively discovers and runs `unittest.TestCase`-based tests, so there is
   deliberately no separate unittest runner; the validation exists purely to
   catch a typo/unsupported value in a hand-edited `ghost.toml` loudly.)
4. If `return_code == 0`: success, stop. (Do **not** classify the output at
   all in this case — a passing test's stdout/stderr won't contain any of
   `classify_error`'s marker strings, so it would fall through to
   `"UNKNOWN"`, which the heal-attempt gate would otherwise wrongly treat as
   "needs healing." Always check the return code first.)
5. Otherwise, `classify_error(stderr, stdout)` (see §9 for the exact rules),
   then:
   - **`SYNTAX` / `RUNTIME` / `UNKNOWN`** → if the heal-attempt budget
     (`attempt_count < max_heal_attempts`, where `max_heal_attempts` is
     forced to `0` when `auto_heal` is `False`) is exhausted, stop and
     report; otherwise regenerate the test file with the error context
     included in the prompt, and loop back to step 3.
   - **`LOGIC`** → if `use_judge` is `False`, stop and report (never touch
     the test). Otherwise consult the Judge (a separate LLM call — see
     §7.3): `BUG_IN_CODE` stops immediately and reports the defect, without
     ever modifying the test; `FIX_TEST` regenerates the test (sharing the
     *same* attempt-count budget as the syntax/runtime heal path) and loops
     back to step 3.
6. The loop has no other exit — it only ends via a `return`/`break` in one
   of the branches above (success, budget exhausted, judge says
   `BUG_IN_CODE`, or an unhandled exception inside the try/except around the
   heal/judge logic, which is caught and reported as a failure).

### 7.1 Test generation prompt — hard requirements

`chat.py::TestGenerator.create_prompt` builds the initial-generation prompt.
Whatever LLM/prompt strategy you use, preserve these **non-negotiable**
constraints (the original prompt states them explicitly and repeatedly,
because LLMs violate them otherwise):

- Output must be **only** raw Python source — no markdown fences, no prose.
  (Defense in depth: `clean_llm_response` also regex-strips a
  ` ```python ... ``` ` block if the model ignores this instruction, and
  falls back to trimming whitespace otherwise.)
- The file's first lines must be a fixed-format header comment:
  `# Generated at: <DD-MM-YYYY HH:MM:SS> | Source: <source_path>`.
- Before any other import, the test file must contain exactly:
  ```python
  import sys
  import os
  sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
  ```
  followed by a normal (non-relative) import of the module under test.
- Must `import <framework>` explicitly.
- Import only from: the standard library, the test framework, or modules
  present in `context.json`'s `GLOBAL CONTEXT`. No wildcard imports, no
  referencing files/symbols absent from that context.
- One test per public function/class; cover standard behavior, edge cases,
  and error/failure paths; assert explicitly (no bare truthiness checks).
- Mock all external dependencies: filesystem, network, env vars, time,
  randomness, UUIDs, subprocesses.
- Must not modify, rewrite, or inline the source code under test.
- The current project tree (`get_project_tree`, filtered to `.py` files,
  ignoring venvs/caches/etc.) and the full `context.json` contents are both
  injected verbatim into the prompt.
- Called with `temperature=0.1` — deliberately low/deterministic, since
  creative variance is undesirable for code generation.

### 7.2 Healing prompt

`create_prompt_test` is the same shape but additionally includes: the
existing (broken) test file's full source, and a `dict` with
`return_code`/`stdout`/`stderr` from the failing run. Instructs the model to
fix *only* what's broken (syntax/import/runtime issues) while preserving
existing test logic and coverage intent, under the same import/mocking/
output-format rules as §7.1.

### 7.3 The Judge prompt

`consult_the_judge` sends: the source code, the *failing* test's full
source, and the error output, and asks the model to decide between exactly
two outputs — `BUG_IN_CODE` or `FIX_TEST` — with explicit instructions to
output nothing else (no punctuation, no explanation). The response is
upper-cased and substring-matched for either literal token; anything else is
returned as-is (effectively falls through to neither branch matching, which
the caller should treat defensively).

### 7.4 What was wrong before (context for why the pipeline looks like this)

The original implementation had `main.py::check_test` and `daemon.py`'s
inline equivalent implementing this same logic **twice**, and they'd
drifted:

- `main.py` ignored `auto_heal`/`use_judge`/`max_heal_attempts` entirely —
  hardcoded `while attempt_count < 3` and always attempted to heal/judge.
- `main.py` never checked `return_code` before classifying — meaning a
  **passing** test (whose output doesn't match any `classify_error` marker,
  so classifies as `"UNKNOWN"`) was treated as needing repair.
- `daemon.py` *did* respect the config flags, but computed
  `max_attempts = max_heal_attempts if auto_heal else 0` and wrapped the
  **entire** loop — including the first, always-should-happen test run — in
  `for attempt in range(max_attempts)`. With `auto_heal=False`,
  `max_attempts=0` meant the loop body (including the first run) never
  executed at all, while the `for...else` clause still logged "max healing
  attempts reached" — describing a test that was never actually run.

If you reintroduce a two-call-site split in a rebuild (foreground vs.
daemon), **extract this state machine into one shared function** (see
`IMPROVEMENTS.md` §3.1) rather than hand-syncing two copies — that's how
this drift happened in the first place.

---

## 8. AST Context Indexing (`init.py`)

Pure static analysis, no LLM involved. For every `.py` file (excluding
`scanner.ignore_dirs`/`ignore_files`):

1. `ast.parse` the source; on `SyntaxError`, skip the file entirely (return
   `None`, which callers treat as "don't index, log a warning").
2. Attach parent pointers to every node (`ast.walk` + `iter_child_nodes`) —
   necessary because stdlib `ast` doesn't track parents, and it's the only
   way to tell a *top-level* function apart from a method nested in a class
   during a single `NodeVisitor.visit_FunctionDef` pass.
3. `CodeAnalyzer(ast.NodeVisitor)` collects: top-level function signatures
   as `"name(arg1, arg2)"` strings (methods inside classes are intentionally
   excluded from the flat function list — they're captured separately, see
   next point), and a `{class_name: [method_names]}` map.
4. Serialize per file as one string:
   `"Functions: f(a, b), g(); Classes: C [Methods: m1, m2]"` (or `"None"`
   for either half if empty).
5. Whole-project version (`walk_and_generate_json`, used by `ghost init`)
   writes the full `{filename: summary}` map to `context.json`.
6. Incremental versions used during watching:
   - `walk_and_modify_json(base_dir, file_path, file)` — re-analyzes one
     file and patches just that key. Returns `None` (and does *not* touch
     `context.json`) if the file has a syntax error — callers use this as
     the signal to skip the generate/heal pipeline entirely for that event.
   - `walk_and_delete_json(base_dir, filename)` — removes one key; used on
     file-delete events (and paired with the change-tracker's own removal —
     see §12).

---

## 9. Error Classification (`runner.classify_error`)

Deliberately simple substring matching over `stdout + stderr` — not a
structural traceback parse. Checked in this exact order (first match wins):

| Substring found | Classification |
|---|---|
| `IndentationError` | `SYNTAX` |
| `ModuleNotFoundError` | `SYNTAX` |
| `ImportError` | `SYNTAX` |
| `AttributeError` | `RUNTIME` |
| `AssertionError` | `LOGIC` |
| *(none of the above)* | `UNKNOWN` |

`UNKNOWN` is treated identically to `SYNTAX`/`RUNTIME` by the pipeline (i.e.
"attempt to heal it") — this is why checking `return_code == 0` *before*
ever calling `classify_error` is load-bearing (§7, step 4): a passing test
has no reason to contain any of these markers and would otherwise
misclassify as `UNKNOWN` → "needs healing."

If you rebuild this, `IMPROVEMENTS.md` §2.3 recommends replacing this with
structured data from `pytest --json-report` or pytest's internal
`TestReport` objects instead of string-matching raw output — worth doing
from day one rather than retrofitting.

---

## 10. Rate Limiting & Retry

Two independent, composable mechanisms in `rate_limiter.py`:

- **`RateLimiter`** — class-level (i.e. process-global) state: a
  `threading.Lock`-guarded `_last_call` timestamp and a `MIN_INTERVAL`
  (seconds). `RateLimiter.wait()` is called at the top of every provider's
  `chat()` implementation (for cloud providers only — local Ollama/LM
  Studio skip it) and blocks (with an animated countdown) if called sooner
  than `MIN_INTERVAL` after the previous call. `TestGenerator.__init__` sets
  `MIN_INTERVAL = 60.0 / max(config.ai.rate_limit_rpm, 1)` — i.e. the
  interval is derived from `ghost.toml`'s `rate_limit_rpm`, not hardcoded
  (though the class docstring's "10s minimum" comment is aspirational/
  historical, not enforced as a floor).
- **`call_with_retry(max_retries=5, base_delay=2.0)`** — a decorator applied
  to every provider's `chat()` method. Catches exceptions, checks the
  stringified message for rate-limit indicators (`"429"`, `"rate limit"`,
  `"too many requests"`, `"quota exceeded"`, etc. — case-insensitive
  substring match), and if matched, sleeps
  `base_delay * 2**attempt + random.uniform(0.1, 1.0)` (exponential backoff
  with jitter — the jitter matters: without it, multiple rate-limited calls
  retry in lockstep and re-collide). Non-rate-limit exceptions are
  re-raised immediately without retry.

---

## 11. Provider Abstraction

### 11.1 Interface

```python
class BaseProvider(ABC):
    def __init__(self, api_key=None, base_url=None): ...
    @abstractmethod
    def _create_client(self): ...        # lazily instantiated via the `client` property
    @abstractmethod
    def chat(self, messages: list[dict], model: str, temperature: float = 0.1) -> str: ...
    @abstractmethod
    def list_models(self) -> list[str]: ...
```

`messages` follows the OpenAI chat-message shape
(`[{"role": "system"|"user", "content": str}, ...]`); each provider adapts
it to its own SDK's call shape internally (Anthropic notably splits the
`system` message out into a separate parameter rather than sending it as a
message).

### 11.2 Concrete providers

| Provider | SDK used | Needs API key? | Notes |
|---|---|---|---|
| Groq | `groq` | yes (`GROQ_API_KEY`) | Fastest/cheapest cloud option; free tier is aggressively rate-limited (hence `RateLimiter`'s existence). |
| OpenAI | `openai` | yes (`OPENAI_API_KEY`) | Standard chat completions. |
| Anthropic | `anthropic` | yes (`ANTHROPIC_API_KEY`) | System message handled separately from the `messages` list; `max_tokens=4096` hardcoded. |
| Ollama | `openai` SDK pointed at `http://localhost:11434/v1` | no | `api_key="ollama"` (placeholder, unchecked by Ollama). `list_models()` hits `/api/tags` live; falls back to a static list on failure. `is_available()` does a 2s-timeout GET to the same endpoint. |
| LM Studio | `openai` SDK pointed at `http://localhost:1234/v1` | no | Same shape as Ollama; `list_models()` just returns `["local-model"]` since LM Studio serves whatever's currently loaded. |
| OpenRouter | `openai` SDK pointed at `https://openrouter.ai/api/v1` | yes (`OPENROUTER_API_KEY`) | Sends extra `HTTP-Referer`/`X-Title` headers (OpenRouter convention); model IDs use OpenRouter's own `vendor/model` slug namespace, which is a *different* namespace from each vendor's native model IDs. |
| Custom | `openai` SDK | yes, plus a required `base_url` | Escape hatch for any other OpenAI-compatible endpoint. |

`get_provider(provider_type: str, **kwargs) -> BaseProvider` is a simple
dict-keyed factory (`{"groq": GroqProvider, ...}`); local providers
(`ollama`, `lmstudio`) have `api_key` stripped from kwargs before
construction since they don't accept one. `auto_detect_provider()` tries
Ollama then LM Studio (both free/local) before falling through.
`list_available_providers()` returns a `{name: bool}` status map (live HTTP
check for local providers, env-var presence for cloud ones) — this backs
`ghost providers`/`ghost doctor`.

### 11.3 Model registry (`POPULAR_MODELS`, `PROVIDER_MODELS`)

Two parallel data structures, used for different purposes — preserve both,
they're not redundant:

- `POPULAR_MODELS: dict[str, ModelConfig]` — display-only aliases (e.g. key
  `"claude-sonnet-5"` → `ModelConfig(name="claude-sonnet-5", provider=...,
  context_length=1_000_000, ...)`), rendered in `ghost providers`'s "Popular
  Models" table. Cosmetic; doesn't drive behavior.
- `PROVIDER_MODELS: dict[str, list[str]]` — per-provider lists of literal
  model-ID strings, used two ways: (1) shown as suggestions during
  `ghost init`'s model-selection prompt, and (2) `ghost init`'s
  `default_models` dict (in `cli.py`, a *third*, separate hardcoded mapping)
  supplies the pre-filled default when the user doesn't type a model —
  **this is the value that actually lands in a new user's `ghost.toml`**,
  making it the highest-impact place to keep current.

Current (verified) Anthropic model IDs, current as of this spec being
written: `claude-opus-5`, `claude-sonnet-5`, `claude-haiku-4-5` (no date
suffix — current-generation Anthropic model IDs don't carry one; only
older/retired pinned snapshots do). These appear in **five** separate
locations that must all agree: `POPULAR_MODELS`'s two Anthropic entries,
`PROVIDER_MODELS["anthropic"]`'s three-entry list, `AnthropicProvider.chat`'s
`model` default parameter, `AnthropicProvider.list_models()`'s return list,
and `cli.py`'s `default_models["anthropic"]`.

### 11.4 Known-stale model IDs to verify before shipping a rebuild

Model IDs drift constantly and none of this should be trusted as current
without checking each provider's live docs/API first. Two concrete,
**verified-stale-as-of-this-writing** findings from smoke-testing this exact
codebase, left unfixed because they were out of scope for the pass that
produced this spec (only Anthropic IDs were verified and corrected):

- **Groq**: every model ID currently hardcoded anywhere in this codebase
  (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`,
  `gemma2-9b-it`, `llama-guard-3-8b` — appearing in `GroqProvider.chat`'s
  default, `GroqProvider.list_models()`, `POPULAR_MODELS`,
  `PROVIDER_MODELS["groq"]`, and `cli.py`'s `default_models["groq"]`) is
  **dead** — a live call against Groq's `/v1/models` endpoint returned 404
  for all of them. Groq's catalog had moved entirely to different model
  families (e.g. `openai/gpt-oss-20b`, `groq/compound`, `qwen/...`) by the
  time this was checked. Query `https://api.groq.com/openai/v1/models`
  yourself before hardcoding a replacement.
- **OpenRouter**: its model list uses OpenRouter's own routing-slug
  namespace (e.g. `anthropic/claude-3.5-sonnet`), which is independently
  versioned from the vendor's native IDs — not verified/updated as part of
  fixing the Anthropic-native entries above; check OpenRouter's own model
  list before trusting these.
- OpenAI's entries (`gpt-4o`, `gpt-4o-mini`, etc.) were also left
  unverified for the same reason (no live check performed against OpenAI
  during the pass that produced this spec).

A rebuild should either fetch each provider's model list live (several
already expose a models-list endpoint — Groq and Ollama both do, and this
codebase already calls Ollama's) rather than hardcoding IDs at all, or at minimum
add a periodic-verification note/test like the one this codebase now has for
Anthropic (`tests/test_providers.py::TestAnthropicModelRegistry` — a
regression guard asserting specific retired snapshot IDs never reappear).

---

## 12. File Watching, Debouncing & Change Tracking

Three layered mechanisms, each solving a *different* problem — preserve all
three distinctly, don't collapse them:

1. **`watchdog.Observer`** — OS-level filesystem event source. The event
   handler filters events before they reach the pipeline at all:
   - Reject anything under a `tests/` directory (path segment match, not
     just filename) or ending in `/tests`.
   - Reject temp-file artifacts: paths ending in `~` (common editor
     swap-file suffix — stripped and re-checked, not just dropped) or
     `.py~`, and anything under `__pycache__`/ending in `.pyc`.
   - Reject filenames containing `"test"` or `"tmp"` (substring, not just
     prefix), `.gitconfig`-style dotfiles, `.log` files, and anything not
     ending in `.py`.
   - `on_deleted` triggers `walk_and_delete_json` (+ change-tracker removal,
     see below); `on_created` just logs; `on_modified` is the one that
     actually enqueues work.
2. **`TimerDebouncer`/`JobQueue`** (`job_queue.py`) — collapses *rapid*
   re-submits of the *same path* within one burst (e.g. an editor writing a
   file in several small syscalls fires multiple `on_modified` events for
   one logical save). Each path gets its own `threading.Timer`; a new event
   for a path cancels and restarts that path's timer; only after
   `debounce_seconds` of quiet does the job actually get pushed onto a
   `Queue` for a single dedicated worker thread to process in FIFO order
   (this — plus `RateLimiter` — is why there's only ever one LLM call in
   flight at a time in the current design; see `IMPROVEMENTS.md` §3.2 for
   deliberately parallelizing this).
3. **`change_tracker.py`** — solves a *different* problem than either of the
   above: skipping the pipeline for content that's genuinely unchanged
   *across separate, time-separated saves* (an editor's "format on save"
   running twice with no diff, a duplicate OS event outside any debounce
   window, reverting to previously-seen content). Computes
   `hashlib.sha256(content.encode()).hexdigest()`, compares against a
   `{filename: hash}` map persisted at `.ghost/hashes.json` (see §5).
   **Critical invariant: the new hash is written only after the pipeline
   completes** (success or a handled failure) — never at the point the
   comparison happens — so a crash mid-pipeline doesn't cause the next
   identical retry-save to be wrongly skipped. A missing/corrupt
   `hashes.json` is treated as "everything has changed" (fail open, not
   closed). Removed alongside `walk_and_delete_json` on file deletion.

---

## 13. Daemon Architecture (`daemon.py`)

Standard Unix background-service pattern:

- **Launch**: `ghost start` (no `--foreground`) spawns
  `python -m ghost.daemon <project_root>` via `subprocess.Popen` with
  `start_new_session=True` (calls `setsid()`, fully detaching from the
  launching terminal) and all three standard streams set to `DEVNULL`.
- **PID file**: written atomically — `os.open(O_CREAT|O_WRONLY|O_TRUNC)`,
  write, `os.fsync`, close — at `.ghost/pid`, both by the daemon itself on
  startup and (redundantly, to avoid a race window) by the launching CLI
  process right after `Popen` returns. Removed only if its content matches
  the *current* process's PID (so a stale/foreign PID file is never
  clobbered by a `stop` that raced with a restart).
- **Liveness check**: `check_pid(pid_file)` reads the PID and calls
  `os.kill(pid, 0)` (a no-op signal used purely to test whether the process
  exists / you have permission to signal it); an `OSError` means dead, and
  the stale file is cleaned up automatically as a side effect of checking.
- **Signal handling**: `SIGTERM`/`SIGINT` are wired to a handler that must
  be **async-signal-safe** — it does nothing but `threading.Event.set()`
  (no logging, no allocation, nothing else) — the actual shutdown sequence
  (stop the observer, join with a 10s timeout, remove the PID file) runs on
  the main thread, blocked on `event.wait()`, after the signal returns.
  `SIGHUP` is explicitly ignored (so closing the launching terminal doesn't
  kill it — though it's already detached via `setsid()` regardless).
- **Logging**: `RotatingFileHandler` (10MB × 5 backups, `delay=True` so the
  file isn't created until the first actual write) at `.ghost/logs/
  ghost.log`; WARNING+ also mirrored to stderr as a last-resort visibility
  path. `sys.stdout`/`sys.stderr` are redirected to `os.devnull` right after
  logging is configured, specifically to prevent any accidental `print()`
  deep in a dependency from going anywhere meaningful.
- **Watcher logic** inside the daemon is a near-duplicate of
  `main.py::start_watching`'s handler, adapted to log instead of using
  `console.py` (no attached terminal to render spinners to) — see §7.4 for
  why keeping these two copies in sync is a known trap.
- **`stop`**: SIGTERM, poll every 0.5s for up to 10s, SIGKILL if still alive
  after the grace period.

---

## 14. Testing Conventions (of Ghost itself)

- One test file per `ghost/` module (`tests/test_<module>.py`), using
  `pytest` with fixtures centralized in `tests/conftest.py`
  (`temp_project`, `ghost_config_content`, `temp_project_with_config`,
  `sample_python_source`, `mock_env_api_key`).
- Tests generally exercise real behavior against `tmp_path` (real files, a
  real subprocess for `run_test`, a real `hashlib` computation) rather than
  mocking internals — mocking is reserved for the genuinely expensive/
  external bits: LLM provider calls (constructor-level object-identity
  checks only, no live `.chat()` calls in the suite) and, where a test needs
  to isolate `check_test`'s control flow from needing a real subprocess +
  LLM, `monkeypatch.setattr` on the specific module-level names
  (`run_test`, `TestGenerator`, `WriteTest`) that a given function
  references.
- `pyproject.toml`'s `[tool.pytest.ini_options]`: `testpaths = ["tests"]`,
  standard `test_*.py`/`Test*`/`test_*` discovery, DeprecationWarnings
  filtered out.
- CI (`.github/workflows/ci.yml`) runs `pytest`, `ruff check`, and `mypy`
  across a 3.11/3.12/3.13 matrix on every push/PR; `pre-commit` runs
  `black`/`isort`/`ruff` locally on every commit. Both should stay green —
  treat a red CI/pre-commit run as a signal to fix the code, not loosen the
  gate.

---

## 15. Using This Spec Alongside `IMPROVEMENTS.md`

Suggested build order for a from-scratch rebuild:

1. Config system (§4) + on-disk state (§5) — everything else depends on
   `GhostConfig` existing, so build it first, and apply `IMPROVEMENTS.md`
   §2.1 (pydantic-settings) *while* building it rather than after.
2. Provider abstraction (§11) — build the interface and at least one real
   provider (Groq or Ollama, since Ollama needs no API key for local
   testing) before anything else touches it.
3. AST context indexing (§8) — self-contained, no LLM dependency, easy to
   get right early and test in isolation.
4. Core pipeline (§7) — this is where `IMPROVEMENTS.md` §3.1 (extract one
   shared `TestPipeline`, don't duplicate foreground/daemon logic) matters
   most: build it *once*, correctly, and have both the foreground watcher
   and the daemon call the same implementation from the start, rather than
   writing it twice and discovering the drift later the way this codebase
   did.
5. File watching/debouncing/change-tracking (§12) — layer these on top of
   a pipeline that already works when called directly/manually.
6. CLI (§6) + daemon process management (§13) — the user-facing shell
   around an already-working core.
7. Console/output layer (§ — see `console.py` in-repo) — deliberately last;
   it's pure presentation and easiest to get wrong early if you let it leak
   into the modules above it (this codebase keeps `Console`/`GhostSpinner`
   calls out of `runner.py`, `change_tracker.py`, `providers.py`, and
   `config.py` entirely — preserve that boundary).

Apply `IMPROVEMENTS.md` §1 (quick wins) as you go rather than as a followup
pass — most of them are cheaper to build in from the start (framework
validation, the config-drift-proofing that falls out of doing §7.4 correctly
the first time, a real CI/pre-commit setup) than to retrofit. Treat §2/§3
(modernization/architecture) as live design decisions to make *during* the
initial build, not deferred cleanup — e.g. decide once, up front, whether
you're going async (§2.2) rather than building sync and converting later.
