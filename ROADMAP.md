# Ghost Rebuild — Roadmap

The living plan. Every stage updates this file **in the same commit that
completes it**, so the repository always states what is built and what is not.

Each stage ships: working code, its tests, all gates green, and an explanation
document in [`docs/stages/`](docs/stages/) written to be learned from.

**Progress: 8 of 13 stages complete.**

| # | Stage | Status | Explanation |
|---|---|---|---|
| 0 | Scaffolding, tooling, CI | ✅ **done** | [00-scaffolding.md](docs/stages/00-scaffolding.md) |
| 1 | Configuration | ✅ **done** | — |
| 2 | Providers + rate limiting | ✅ **done** | — |
| 3 | AST project indexing | ✅ **done** | — |
| 4 | Prompts + LLM client | ✅ **done** | — |
| 5 | Test runner + classification | ✅ **done** | — |
| 6 | The pipeline | ✅ **done** | — |
| 7 | Change tracking | ✅ **done** | — |
| 8 | **Debounce + job queue** | ⬜ next | — |
| 9 | File watcher | ⬜ pending | — |
| 10 | CLI completion | ⬜ pending | — |
| 11 | Daemon | ⬜ pending | — |
| 12 | Console + end-to-end | ⬜ pending | — |

---

## Locked decisions

These were settled before Stage 0 and are not revisited without a written reason.

| Decision | Choice | Why |
|---|---|---|
| Concurrency | **asyncio** | Everything Ghost does is I/O-bound; lets independent files process concurrently |
| Provider | **Groq only** for stages 0–12 | The one testable here today; the interface is built so others slot in |
| Default model | `openai/gpt-oss-120b` | Verified live; best code generation of Groq's current catalogue |
| Python | develop on **3.13**, support **≥3.11** | 3.11 is the floor for `tomllib`/`TaskGroup`; 3.13 is the newest with full dependency support |
| Layout | `src/` | Makes it impossible to import the working tree instead of the installed package |
| Test results | **first-party pytest plugin** | The only approach that can identify an `AssertionError` — see Stage 5 |

---

## What exists now

```
src/ghost/
├── __init__.py        # __version__, read from package metadata
├── errors.py          # exception taxonomy rooted at GhostError
├── config.py          # GhostConfig, layered loading, validation
├── rate_limiter.py    # GCRA rate limiting and backoff calculation
├── providers.py       # BaseProvider template method, GroqProvider, registry
├── indexer.py         # AST static analysis, shared ignore logic, budgeting
├── prompts.py         # pure prompt templates and deterministic construction
├── client.py          # LLM client, response validation, judge evaluation
├── pytest_plugin.py   # first-party pytest plugin extracting true exception classes
├── runner.py          # subprocess test runner with timeout and classification
├── pipeline.py        # unified generate -> run -> classify -> heal -> judge state machine
├── change_tracker.py  # SHA-256 content cache, atomic persistence, path keys
└── cli.py             # command group: version, doctor, config, providers, models, index, prompt, run-tests, generate
tests/
├── conftest.py            # shared fixtures
├── test_architecture.py   # structural invariants the linters cannot express
├── test_cli.py            # command behaviour via CliRunner
├── test_config.py         # configuration loading, precedence, and validation
├── test_rate_limiter.py   # GCRA rate limiter scheduling and backoff
├── test_providers.py      # BaseProvider retry/rate-limiting template, Groq
├── test_indexer.py        # AST extraction, type hints, tree, budgeting
├── test_prompts.py        # pure prompt generation and golden-file contracts
├── test_client.py         # LLM client, AST validation, overwrite protection
├── test_runner.py         # subprocess runner, timeouts, and error classification
├── test_pipeline.py       # unified pipeline, healing loop, judge safety valve
└── test_change_tracker.py # SHA-256 caching, atomic replace, fail-open invariants
```

Working commands: `ghost version`, `ghost doctor`, `ghost config --show`, `ghost providers`, `ghost models`, `ghost index --show`, `ghost prompt FILE`, `ghost run-tests TEST_FILE`, `ghost generate FILE [--if-changed]`, `ghost --help`.

---

## Stage detail

Every stage adds one thin CLI command, so there is always something to run and
see rather than nine invisible stages followed by a big reveal.

### ✅ Stage 0 — Scaffolding, tooling, CI
**Adds:** `ghost version`, `ghost doctor`
Packaging (`pyproject.toml`, hatchling, `src/` layout), the five quality gates
(ruff lint, ruff format, mypy strict, import-linter, pytest), pre-commit, GitHub
Actions CI on a 3.11/3.12/3.13 matrix plus a non-blocking 3.14 forward-compat
job, `errors.py`, and the CLI skeleton with its error boundary.

### ✅ Stage 1 — Configuration
**Adds:** `ghost config --show`
`GhostConfig` as a frozen `pydantic-settings` model with `extra="forbid"`.
Layered precedence: defaults → `ghost.toml` → `.env` → environment. **One**
template-writing function (the original had two, and they drifted). A malformed
config produces an error naming the offending key, not a silent default.

### ✅ Stage 2 — Providers + rate limiting
**Adds:** `ghost providers`, `ghost models`
`BaseProvider` using the template-method pattern so retry and rate limiting
cannot be forgotten by a subclass (in the original the retry decorator sat on an
abstract method and was a silent no-op). Model IDs fetched **live** — every ID
hardcoded in the original is dead today. Virtual-scheduling rate limiter that is
correct under concurrency.

### ✅ Stage 3 — AST project indexing
**Adds:** `ghost index --show`
Static analysis with `ast`: function signatures *with type hints*, classes,
methods, docstrings. Keyed by path, not basename (the original collided on two
files sharing a name). Context budgeting so a large project cannot overflow the
model's context window.

### ✅ Stage 4 — Prompts + LLM client
**Adds:** `ghost prompt FILE` (prints the prompt, makes no API call)
Prompts as pure functions with golden-file tests. Generated code is validated
with `ast.parse` before being written to disk. A third judge outcome,
`UNCLEAR`, which never heals — the original had no branch for it.

### ✅ Stage 5 — Test runner + classification
**Adds:** `ghost run-tests TEST_FILE`
`asyncio.create_subprocess_exec` with a real timeout, process-group kill, and
zombie reaping (the original had no timeout at all — a generated infinite loop
wedged it permanently). A first-party pytest plugin yields the true exception
class, so classification is a lookup table rather than substring matching.

### ✅ Stage 6 — The pipeline
**Adds:** `ghost generate FILE`
The generate → run → classify → heal → judge state machine, implemented **once**.
The original implemented it twice and the copies drifted, which is this
project's defining bug. Refuses to overwrite hand-written test files.

### ✅ Stage 7 — Change tracking
**Adds:** `--if-changed`
SHA-256 content cache that skips the pipeline for unchanged saves. Recorded only
*after* the pipeline completes, and never on cancellation, so a crash cannot be
mistaken for "already done".

### ⬜ Stage 8 — Debounce + job queue
Per-path debouncing with asyncio deadline-bumping, and a worker pool whose
per-path guard actually holds under concurrency (the original's does not, and
`IMPROVEMENTS.md` §3.2 incorrectly claims it does).

### ⬜ Stage 9 — File watcher
**Adds:** `ghost watch`
The single `watchdog`-thread → event-loop boundary. Handles atomic-rename saves,
which the original misses entirely — meaning it never sees saves from vim,
JetBrains IDEs, or a formatter.

### ⬜ Stage 10 — CLI completion
**Adds:** `ghost init`, and wires `start`/`stop`/`status`/`logs`
The init wizard (using live model listings), structured exit codes, full
`CliRunner` coverage.

### ⬜ Stage 11 — Daemon
Background process management: `flock`-based PID file (immune to PID reuse),
asyncio signal handling, non-blocking log rotation, and a documented nine-step
shutdown sequence.

### ⬜ Stage 12 — Console + end-to-end
`rich` presentation in roughly 150 lines rather than the original's 702, and a
full end-to-end test in which everything is real except the LLM.

---

## Optional stages (after 12, your choice)

| # | Stage | What it adds |
|---|---|---|
| 13 | More providers | OpenAI, Anthropic, Ollama, LM Studio, OpenRouter |
| 14 | Extra features | Coverage reporting, cost/token tracking, heal history, batch generate |
| 15 | Publish | PyPI packaging and release workflow |

---

## Defects being fixed

The rebuild fixes 20 known defects. Nine are documented in `SPEC.md` /
`IMPROVEMENTS.md`; the other eleven were found by reading the original source
during planning and are recorded here because nothing else records them.

| Defect | Fixed in |
|---|---|
| Pipeline implemented twice, copies drifted (SPEC §7.4) | 6 |
| `ghost watch` ignores `auto_heal` / `use_judge` (IMPROVEMENTS §1.3) | 6 |
| `auto_heal=False` skips the first test run entirely (SPEC §7.4) | 6 |
| A *passing* test misclassified as `UNKNOWN`, then "healed" (SPEC §7, §9) | 5, 6 |
| Substring-matching error classification (IMPROVEMENTS §2.3) | 5 |
| Two `ghost.toml` templates that drifted (SPEC §4.1) | 1, 10 |
| Dead model IDs duplicated across five locations (SPEC §11.3) | 2 |
| On-disk state keyed by basename, so same-named files collide (SPEC §5) | 3, 6, 7 |
| Judge returning `UNCLEAR` matches no branch (SPEC §7.3) | 4, 6 |
| **No subprocess timeout — a generated infinite loop wedges Ghost forever** | 5 |
| **Queue dedup breaks with more than one worker** | 8 |
| **`get_project_tree` sits in the runner with its own divergent ignore list** | 3 |
| **The retry decorator is applied to an abstract method: a silent no-op** | 2 |
| **`on_moved` unhandled — misses saves from every atomic-rename editor** | 9 |
| **`"test" in name` silently ignores `latest.py`, `manifest.py`, `contest.py`** | 9 |
| **Overwrites hand-written test files without asking** | 4, 6 |
| **`sys.executable` lacks the target project's dependencies** | 5 |
| **The entire project index goes into every prompt — context overflow** | 3 |
| **Unparseable LLM output written to disk before anyone checks it** | 4 |
| **`stop()` neither drains nor cancels in-flight work** | 8 |

Bold entries are undocumented in the source material; each gets a named
regression test in the stage that fixes it.
