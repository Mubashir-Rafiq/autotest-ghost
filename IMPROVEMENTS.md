# Improvement Roadmap

A grounded list of things you could fix, modernize, or add if you rebuild this
project yourself — ordered roughly by effort. Every item below references the
actual file/behavior in the current codebase, not a hypothetical. Use it as a
punch list for your own fork, not something to copy verbatim.

---

## 1. Quick wins (an evening each)

These are small, contained, and teach a specific lesson.

### 1.1 Actually use `rich` (it's already a dependency you're not using)
`pyproject.toml` declares `rich>=13.0.0`, but `ghost/console.py` hand-rolls
~670 lines of raw ANSI escape codes, a manual spinner thread, and its own
color enum instead. Swap it for `rich.console.Console`, `rich.spinner`,
`rich.progress`, and `rich.syntax.Syntax` (for pretty-printing the generated
test code before writing it). You'll delete most of `console.py` and gain
proper Windows-terminal support and automatic "no-color when piped" handling
for free — both of which the current hand-rolled version doesn't have.

### 1.2 Honor `config.tests.framework` in the test runner
`ghost/runner.py:run_test` always shells out to
`python -m pytest <test_file>`, no matter what `ghost.toml` says under
`[tests] framework = "..."`. Meanwhile `ghost/chat.py`'s prompts *do*
parameterize on `framework` and will happily ask the LLM for `unittest`-style
tests — which then get executed with `pytest` anyway (which mostly works by
accident since pytest can run unittest-style tests, but breaks down for
anything else). Fix: make `run_test` dispatch on `config.tests.framework`
(pytest / unittest / others), or drop the illusion of choice from the config
schema until it's real.

### 1.3 Fix the drift between `main.py` and `daemon.py`
`daemon.py`'s `_process_file` respects `config.tests.auto_heal` and
`config.tests.use_judge` (it skips healing/judging when they're `False`).
`main.py`'s `check_test` does **not** check either flag — it always tries to
heal and always calls the Judge. This means `ghost watch` (foreground) and
`ghost start` (daemon) behave differently for the exact same config file,
which is a real, reproducible bug, not a style nit. See §3.1 for the deeper
fix (deduplicating the two implementations entirely).

### 1.4 Add CI
There is no `.github/workflows/` — only an issue template. Add a
`ci.yml` that runs on push/PR:
```yaml
- uv sync --dev
- uv run pytest -q
- uv run ruff check .
- uv run mypy ghost
```
`black`, `isort`, `mypy`, and `ruff` are already listed as `dev` extras in
`pyproject.toml` but nothing currently runs them — they're decorative until
CI (or a pre-commit hook, see below) actually invokes them.

### 1.5 Add `pre-commit`
Wire the same tools from §1.4 into a `.pre-commit-config.yaml` so they run
locally before every commit instead of only in CI. Five-minute setup, saves
CI round-trips.

### 1.6 Skip regeneration for unchanged files
Every file-modified event currently triggers a full LLM call, even if the
save was a no-op (e.g. an editor "format on save" that didn't change
anything, or a second save of identical content within the debounce window).
Store a content hash (`hashlib.sha256`) per file in `.ghost/context.json` (or
a new `.ghost/hashes.json`) and skip the whole pipeline when the hash hasn't
changed. Cheap to add, directly saves API calls and money.

### 1.7 Update the model registry
`ghost/providers.py`'s `POPULAR_MODELS` dict hardcodes specific model
snapshots (e.g. `claude-3-5-sonnet-20241022`, `claude-3-haiku-20240307`) that
are no longer the current flagship models. If you're using Anthropic models,
the current lineup is `claude-opus-5`, `claude-sonnet-5`, and
`claude-haiku-4-5` — update the registry (and check each provider's docs for
their current model IDs, since those change independently of this project).

---

## 2. Modernization (outdated patterns → current idiom)

### 2.1 Config: replace hand-written `from_dict` with `pydantic-settings`
`ghost/config.py` manually maps a parsed TOML dict onto nested dataclasses
field-by-field (`GhostConfig.from_dict`, ~35 lines of `.get(...)` calls) and
manually re-implements env-var override logic (`_apply_env_overrides`).
`pydantic` is already pulled in transitively (via the `openai`/`anthropic`
SDKs) — promote it to a direct dependency, add `pydantic-settings`, and
define `GhostConfig` as a `BaseSettings` subclass. You get: automatic env var
binding, type coercion + validation with real error messages (right now a
malformed `ghost.toml` value fails silently or with a raw `KeyError`), and
delete most of the manual mapping code.

### 2.2 Go async for I/O-bound work
Everything in the pipeline is I/O-bound (LLM API call, subprocess run,
file reads) yet it's all synchronous:
- `runner.run_test` uses blocking `subprocess.run`.
- `chat.TestGenerator._call_api` calls a synchronous provider `.chat()`.
- `JobQueue` uses a single OS thread with a blocking `Queue.get`.

Switching the provider classes to their async variants (`AsyncAnthropic`,
`AsyncOpenAI`, `httpx.AsyncClient` for Ollama/LM Studio) and the job queue to
`asyncio.Queue` + `asyncio.create_subprocess_exec` would let Ghost process
multiple changed files concurrently (bounded by the existing rate limiter)
instead of one at a time — directly enables §3.2 below.

### 2.3 Replace substring-matching error classification
`runner.classify_error` decides SYNTAX vs RUNTIME vs LOGIC by checking
whether strings like `"AssertionError"` appear anywhere in captured
stdout+stderr. This breaks the moment a test's *own* string data happens to
contain one of those words, and can't distinguish "the assertion in test A
failed" from "test B also printed the word AssertionError in a docstring."
Use the `pytest-json-report` plugin (`pytest --json-report`) or hook into
pytest's internal `TestReport` objects to get structured, per-test outcome
and exception-type data instead of grepping raw text.

### 2.4 Consolidate on one HTTP stack
`requests` is used for the local/custom providers while `httpx` is already
present transitively (via `openai`/`anthropic`). Standardizing on `httpx`
(sync *and* async client, same library) removes a dependency and is a
prerequisite for §2.2.

### 2.5 CLI: consider `typer` instead of raw `click`
`click` (used throughout `cli.py`) works fine, but `typer` — a thin layer on
top of click — generates the same CLI from type-hinted function signatures,
with less `@click.option(...)` boilerplate per command and auto-generated
`--help` from docstrings. Not a correctness issue, purely a "if starting
fresh today" idiom choice.

---

## 3. Architecture rework (bigger, good learning exercises)

### 3.1 Deduplicate the generate → run → heal → judge loop
This exact state machine is implemented twice:
- `main.py`: `make_tests` + `check_test` (used by `ghost watch`)
- `daemon.py`: `_build_watcher`'s inner `_process_file` (used by `ghost start`)

They've already drifted (§1.3). Extract a single `TestPipeline` class (e.g.
new file `ghost/pipeline.py`) with one method like
`run(file_path, source, config) -> PipelineResult`, and have both the
foreground watcher and the daemon call it — the only difference between them
should be *how output is reported* (pretty console output vs. log file), not
the actual control flow. This is the single highest-value refactor in the
codebase and a great exercise in separating "business logic" from
"presentation."

### 3.2 Real concurrency in the job queue
`JobQueue.__init__` already accepts `max_workers`, but both call sites
(`main.py`, `daemon.py`) hardcode `max_workers=1`. For a project where many
files change at once (e.g. checking out a branch, running a formatter across
the repo), tests could generate for independent files in parallel — the
existing `RateLimiter` already exists to bound *API* concurrency safely, so
raising `max_workers` is mostly safe once file-level locking (don't process
the same path twice concurrently — `JobQueue` already dedupes by path) is
double-checked under concurrency.

### 3.3 Make "framework" a real abstraction, not a prompt string
Introduce a small `TestRunner` interface (`PytestRunner`, `UnittestRunner`,
...) so `[tests] framework = "..."` in `ghost.toml` actually changes what
command gets executed (see §1.2), not just what the LLM is told to pretend
to write.

### 3.4 Richer project context instead of flat signature strings
`.ghost/context.json` currently stores one string per file like
`"Functions: parse(text); Classes: None"`. This is cheap and works, but loses
type hints, docstrings, and return types — all of which would make generated
tests meaningfully better without needing a heavyweight RAG/embeddings
pipeline. Since `init.py` already walks the AST, capturing `ast.arg.annotation`
and `ast.get_docstring()` is a small extension of code you already have open.

### 3.5 Provider registry via entry points
`get_provider()` already uses a clean dict dispatch (not if/elif — that part
is fine), but adding a new provider still means editing a central function
and the `ProviderType` enum. If you want this to be genuinely pluggable
(e.g. so a user could `pip install ghost-provider-mistral` without touching
core code), move to a `setuptools` entry-points registry where providers
self-register.

---

## 4. New features (good "extra credit," low-to-medium effort)

- **Batch generate** — `ghost generate <directory>` or `ghost generate --all`
  to generate tests for every source file at once, not just one file per
  invocation (today's `generate` command takes exactly one file).
- **Coverage reporting** — after a successful run, invoke
  `pytest --cov` and print the delta. `pytest-cov` isn't a dependency yet;
  it's a two-line addition.
- **Git hook installer** — `ghost install-hook` that drops a pre-commit or
  pre-push hook running the generated tests before allowing a commit.
- **Heal history / rollback** — snapshot each attempt's generated test into
  `.ghost/history/<file>/attempt_N.py` so you can diff what the healing step
  actually changed, or revert if an attempt made things worse. Straightforward
  since `WriteTest`/`_process_file` already know the full test content at
  each attempt.
- **Judge self-consistency** — call `consult_the_judge` 2–3 times and take a
  majority vote before declaring `BUG_IN_CODE`, since a single LLM call (even
  at `temperature=0.1`) isn't fully deterministic and a false "bug in your
  code" alert is disruptive. Cheap to add since the call is already isolated
  in one method.
- **Usage/cost tracking** — every provider SDK response includes token usage
  (`response.usage` for OpenAI/Anthropic, similar for Groq); Ghost currently
  discards it. Log it per run and print a running total in `ghost status`.
- **Notification webhook** — `requests` is already a dependency; add a
  `[notify] webhook_url` config option and POST a message when the Judge
  finds `BUG_IN_CODE` (currently this only logs/prints locally and is easy to
  miss if you're not watching the terminal).
- **Property-based tests via Hypothesis** — since prompt construction already
  branches on `framework`, add `"hypothesis"` as a recognized option and
  extend the prompt rules for property-based test generation as an
  alternative to plain example-based pytest tests.

---

## 5. Testing / DX gaps worth closing

- **No HTTP-level mocking in `tests/test_providers.py`** — the existing tests
  only check that `get_provider("groq", ...)` returns a `GroqProvider`
  instance; nothing exercises `.chat()` against a mocked response. Add
  `respx` (pairs with `httpx`, see §2.4) or `pytest-httpx` to test the actual
  request/response handling and error paths (rate limits, malformed
  responses) without hitting real APIs.
- **No mypy enforcement** — `[tool.mypy]` is configured in `pyproject.toml`
  with real settings, but nothing runs it (see §1.4). A lot of the codebase
  (especially `cli.py`) has loose typing that mypy would immediately flag.
- **No integration test for the full pipeline** — there's good unit coverage
  per module (79 tests), but nothing exercises "watch a temp directory, edit
  a file, assert a test file appears" end-to-end with a mocked LLM provider.
  Worth adding once §3.1's `TestPipeline` extraction makes it easy to inject
  a fake provider.

---

## 6. If you only have a weekend

Priority order that teaches the most per hour spent:

1. §1.4 CI + §1.5 pre-commit (so every later change is safety-netted)
2. §3.1 deduplicate the pipeline (fixes §1.3's real bug as a side effect)
3. §1.1 swap in `rich` (immediate visual payoff, low risk)
4. §2.1 `pydantic-settings` config (makes every future feature easier to add
   safely)
5. Pick one feature from §4 (batch generate or heal history are the easiest)
