---
name: ghost-architect
description: Use whenever writing, adding, refactoring, wiring, or reviewing code anywhere in this repo (the "Ghost"/autotest-ghost self-healing pytest-generator rebuild) — creating or touching a ghost/ module, the CLI, config, providers, the generate/run/heal/judge pipeline, the watcher, the daemon, or their tests. Also use when deciding where new code belongs, how to structure a new feature, or how to keep the codebase modular and debuggable as it grows. Enforces the module boundaries, one-shared-pipeline rule, typed contracts, and testing discipline this rebuild is committed to, so mistakes documented in SPEC.md/IMPROVEMENTS.md are not reintroduced. Not for questions about how the *original* Ghost implementation behaved — read EXPLAINED.md directly for that; this skill is about how the rebuild should be written.
---

# Role: Architect of the Ghost rebuild

This repo is a from-scratch rebuild of Ghost, a CLI that watches Python
source, generates pytest tests via an LLM, runs them, and self-heals
failures behind a "Judge" safety valve. Three documents already define this
project — **read the relevant section before writing code that touches it**,
never guess at behavior that's already specified:

- **`SPEC.md`** — the authoritative contract: module layout (§3), config
  schema (§4), on-disk state (§5), CLI surface (§6), the core pipeline state
  machine (§7), AST indexing (§8), error classification (§9), rate
  limiting (§10), provider interface (§11), file watching (§12), daemon
  design (§13), testing conventions (§14), and the suggested build order
  (§15).
- **`IMPROVEMENTS.md`** — the deciding vote wherever it disagrees with
  SPEC.md's description of the old baseline (e.g. pydantic-settings instead
  of hand-rolled `from_dict`, structured error classification instead of
  substring matching, one shared pipeline instead of two). Build the
  improved version directly — don't build the baseline and refactor after.
- **`EXPLAINED.md`** — the "why," useful for understanding intent, not a
  build spec.

If a task isn't covered by these three files, you're setting precedent —
write the decision down (module docstring, ADR-style comment) with its
reason, the same way SPEC.md documents its own rationale.

---

## The prime directive of this rebuild

The old Ghost codebase's worst bug (SPEC.md §7.4) happened because
`main.py` and `daemon.py` each hand-implemented the same generate → run →
classify → heal → judge state machine, and they drifted: one ignored
`auto_heal`/`use_judge` entirely, the other miscounted the first test run.
**Every mistake in this rebuild's history rhymes with that one shape:** the
same logic written twice, in two places, that quietly stop agreeing.

Before adding *any* code, ask: **does something like this already exist
elsewhere in the codebase?** If yes, extend or call it — never duplicate it.
This is the single highest-leverage rule in this skill.

---

## Module boundaries (SPEC.md §3) — put new code in the right place

```
ghost/
├── __init__.py        # Public API surface + __version__ — only what's re-exported
├── cli.py             # click CLI only — no business logic, delegates everything
├── pipeline.py         # THE ONE generate→run→classify→heal→judge state machine
├── daemon.py            # Background process management — calls pipeline.py, doesn't reimplement it
├── change_tracker.py    # Content-hash cache only
├── job_queue.py          # Debounce + worker queue only
├── init.py               # AST scanner — pure static analysis, no LLM, no I/O beyond the fs
├── config.py             # pydantic-settings model — the ONLY place that reads ghost.toml/.env/env
├── chat.py               # Prompt construction + LLM call + response cleanup + Judge
├── providers.py          # BaseProvider + concrete providers behind one interface
├── runner.py             # Runs the generated test, returns structured results
├── rate_limiter.py       # Rate limiting + retry — no knowledge of what it's limiting
└── console.py            # Presentation only — never imported by runner/config/providers/change_tracker
tests/                    # One test file per module above, fixtures in conftest.py
```

**Dependency direction is one-way and acyclic:** `cli.py` → `pipeline.py` →
(`chat.py`, `runner.py`, `change_tracker.py`, `job_queue.py`) →
(`config.py`, `providers.py`). `console.py` is a leaf everything may call
into for output, but it may call into nothing. `init.py` is self-contained.
If you find yourself importing "up" this chain, the code is in the wrong
module — move it, don't add an exception.

Before creating a new top-level module, check whether the concern actually
belongs inside an existing one from this table. A new module is justified
only when it has a genuinely different reason to change than every existing
one — say what that reason is in its docstring.

---

## Standing rules for this rebuild

Each rule below exists because the old codebase violated it and paid for it
(cited section is where SPEC.md/IMPROVEMENTS.md documents the incident).
Treat a violation here as a defect, not a style preference.

1. **One pipeline, one implementation.** The generate/run/classify/heal/judge
   state machine lives in exactly one function/class (`pipeline.py`,
   IMPROVEMENTS.md §3.1). The foreground watcher and the daemon call it;
   neither reimplements it. The *only* difference between them is how
   results are reported (rich console vs. log file) — never control flow.
2. **One config-writing function.** `ghost init` must have exactly one code
   path that writes a default `ghost.toml` (SPEC.md §4.1 documents two that
   drifted in the original). If you need the template in two places, extract
   a function — don't write it twice.
3. **Config is validated, not hand-mapped.** Use `pydantic-settings` for
   `GhostConfig` (IMPROVEMENTS.md §2.1) — real type coercion and error
   messages on a malformed `ghost.toml`, not a silent `.get()` fallback or a
   raw `KeyError`.
4. **Failure classification is structural, not stringly-typed.**
   Classify test outcomes from `pytest --json-report` / `TestReport` objects
   (IMPROVEMENTS.md §2.3), never by grepping stdout/stderr for substrings
   like `"AssertionError"` — that breaks the moment a test's own output
   contains the word.
5. **Always check the success case before classifying failure.** A passing
   run must short-circuit before any classifier runs (SPEC.md §7 step 4) —
   this is what the original `main.py` got wrong, treating a pass as
   `UNKNOWN` and trying to "heal" it.
6. **Config flags gate behavior identically everywhere they apply.**
   `auto_heal`, `use_judge`, `max_heal_attempts` must produce the same
   behavior regardless of call site. If two call sites can read the same
   config field and behave differently, that's the bug class from SPEC.md
   §7.4 — fix it by removing the second call site, not by syncing both.
7. **Providers are one interface, dispatched through a registry, never
   `if`/`elif` on provider name** (SPEC.md §11.1–11.2). Adding the Nth
   provider should mean adding one class and one registry entry — not
   editing a chain of conditionals.
8. **No provider SDK leaks past `providers.py`.** `chat.py`/`pipeline.py`
   only ever talk to `BaseProvider`.
9. **Presentation never leaks into logic.** `console.py`/rich output stays
   out of `runner.py`, `change_tracker.py`, `providers.py`, `config.py`,
   `pipeline.py` (SPEC.md §15 step 7). Business logic returns data; the
   caller decides how to display it.
10. **Decide sync vs. async once, up front** (IMPROVEMENTS.md §2.2) — don't
    build synchronous and bolt on async later. If the project has committed
    to one, follow it consistently across `runner.py`, providers, and
    `job_queue.py`.
11. **Model IDs and other fast-drifting facts live in one place**, ideally
    fetched live from the provider where an endpoint exists, not copy-pasted
    across five locations (SPEC.md §11.3 documents five places that had to
    agree and didn't).
12. **On-disk state files fail open, not closed**, and are written only
    after the operation they track fully completes (SPEC.md §5, §12.3) — a
    crash mid-pipeline must never be mistaken for "already done."

---

## Definition of done, per module/feature

Don't call a module finished until:

- [ ] It lives in the module named for it in the boundaries table above, and
      doesn't import "backward" up the dependency chain.
- [ ] It has a module-level docstring stating its contract: what it owns,
      what it explicitly does not do, what it guarantees.
- [ ] Nothing in it duplicates logic that already exists elsewhere — grep
      first.
- [ ] Public functions/classes are fully type-hinted; `mypy` passes on it
      (SPEC.md §14, IMPROVEMENTS.md §5).
- [ ] It has a matching `tests/test_<module>.py` using real behavior against
      `tmp_path` where practical, mocking only genuinely external/expensive
      calls (LLM providers, network) — SPEC.md §14's convention.
- [ ] Errors raised or returned name the offending input, not a generic
      "invalid input" message.
- [ ] `ruff check`, `black --check`, `isort --check`, `mypy`, and `pytest`
      all pass locally before considering the change complete — these are
      wired into CI and pre-commit (SPEC.md §14, IMPROVEMENTS.md §1.4/§1.5)
      and a red run means fix the code, not skip the gate.
- [ ] If this change replaced an older approach, the old code is deleted in
      the same change — no dead second implementation left "just in case."

---

## Workflow for any nontrivial change

1. **Locate the spec.** Find the SPEC.md section (and IMPROVEMENTS.md
   counterpart) covering this behavior. If none exists, you're setting a
   precedent — write the contract down before implementing.
2. **Check for an existing home.** Use the module boundaries table; resist
   creating a new module unless the concern truly doesn't fit an existing
   one.
3. **Search for duplication risk.** Grep for the state machine, template, or
   logic you're about to write — if something adjacent already exists,
   extend it instead of writing a parallel version (see the prime directive
   above).
4. **Write the contract, then the code, then the test** — in that order, not
   test-after-the-fact and not contract-as-comment-after-implementation.
5. **Run the full gate** (`pytest`, `ruff`, `black`, `isort`, `mypy`) before
   reporting the change complete.
6. **Update SPEC.md/IMPROVEMENTS.md** if the change intentionally departs
   from what they describe — they're meant to stay accurate as the rebuild
   progresses, not frozen as day-one documents.

---

## Smells specific to this codebase's known failure history

| Smell | What it actually means here | Remedy |
|---|---|---|
| Same generate/run/heal/judge logic appearing in two files | The exact SPEC.md §7.4 bug reappearing | Extract to `pipeline.py`; both callers call it |
| A second function that writes `ghost.toml` | The SPEC.md §4.1 drift reappearing | One template-writing function, used by every caller |
| `if "AssertionError" in output` / similar string checks | The SPEC.md §9 fragile classifier | Use structured pytest result data |
| A provider-name `if`/`elif` chain outside `providers.py`'s registry | Provider abstraction leaking | Route through `get_provider()`, extend the registry |
| A config flag read in one call site but not another | The auto_heal/use_judge drift bug class | Single call site, or a shared function both use |
| `console.py`/rich imports inside `runner.py`, `config.py`, etc. | Presentation leaking into logic | Return data; let the caller print it |
| A hardcoded model ID outside the registry | SPEC.md §11.3's five-places-to-update problem | One source of truth, fetched live if the provider supports it |
