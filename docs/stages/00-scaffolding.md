# Stage 0 — Scaffolding, Tooling, and CI

> **What you can run after this stage:** `ghost version`, `ghost doctor`, `ghost --help`
> **Files added:** `pyproject.toml`, `src/ghost/{__init__,errors,cli}.py`, `tests/`, `.importlinter`, `Makefile`, CI, pre-commit
> **Gates:** ruff · ruff-format · mypy (strict) · import-linter · pytest — all green

---

## 1. The problem this stage solves

We are about to write roughly 3,000 lines of asynchronous Python that talks to a
network API, spawns subprocesses, watches the filesystem, and runs as a
background daemon. Every one of those is a category of code that is *hard to
debug after the fact*.

So the question Stage 0 answers is: **what do we want in place before the first
line of real logic exists?**

The tempting answer is "nothing, let's build the fun part." The reason that's
wrong is worth stating precisely rather than as a slogan.

Suppose you write ten modules and *then* turn on a strict type checker. You now
have, say, 180 type errors. You cannot tell which are real bugs and which are
noise, you cannot fix them incrementally because they are all in your way at
once, and the overwhelmingly common outcome is that you loosen the checker until
it goes quiet. That is exactly what happened to the codebase we're rebuilding:
its `pyproject.toml` configures `mypy` with `strict = false` and configures
`ruff` with *no lint rules selected at all*. The tools were installed, listed as
dependencies, and mentioned in the README — and they checked essentially
nothing.

Turn the same checker on when there are zero modules and it never accumulates a
backlog. Each new file is either clean or it doesn't get committed. The cost per
file is small and constant; the cost of retrofitting grows with the size of the
codebase. **This is the entire argument for doing tooling first**, and it
generalises well beyond this project.

---

## 2. Concepts

### 2.1 The `src/` layout, and a genuine import bug it prevents

There are two ways to arrange a Python package:

```
flat layout                     src layout  (what we use)
───────────                     ──────────
myproject/                      myproject/
├── ghost/                      ├── src/
│   └── __init__.py             │   └── ghost/
├── tests/                      │       └── __init__.py
└── pyproject.toml              ├── tests/
                                └── pyproject.toml
```

The difference matters because of how Python resolves imports. `sys.path` — the
list of directories Python searches — normally begins with the **current working
directory**. So with the flat layout, running anything from the project root
means `import ghost` finds `./ghost/`, your working copy, *not* the version
installed into the virtual environment.

Usually those are the same files and nobody notices. The times it bites:

- Your package needs a build step or generated files; the installed copy has
  them, the source tree doesn't.
- You've declared a dependency in `pyproject.toml` but forgotten to import it
  correctly; tests pass locally against the source tree and fail for anyone who
  installs the package.
- **The one that matters most for Ghost:** we will spawn subprocesses with a
  deliberately manipulated `PYTHONPATH` in order to run generated tests against
  a *different* project. When Ghost's own directory and the target project's
  directory both sit on `sys.path`, "which `ghost` module is this?" stops being
  a rhetorical question.

With `src/`, the project root contains no importable package at all. `import
ghost` can *only* resolve to the installed one. The class of bug is removed
rather than avoided by discipline.

### 2.2 The build backend and editable installs

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Python separates *what your project is* (metadata in `pyproject.toml`) from *the
program that turns it into an installable artifact* (the **build backend**).
`setuptools` is the historical default; `hatchling` is a modern one, and it's
what we use because it needs no configuration beyond pointing at `src/`.

```toml
[project.scripts]
ghost = "ghost.cli:main"
```

This one line is what makes `ghost` a command you can type. At install time the
backend generates a small executable on your `PATH` whose body is essentially
`from ghost.cli import main; sys.exit(main())`. Note the shape it demands:
`main` must **return** an exit code rather than call `sys.exit` itself. We'll
come back to why that's also better for testing.

`uv sync` installs the project in **editable** mode — the installed package is a
link back to `src/ghost/`, so your edits take effect immediately with no
reinstall. You get the correctness of installed-package semantics *and* a fast
edit loop.

### 2.3 Lockfiles: `pyproject.toml` vs `uv.lock`

Two files, two different jobs, and the distinction confuses people:

| | `pyproject.toml` | `uv.lock` |
|---|---|---|
| Contains | Ranges: `click>=8.1.7` | Exact pins: `click==8.5.0` |
| Written by | You | `uv` |
| Answers | "What is this project compatible with?" | "What exactly is installed right now?" |
| Committed? | Yes | Yes, **for an application** |

The rule of thumb: an *application* (like Ghost) commits its lockfile, because
you want every developer and every CI run to get byte-identical dependencies —
"works on my machine" is nearly always a dependency-version difference. A
*library* typically does not, because it must remain compatible with whatever
its consumers already have.

### 2.4 Five gates, and why none of them overlaps

This is the part worth internalising: each tool checks something the others
*structurally cannot*, and we deliberately never check the same thing twice.

| Gate | Question it answers | Example it catches |
|---|---|---|
| **ruff** (lint) | Is any single line wrong or dangerous? | `print()` in library code; blocking `time.sleep` inside `async def` |
| **ruff format** | Is the code formatted canonically? | Ends all formatting debate; diffs show logic changes only |
| **mypy --strict** | Do the types actually line up? | Passing `str` where `Path` is expected; a forgotten `await` |
| **import-linter** | Does module A import module B? | `runner.py` importing `console.py`, violating the layering |
| **pytest** | Does it behave correctly? | Everything above is about *shape*; only tests check *behaviour* |

Two of these deserve elaboration.

**`ruff`'s `ASYNC` ruleset** is the most valuable single choice in this config,
because we picked an asyncio architecture. The characteristic async bug is a
*blocking* call inside a coroutine: `time.sleep(5)`, `requests.get(...)`, or a
plain `open()`. None of these is an error — the code runs and produces the right
answer. It just freezes the **entire event loop** for the duration, stalling
every other task in the program. There is no exception and no log line; the
program is merely mysteriously slow. `ASYNC` makes each one a lint error.

**mypy's `unused-awaitable`** catches the mirror-image mistake:

```python
async def fetch() -> str: ...

result = fetch()          # BUG: forgot `await`. `result` is a coroutine object.
```

At runtime this produces a `RuntimeWarning: coroutine 'fetch' was never awaited`
— and warnings are, by default, invisible. The function simply never runs.
Enabling this error code turns a silent no-op into a build failure. We *also*
set `filterwarnings = ["error"]` in the pytest config, which promotes that
runtime warning into a test failure, so the mistake is caught twice by two
independent mechanisms.

### 2.5 Architecture as an executable contract

`SPEC.md` says Ghost's dependencies point one way and never cycle.
`.claude/skills/ghost-architect/SKILL.md` repeats it. Both are prose, and prose
is not enforcement — the original project had the same rule written down and
violated it anyway.

`.importlinter` turns the rule into a build failure:

```ini
[importlinter:contract:layers]
name = Ghost layered architecture
type = layers
layers =
    ghost.cli
    ghost.errors
```

A `layers` contract reads top-down: a module may import anything **below** it
and nothing **above** it. Right now the claim is just "the CLI may use the error
types; the error types know nothing about the CLI." It grows by one module per
stage, and by Stage 12 it will encode the project's whole dependency structure.

Placing your new module in this file is part of finishing a stage. If you can't
decide which layer it belongs to, that is a real signal: the module probably has
more than one reason to change, and wants splitting.

### 2.6 Designing an exception taxonomy

`errors.py` is 60 lines and contains no logic, but the decisions in it shape
error handling everywhere else.

**Why a single base class.** Every Ghost error inherits `GhostError`. That gives
the CLI one boundary to catch, and — more importantly — it creates a sharp
distinction between two things that are usually muddled:

- A **`GhostError`** is a condition we *anticipated* and can explain. "No
  `ghost.toml` here; run `ghost init`." The user should see a clear sentence.
- **Anything else** is a *bug in Ghost*. The user should see the full traceback,
  because hiding it makes the bug harder to find and report.

A program that catches `Exception` broadly and prints "Something went wrong"
has erased that distinction and thrown away the information needed to fix it.

**Why errors name their input.** Compare:

```python
raise ConfigError("invalid configuration")                        # useless
raise ConfigError(f"unknown key 'auto_heel' in {path} [tests]")   # actionable
```

The second tells you the file, the section, and the exact typo. This is why
`ProjectNotInitializedError` takes the path it searched from as a constructor
argument — a caller *cannot* raise it without supplying that context:

```python
class ProjectNotInitializedError(GhostError):
    def __init__(self, searched_from: Path) -> None:
        self.searched_from = searched_from
        message = (
            f"no ghost.toml found in {searched_from} or any parent directory. "
            f"Run 'ghost init' in your project root to create one."
        )
        super().__init__(message)
```

Note the message also states the **remedy**. A good error answers three
questions: what went wrong, what specific input caused it, and what to do next.

**Why two error types and not one.** `ConfigError` and
`ProjectNotInitializedError` are separate because they need different responses:
a missing config is fixed by running one command, while a malformed config needs
a human to edit a file. Merging them to save a class would mean giving a vaguer
answer to both.

---

## 3. The code

### 3.1 One source of truth for the version

```python
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("autotest-ghost")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
```

The obvious implementation is `__version__ = "0.1.0"` — but then the version
exists in two places (here and `pyproject.toml`) and they *will* disagree
eventually. Reading it from installed metadata makes `pyproject.toml` the single
source of truth, and `tests/test_architecture.py` asserts the two agree so the
failure mode is a red test rather than `ghost --version` quietly lying.

This is a tiny instance of a rule that runs through the whole project: **when
the same fact appears in two places, they drift.** The bug that defines the
original Ghost codebase is exactly this pattern at a larger scale.

### 3.2 The CLI error boundary

```python
def main(argv: Sequence[str] | None = None) -> int:
    try:
        cli.main(args=argv, standalone_mode=False)
    except click.exceptions.Exit as exc:
        return exc.exit_code
    except click.exceptions.Abort:
        click.echo("Aborted.", err=True)
        return 130
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except GhostError as exc:
        click.echo(f"Error: {exc}", err=True)
        return 1
    except SystemExit as exc:
        return _system_exit_code(exc)
    return 0
```

Two deliberate choices here.

**`standalone_mode=False`.** Normally `click` handles errors and calls
`sys.exit()` itself. Turning that off hands control back to us, which is what
lets this function *be* the error boundary. It also makes `main` a normal
function returning an `int`, so a test can call `main(["version"]) == 0` without
catching `SystemExit`.

**There is no `except Exception`.** That omission is the design. An unanticipated
exception propagates with its traceback intact, because it represents a bug we
want to see rather than a condition we want to explain.

Exit code `130` for `Abort` is the Unix convention for "terminated by SIGINT"
(128 + signal 2) — the code a shell reports when you press Ctrl+C.

### 3.3 A `doctor` command that grows

`ghost doctor` currently reports the interpreter and each dependency's installed
version. That's genuinely useful — "which Python is this actually running?" is
the first question in half of all environment problems, and the answer is often
surprising.

It is deliberately *small*. It does not look for `ghost.toml`, because config
discovery doesn't exist until Stage 1 and writing a second, simpler version of
it here would create exactly the duplication this project exists to avoid. The
command grows one section per stage, as the concepts it reports on come into
existence.

### 3.4 Tests that check what linters cannot

`tests/test_architecture.py` deliberately does **not** re-check anything ruff or
import-linter already covers. It tests only claims those tools cannot express:

```python
def test_asyncio_run_is_called_in_exactly_one_place(source_root: Path) -> None:
    ...
```

A linter can ban a function call, but it cannot say "this may appear **once**,
in **this file**." Ghost is an async program with exactly one synchronous
boundary; a second `asyncio.run` means either a nested-event-loop crash or a
second entry point that will drift from the first. Counting occurrences across
files is a genuine gap in what static analysis offers, so it's a test.

The file's docstring records the division of labour explicitly, because the
temptation to add "just one more safety check" here is strong and every such
check is duplication.

---

## 4. Decisions and rejected alternatives

**`ruff` instead of `black` + `isort` + `flake8`.** The original used three
tools with three configs. `ruff` does all three jobs, is roughly an order of
magnitude faster, and — the real benefit — has one configuration section, so
formatter and linter cannot disagree and undo each other's work.

**`click` rather than `typer`.** `IMPROVEMENTS.md` §2.5 suggests considering
`typer`, which generates a CLI from type hints with less boilerplate. We stayed
with `click` for a specific reason: `typer` has no native support for
asynchronous commands, so with an asyncio architecture you end up writing the
`asyncio.run` wrapper by hand anyway — while gaining a layer between you and
click's `CliRunner` and context handling, both of which the Stage 10 wizard
needs. The boilerplate saving turns out to be illusory here.

**Develop on Python 3.13, support 3.11+.** The system Python here is 3.14.4, but
`watchdog` 6.0 advertises support only through 3.13 and `httpx` through 3.12.
Developing on 3.13 removes "is this a 3.14 change or my bug?" from the debugging
surface while building async code from scratch. CI still runs 3.11/3.12/3.13 as
required jobs, plus a **non-blocking** 3.14 job for early warning. We declare
`requires-python = ">=3.11"` because 3.11 is the genuine floor: `tomllib`,
`asyncio.TaskGroup`, and `asyncio.timeout()` all arrive there.

**Full dependency set declared now, though most is unused until later stages.**
The alternative — adding each dependency in the stage that needs it — is
arguably more instructive, but it means the lockfile re-resolves six times and a
Stage 8 change could silently upgrade a Stage 2 dependency. A stable lockfile
across the whole build is worth more. Each entry in `pyproject.toml` is
commented with the stage that first uses it.

---

## 5. A real bug this stage caught — in the tooling itself

Worth recording, because it happened while writing this stage and is a better
lesson than a hypothetical.

Running `ruff format .` for the first time reported `1 file reformatted` — and
`git status` showed the modified file was **`SPEC.md`**. Recent versions of ruff
format Python code blocks inside Markdown, and it had rewritten a code sample in
the specification:

```diff
-  sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
+
+  sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
```

That looks harmless. It is not. `SPEC.md` §7.1 declares that exact snippet a
**non-negotiable literal** that every generated test file must contain. The
formatter had silently edited a specification of exact output — changing quote
style and inserting a blank line in the very text we will later assert against.

The fix, in `pyproject.toml`:

```toml
extend-exclude = ["*.md"]
```

Two takeaways worth more than the incident:

1. **Any tool with write access will eventually write something you didn't
   intend.** The reason this was caught in seconds rather than discovered in
   Stage 4 as a baffling prompt mismatch is that the repository was already
   under version control with a clean working tree, so `git status` made the
   unintended change obvious. Committing the scaffolding *before* running the
   tools is what made the tools safe to run.
2. **Documents that specify exact strings are code**, and need protecting from
   automation like code does.

---

## 6. Try it yourself

```bash
uv sync              # create the venv and install everything
make check           # run all five gates the way CI does
```

Then explore:

```bash
uv run ghost version         # note the interpreter path — it's the project venv
uv run ghost doctor
uv run ghost --help
uv run ghost badcommand; echo "exit code: $?"     # non-zero, with a usage error
```

### Exercises

1. **Break the layering.** Add `import ghost.cli` to the top of
   `src/ghost/errors.py`, then run `uv run lint-imports`. Read the failure
   message — it names the exact forbidden import chain. Undo it.

2. **Break the types.** In `cli.py`, change `_installed_version`'s return
   annotation from `str | None` to `str`, then run `uv run mypy`. It finds the
   `return None` branch. Notice it *doesn't* need the function to be called to
   know it's wrong.

3. **Trip the async rule early.** Add this to `cli.py`:
   ```python
   import time
   async def slow() -> None:
       time.sleep(1)
   ```
   Run `uv run ruff check .`. You'll get `ASYNC251`. This is the rule that will
   protect every async module from Stage 2 onward.

4. **See the version test do its job.** Change `version` in `pyproject.toml` to
   `0.2.0` and run `uv run pytest -q` *without* re-running `uv sync`. The
   architecture test fails, because the installed metadata still says `0.1.0`.
   Run `uv sync` and watch it pass. Revert afterwards.

5. **Read the gate config.** Open `pyproject.toml` and find the
   `banned-api` section. Each entry is a rule from the project plan turned into
   a mechanical check. Ask yourself, for each: what bug is this preventing?

---

## 7. What's next

**Stage 1 — Configuration.** `GhostConfig` as a frozen `pydantic-settings`
model: layered precedence (defaults → `ghost.toml` → `.env` → environment),
validation that produces errors naming the offending key, and exactly **one**
function that writes the config template. The original had two such functions,
they drifted apart, and `SPEC.md` §4.1 documents the resulting inconsistency —
which is the first defect on our list that we get to design out of existence
rather than fix.
