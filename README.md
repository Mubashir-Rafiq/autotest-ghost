# Ghost

Ghost watches your Python files. Every time you save one, it writes a `pytest`
test for it using an LLM, runs that test in a real subprocess, and — if the test
breaks — tries to fix it.

The interesting part is what it does when a test *fails an assertion*. A naive
tool rewrites the test until it passes, which happily rubber-stamps a genuine
bug in your code. Ghost asks a second question first — **is the test wrong, or
is the source wrong?** — and when the answer is "the source", it stops and tells
you instead of touching the test.

> **Status: under construction.** This is a from-scratch rebuild, built in 13
> documented stages. See [`ROADMAP.md`](ROADMAP.md) for what works today and
> what is still to come, and [`docs/stages/`](docs/stages/) for a detailed
> explanation of how each piece works.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:Mubashir-Rafiq/autotest-ghost.git
cd autotest-ghost
uv sync
```

## Try it

```bash
uv run ghost version    # Ghost, Python, and platform versions
uv run ghost doctor     # check the environment is healthy
uv run ghost --help     # everything available so far
```

## Development

```bash
make check    # every gate: format, lint, types, architecture, tests
make test     # just the tests
make fmt      # format in place
```

The gates are not decorative — `mypy` runs in strict mode, `ruff` runs a broad
rule selection, and `import-linter` enforces the module layering described in
[`SPEC.md`](SPEC.md). A red gate means fix the code, not loosen the gate.

## Documentation

| Document | What it is |
|---|---|
| [`ROADMAP.md`](ROADMAP.md) | Build stages, their status, and what each contains |
| [`docs/stages/`](docs/stages/) | A detailed explanation of every stage, written to be learned from |
| [`SPEC.md`](SPEC.md) | The authoritative technical specification |
| [`IMPROVEMENTS.md`](IMPROVEMENTS.md) | Everything being done better than the original |
| [`EXPLAINED.md`](EXPLAINED.md) | Background: what the tool does and why it is built this way |

## License

MIT
