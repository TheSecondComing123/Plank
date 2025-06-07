# Plank

Plank is a small interpreted language used for experimenting with parsing and interpretation in Python. The interpreter is implemented entirely in the `plank` package.

## Running a Program

Execute a `.plank` file using Python's module runner:

```bash
python -m plank.main path/to/file.plank
```

If no file is provided, `plank.main` starts the interactive REPL instead. You can also launch the REPL directly:

```bash
python -m plank.repl
```

## REPL Example

When the REPL starts it prints some example commands. A snippet from `plank/repl.py` looks like:

```python
print("--- Mini Language REPL ---")
print("Type 'quit' to exit.")
print("Examples:")
print("  a <- 10; b <- 5; out <- a + b")
print("  a +<- 2; out <- a")
print("  out <- 'Hello' <- ' ' <- 'World!'")
```

Type expressions or statements at the `>>>` prompt. Multi-line blocks such as loops use `...` continuation lines.

## Language Overview

The language supports:

- Variables and assignment using `<-` with augmented forms like `+<-`.
- Arithmetic on numbers and strings.
- Lists and indexing.
- `for` ranges and `while` loops with `break`/`continue`.
- Lambda functions, closures and currying (`c` keyword).
- Conditionals with `if` / `else`.
- Built‑in helpers such as `len`, `head`, `tail`, `push`, `pop`, `contains`,
  `remove`, `insert`, `extend`, `keys`, `values`, `items`, `union`,
  `intersection`, `difference`, `map`, `filter` and more.

See `plank/tests/test_interpreter.py` for many usage examples.

## Running Tests

Install any dependencies and run:

```bash
pytest
```

`pytest` discovers the unit tests in `plank/tests/` and executes them.
