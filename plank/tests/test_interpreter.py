import contextlib
import io
from contextlib import nullcontext
from unittest.mock import patch

import pytest

from plank.interpreter import Interpreter
from plank.lexer import Lexer
from plank.parser import Parser


def run_plank(code: str, user_input: str | None = None) -> str:
    """Execute code and capture printed output."""
    parser = Parser(Lexer(code))
    interpreter = Interpreter(parser)
    buf = io.StringIO()
    patch_ctx = patch('builtins.input', return_value=user_input) if user_input is not None else nullcontext()
    with patch_ctx, contextlib.redirect_stdout(buf):
        interpreter.interpret()
    return buf.getvalue()


@pytest.mark.parametrize(
    "code,expected",
    [
        ("a <- 10; b <- 5; out <- a + b", 15),
        ("a <- 0; a +<- 2; out <- a", 2),
        ("out <- 'Hello' <- ' ' <- 'World!'", "Hello World!"),
        ("out <- 'abc' * 3", "abcabcabc"),
        ("out <- 'Line 1\\nLine 2\\tTabbed' <- '\\n'", "Line 1\nLine 2\tTabbed\n"),
        ("out <- true and false; out <- '\\n'", False),
        ("out <- not true; out <- '\\n'", False),
        ("(x for 1..3) -> { out <- x out <- ' ' }", None),
        ("(x for 2..10..2) -> { out <- x out <- ' ' }", "2 4 6 8 10 "),
        ("(x for 5..1..-1) -> { out <- x out <- ' ' }", "5 4 3 2 1 "),
        ("x <- 0; (x while x < 5) -> { out <- x; x +<- 1; out <- '\\n' }", None),
        ("my_list <- [10, 'hello', true]; out <- my_list[1]; out <- '\\n'", "hello"),
        (
            "my_list <- [10, 'hello', true]; my_list[0] <- 99; out<- my_list[0]; out <- '\\n'",
            99,
        ),
        ("my_list <- 50; out <- my_list; out <- '\\n'", 50),
        ("square <- (x) <- x * x; out <- square(4); out <- '\\n'", 16),
        ("add <- (a, b) <- a + b; out <- add(10, 20); out <- '\\n'", 30),
        (
            "closure_example <- (y) <- (x) <- x + y; f <- closure_example(10); out <- f(5); out <- '\\n'",
            15,
        ),
        ("adder_block <- (x) <- { y <- x + 1; y }; out <- adder_block(4); out <- '\\n'", 5),
        (
            "curried_sum <- c(a, b) <- a + b; add_five <- curried_sum(5); out <- add_five(3); out <- '\\n'",
            8,
        ),
        (
            "x <- 5; result <- if x < 0 -> { 'negative' } else if x == 0 -> { 'zero' } else { 'positive' }; out <- result",
            "positive",
        ),
    ],
)
def test_examples(code: str, expected):
    output = run_plank(code)
    if expected is not None:
        normalized = output if isinstance(expected, str) and expected.endswith("\n") else output.rstrip("\n")
        assert normalized == str(expected)


@pytest.mark.parametrize(
    "code,inp,expected",
    [
        ("a, b <- int; out <- a + b", "3 4", "7"),
        ("s <- string; out <- s", "hello", "hello"),
        ("b1, b2 <- bool; out <- b1 and b2", "true false", "False"),
        ("lst <- list; out <- lst[1]", "[1, 2, 3]", "2"),
        ("out <- ('123' -> int) + 1", None, "124"),
    ],
)
def test_input_and_cast(code: str, inp: str | None, expected: str):
    output = run_plank(code, inp)
    assert output == expected
