import contextlib
import io
import unittest
import unittest.mock
from dataclasses import dataclass
from typing import Any, Optional

from plank.interpreter import Interpreter
from plank.lexer import Lexer
from plank.parser import Parser


@dataclass
class Case:
    code: str
    expected: Any | None
    input_data: Optional[str] = None


def run(code: str, input_data: Optional[str] = None):
    lexer = Lexer(code)
    parser = Parser(lexer)
    interpreter = Interpreter(parser)
    buffer = io.StringIO()
    input_patch = unittest.mock.patch('builtins.input', return_value=input_data) if input_data is not None else contextlib.nullcontext()
    with contextlib.redirect_stdout(buffer), input_patch:
        result = interpreter.interpret()
    return result, buffer.getvalue()


class PlankTest(unittest.TestCase):
    sections = {
        "arithmetic": [
            Case("a <- 10; b <- 5; out <- a + b", 15),
            Case("a <- 0; a +<- 2; out <- a", 2),
        ],
        "strings": [
            Case("out <- 'Hello' <- ' ' <- 'World!'", "Hello World!"),
            Case("out <- 'abc' * 3", "abcabcabc"),
            Case("out <- 'Line 1\\nLine 2\\tTabbed' <- '\\n'", "Line 1\nLine 2\tTabbed\n"),
        ],
        "booleans": [
            Case("out <- true and false; out <- '\\n'", False),
            Case("out <- not true; out <- '\\n'", False),
        ],
        "loops": [
            Case("(x for 1..3) -> { out <- x out <- ' ' }", None),
            Case("(x for 2..10..2) -> { out <- x out <- ' ' }", "2 4 6 8 10 "),
            Case("(x for 5..1..-1) -> { out <- x out <- ' ' }", "5 4 3 2 1 "),
            Case("x <- 0; (x while x < 5) -> { out <- x; x +<- 1; out <- '\\n' }", None),
        ],
        "lists": [
            Case("my_list <- [10, 'hello', true]; out <- my_list[1]; out <- '\\n'", "hello"),
            Case("my_list <- [10, 'hello', true]; my_list[0] <- 99; out<- my_list[0]; out <- '\\n'", 99),
            Case("my_list <- 50; out <- my_list; out <- '\\n'", 50),
        ],
        "functions": [
            Case("square <- (x) <- x * x; out <- square(4); out <- '\\n'", 16),
            Case("add <- (a, b) <- a + b; out <- add(10, 20); out <- '\\n'", 30),
            Case("closure_example <- (y) <- (x) <- x + y; f <- closure_example(10); out <- f(5); out <- '\\n'", 15),
            Case("adder_block <- (x) <- { y <- x + 1; y }; out <- adder_block(4); out <- '\\n'", 5),
            Case("curried_sum <- c(a, b) <- a + b; add_five <- curried_sum(5); out <- add_five(3); out <- '\\n'", 8),
        ],
        "conditionals": [
            Case("x <- 5; result <- if x < 0 -> { 'negative' } else if x == 0 -> { 'zero' } else { 'positive' }; out <- result", "positive"),
        ],
        "input and cast": [
            Case("a, b <- int; out <- a + b", "7", "3 4"),
            Case("s <- string; out <- s", "hello", "hello"),
            Case("b1, b2 <- bool; out <- b1 and b2", "False", "true false"),
            Case("lst <- list; out <- lst[1]", "2", "[1, 2, 3]"),
            Case("out <- ('123' -> int) + 1", "124"),
        ],
        "builtins": [
            Case("out <- len([1, 2, 3])", 3),
            Case("out <- len [1, 2, 3]", 3),
            Case("out <- head([10,20,30])", 10),
            Case("out <- head [10,20,30]", 10),
            Case("out <- tail([1,2,3])", "[2, 3]"),
            Case("out <- tail [1,2,3]", "[2, 3]"),
            Case("out <- 1 : [2,3]", "[1, 2, 3]"),
            Case("out <- [1,2] : 3", "[1, 2, 3]"),
            Case("out <- [1,2] : [3,4]", "[1, 2, 3, 4]"),
            Case("out <- abs(-5)", 5),
            Case("out <- abs -5", 5),
            Case("out <- min(3, 7)", 3),
            Case("out <- min 3, 7", 3),
            Case("out <- max(3, 7)", 7),
            Case("out <- max 3, 7", 7),
            Case("out <- clamp(5, 0, 10)", 5),
            Case("out <- clamp 5, 0, 10", 5),
            Case("lst <- [1]; push(lst, 2); out <- lst[1]", 2),
            Case("lst <- [1,2]; out <- pop(lst)", 2),
            Case("out <- map((x) <- x + 1, [1,2])[1]", 3),
            Case("out <- filter((x) <- x > 1, [1,2,3])[0]", 2),
            Case("out <- foldl((a,b) <- a + b, 0, [1,2,3])", 6),
            Case("out <- foldr((a,b) <- a - b, 0, [1,2,3])", 2),
            Case("out <- sort([3,1,2])[0]", 1),
            Case("out <- split('a,b,c', ',')[1]", 'b'),
            Case("out <- join('-', ['a','b','c'])", 'a-b-c'),
            Case("out <- find('hello', 'e')", 1),
            Case("out <- replace('hello', 'l', 'x')", 'hexxo'),
            Case("out <- zip([1,2], [3,4])[1][1]", 4),
            Case("out <- enumerate(['a','b'])[1][0]", 1),
        ],
        "comments": [
            Case("a <- 1 # assign a\nout <- a", 1),
            Case("# only comment line\nout <- 2", 2),
        ],
    }

    def test_sections(self):
        for section, cases in self.sections.items():
            for case in cases:
                with self.subTest(section=section, code=case.code):
                    _, output = run(case.code, case.input_data)
                    if case.expected is not None:
                        expected_str = str(case.expected)
                        out = output if isinstance(case.expected, str) and case.expected.endswith("\n") else output.rstrip("\n")
                        self.assertEqual(out, expected_str)


if __name__ == "__main__":
    unittest.main()
