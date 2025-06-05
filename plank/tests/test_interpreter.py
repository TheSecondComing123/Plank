import contextlib
import io
import unittest
import unittest.mock

from plank.interpreter import Interpreter
from plank.lexer import Lexer
from plank.parser import Parser


def evaluate(code):
        lexer = Lexer(code)
        parser = Parser(lexer)
        interpreter = Interpreter(parser)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
                result = interpreter.interpret()
        output = buffer.getvalue()
        return result, output

def evaluate_with_input(code, inp):
        lexer = Lexer(code)
        parser = Parser(lexer)
        interpreter = Interpreter(parser)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
                with unittest.mock.patch('builtins.input', return_value=inp):
                        result = interpreter.interpret()
        output = buffer.getvalue()
        return result, output


class TestPlankExamples(unittest.TestCase):
        
        def test_examples(self):
                cases = [
                        ("a <- 10; b <- 5; out <- a + b", 15),
                        ("a <- 0; a +<- 2; out <- a", 2),
                        ("out <- 'Hello' <- ' ' <- 'World!'", "Hello World!"),
                        ("out <- 'abc' * 3", "abcabcabc"),
                        ("out <- 'Line 1\\nLine 2\\tTabbed' <- '\\n'", "Line 1\nLine 2\tTabbed\n"),
                        ("out <- true and false; out <- '\\n'", False),
                        ("out <- not true; out <- '\\n'", False),
                        # may just print
                        ("(x for 1..3) -> { out <- x out <- ' ' }", None),
                                ("(x for 2..10..2) -> { out <- x out <- ' ' }", "2 4 6 8 10 "),
                                ("(x for 5..1..-1) -> { out <- x out <- ' ' }", "5 4 3 2 1 "),
                        ("x <- 0; (x while x < 5) -> { out <- x; x +<- 1; out <- '\\n' }", None),
                        ("my_list <- [10, 'hello', true]; out <- my_list[1]; out <- '\\n'", 'hello'),
                        ("my_list <- [10, 'hello', true]; my_list[0] <- 99; out <- my_list[0]; out <- '\n'", 99),
                        ("my_list <- 50; out <- my_list; out <- '\\n'", 50),
                        ("square <- (x) <- x * x; out <- square(4); out <- '\\n'", 16),
                        ("add <- (a, b) <- a + b; out <- add(10, 20); out <- '\\n'", 30),
                        ("closure_example <- (y) <- (x) <- x + y; f <- closure_example(10); out <- f(5); out <- '\\n'", 15),
                        ("adder_block <- (x) <- { y <- x + 1; y }; out <- adder_block(4); out <- '\\n'", 5),
                        ("curried_sum <- c(a, b) <- a + b; add_five <- curried_sum(5); out <- add_five(3); out <- '\\n'", 8),
                        (
                                "x <- 5; result <- if x < 0 -> { 'negative' } else if x == 0 -> { 'zero' } else { 'positive' }; out <- result",
                                'positive'),
                ]
                
                for code, expected in cases:
                        with self.subTest(code=code):
                                result, output = evaluate(code)
                                if expected is not None:
                                        expected_str = str(expected)
                                        out = output
                                        if not (isinstance(expected, str) and expected.endswith('\n')):
                                                out = out.rstrip('\n')
                                        self.assertEqual(out, expected_str)

        def test_input_and_cast(self):
                # integer input with whitespace
                code = "a, b <- int; out <- a + b"
                result, output = evaluate_with_input(code, "3 4")
                self.assertEqual(output, "7")

                # string input
                code = "s <- string; out <- s"
                result, output = evaluate_with_input(code, "hello")
                self.assertEqual(output, "hello")

                # bool input
                code = "b1, b2 <- bool; out <- b1 and b2"
                result, output = evaluate_with_input(code, "true false")
                self.assertEqual(output, "False")

                # list input and indexing
                code = "lst <- list; out <- lst[1]"
                result, output = evaluate_with_input(code, "[1, 2, 3]")
                self.assertEqual(output, "2")

                # type casting
                code = "out <- ('123' -> int) + 1"
                result, output = evaluate(code)
                self.assertEqual(output, "124")


if __name__ == '__main__':
        unittest.main()
