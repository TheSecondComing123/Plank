import unittest

from plank.interpreter import Interpreter
from plank.lexer import Lexer
from plank.parser import Parser


def evaluate(code):
	lexer = Lexer(code)
	parser = Parser(lexer)
	interpreter = Interpreter(parser)
	return interpreter.interpret()


class TestPlankExamples(unittest.TestCase):
	
	def test_examples(self):
		cases = [
			("a <- 10; b <- 5; out <- a + b", 15),
			("a +<- 2; out <- a", 2),
			("out <- 'Hello' <- ' ' <- 'World!'", "Hello World!"),
			("out <- 'abc' * 3", "abcabcabc"),
			("out <- 'Line 1\\nLine 2\\tTabbed' <- '\\n'", "Line 1\nLine 2\tTabbed\n"),
			("out <- true and false; out <- '\\n'", False),
			("out <- not true; out <- '\\n'", False),
			# may just print
			("(x for 1..3) -> { out <- x out <- ' ' }", None),
			("x <- 0; (x while x < 5) -> { out <- x; x +<- 1; out <- '\\n' }", None),
			("my_list <- [10, 'hello', true]; out <- my_list[1]; out <- '\\n'", 'hello'),
			("my_list[0] <- 99; out <- my_list[0]; out <- '\\n'", 99),
			("my_list <- 50; out <- my_list; out <- '\\n'", 50),
			("square <- (x) <- x * x; out <- square(4); out <- '\\n'", 16),
			("add <- (a, b) <- a + b; out <- add(10, 20); out <- '\\n'", 30),
			("closure_example <- (y) <- (x) <- x + y; f <- closure_example(10); out <- f(5); out <- '\\n'", 15),
			("adder_block <- (x) <- { y <- x + 1; y }; out <- adder_block(4); out <- '\\n'", 5),
			("curried_sum <- c(a, b) <- a + b; add_five <- curried_sum(5); out <- add_five(3); out <- '\\n'", 8),
			(
			"x <- 5; result <- if x < 0 -> { 'negative' } else if x == 0 -> { 'zero' } else { 'positive' }; out <- result; out <- '\\n'",
			'positive'),
			(
			"x <- 0; result <- cond(x < 0) -> { 'neg' } .else_if(x == 0) -> { 'zero' } .else -> { 'pos' }; out <- result; out <- '\\n'",
			'zero'),
		]
		
		for code, expected in cases:
			with self.subTest(code=code):
				result = evaluate(code)
				if expected is not None:
					self.assertEqual(result, expected)


if __name__ == '__main__':
	unittest.main()
