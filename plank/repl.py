from plank.interpreter import Interpreter
from plank.lexer import Lexer
from plank.parser import Parser

if __name__ == '__main__':
	print("--- Mini Language REPL ---")
	print("Type 'quit' to exit.")
	print("Examples:")
	print("  a <- 10; b <- 5; out <- a + b")
	print("  a +<- 2; out <- a")
	print("  out <- 'Hello' <- ' ' <- 'World!'")
	print("  out <- 'abc' * 3")
	print("  out <- 'Line 1\\nLine 2\\tTabbed' <- '\\n'")
	print("  out <- true and false; out <- '\\n'")
	print("  out <- not true; out <- '\\n'")
	print("  (x for 1..3) -> { out <- x out <- ' ' }")
	print("  x <- 0; (x while x < 5) -> { out <- x; x +<- 1; out <- '\\n' }")
	print("  my_list <- [10, 'hello', true]; out <- my_list[1]; out <- '\\n'")
	print("  my_list[0] <- 99; out <- my_list[0]; out <- '\\n'")
	print("  my_list <- 50; out <- my_list; out <- '\\n'")  # Dynamic typing example
	print("  square <- (x) <- x * x; out <- square(4); out <- '\\n'")
	print("  add <- (a, b) <- a + b; out <- add(10, 20); out <- '\\n'")
	print("  closure_example <- (y) <- (x) <- x + y; f <- closure_example(10); out <- f(5); out <- '\\n'")
	print("  # New: Lambda with block body (implicit return)")
	print("  adder_block <- (x) <- { y <- x + 1; y }; out <- adder_block(4); out <- '\\n'")
	print("  # New: Currying example")
	print("  curried_sum <- c(a, b) <- a + b; add_five <- curried_sum(5); out <- add_five(3); out <- '\\n'")
	print("-" * 30)
	
	interpreter = Interpreter()  # Initialize interpreter without a parser initially
	
	while True:
		try:
			line = input(">>> ")
			if line.lower() == 'quit':
				break
			
			# Simple multi-line input handling for blocks like for/while loops
			full_program_text = line
			if line.strip().endswith('{'):
				while True:
					next_line = input("... ")
					full_program_text += "\n" + next_line
					if next_line.strip() == '}':
						break
			
			lexer = Lexer(full_program_text)
			parser = Parser(lexer)
			interpreter.parser = parser  # Set the parser for the interpreter
			interpreter.interpret()
			# Print the global scope (the first scope in the stack)
			print("\nVariables (Global Scope):", interpreter.scopes[0])
		
		except EOFError:
			print("\nExiting REPL.")
			break
		except Exception as e:
			print(f"Error: {e}")
