# main.py

import sys

from plank.interpreter import Interpreter
from plank.lexer import Lexer
from plank.parser import Parser
from plank.repl import main_repl


def run_file(filename):
	with open(filename, 'r') as f:
		code = f.read()
	lexer = Lexer(code)
	parser = Parser(lexer)
	interpreter = Interpreter(parser)
	interpreter.interpret()


def main():
	if len(sys.argv) == 2:
		run_file(sys.argv[1])
	else:
		print("Plank REPL. Run with `main.py filename.plank` to execute a file.")
		main_repl()


if __name__ == '__main__':
	main()
