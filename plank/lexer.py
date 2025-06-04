from dataclasses import dataclass
from plank.token_types import TokenType
from plank.token_types import *

# Mapping of keywords to token types and values
KEYWORDS = {
	'int': (KEYWORD_INT, 'int'),
	'out': (KEYWORD_OUT, 'out'),
	'for': (KEYWORD_FOR, 'for'),
	'true': (KEYWORD_TRUE, True),
	'false': (KEYWORD_FALSE, False),
	'and': (KEYWORD_AND, 'and'),
	'or': (KEYWORD_OR, 'or'),
	'not': (KEYWORD_NOT, 'not'),
	'while': (KEYWORD_WHILE, 'while'),
	'if': (KEYWORD_IF, 'if'),
	'else': (KEYWORD_ELSE, 'else'),
	'c': (KEYWORD_C, 'c'),
}

# Mapping of single-character tokens to their types
SINGLE_CHAR_TOKENS = {
	',': COMMA,
	'(': LPAREN,
	')': RPAREN,
	'{': LBRACE,
	'}': RBRACE,
	';': SEMICOLON,
	'[': LBRACKET,
	']': RBRACKET,
}


# --- Token Class ---
# Represents a single token found by the lexer.
@dataclass(slots=True)
class Token:
	type: TokenType
	value: object

	def __str__(self):
		"""String representation of the Token object."""
		return f"Token({self.type}, {repr(self.value)})"

	__repr__ = __str__


# --- Lexer (Tokenizer) ---
# Reads the input text and converts it into a stream of tokens.
class Lexer:
	def __init__(self, text):
		self.text = text
		self.pos = 0  # Current position in the input text
		self.current_char = self.text[self.pos] if self.text else None	# Current character
	
	def advance(self):
		"""Move to the next character in the input text."""
		self.pos += 1
		if self.pos < len(self.text):
			self.current_char = self.text[self.pos]
		else:
			self.current_char = None  # Indicates end of input
	
	def peek(self, offset=1):
		"""Look ahead without advancing the position."""
		peek_pos = self.pos + offset
		if peek_pos < len(self.text):
			return self.text[peek_pos]
		return None
	
	def skip_whitespace(self):
		"""Skip over any whitespace characters."""
		while self.current_char is not None and self.current_char.isspace():
			self.advance()
	
	def integer(self):
		"""Parse an integer literal from the input."""
		result = ''
		while self.current_char is not None and self.current_char.isdigit():
			result += self.current_char
			self.advance()
		return int(result)
	
	def string(self):
		"""Parse a string literal from the input (enclosed in double or single quotes).
		Handles escape sequences: \n, \t, \\
		"""
		quote_char = self.current_char	# Capture the opening quote character
		self.advance()	# Consume the opening quote
		result = ''
		while self.current_char is not None and self.current_char != quote_char:
			if self.current_char == '\\':  # Handle escape sequences
				self.advance()	# Consume the backslash
				if self.current_char == 'n':
					result += '\n'
				elif self.current_char == 't':
					result += '\t'
				elif self.current_char == '\\':
					result += '\\'
				else:
					raise Exception(
						f'Lexer error: Invalid escape sequence \\{self.current_char} at position {self.pos - 1}')
			else:
				result += self.current_char
			self.advance()
		if self.current_char is None:
			raise Exception(f'Lexer error: Unclosed string literal starting with {quote_char} at position {self.pos}')
		self.advance()	# Consume the closing quote
		return result
	
	def _id(self):
		"""Parse an identifier or keyword."""
		result = ''
		while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
			result += self.current_char
			self.advance()
		if result in KEYWORDS:
			return Token(*KEYWORDS[result])
		return Token(IDENTIFIER, result)
		
	def get_next_token(self):
		"""Get the next token from the input text."""
		while self.current_char is not None:
			self.skip_whitespace()	# Always skip whitespace before processing
			
			if self.current_char is None:  # Check again after skipping whitespace
				break
			
			if self.current_char.isdigit():
				return Token(INTEGER, self.integer())
			
			if self.current_char == '"' or self.current_char == "'":
				return Token(STRING, self.string())
			
			if self.current_char.isalpha() or self.current_char == '_':
				return self._id()
			
			# Handle multi-character operators and augmented assignments (longest match first)
			if self.current_char == '*':
				if self.peek() == '*':
					self.advance()	# Consume first '*'
					if self.peek() == '<' and self.peek(2) == '-':
						self.advance()	# Consume second '*'
						self.advance()	# Consume '<'
						self.advance()	# Consume '-'
						return Token(EXPONENT_ASSIGN, '**<-')
					self.advance()	# Consume second '*'
					return Token(EXPONENT, '**')
				elif self.peek() == '<' and self.peek(2) == '-':
					self.advance()	# Consume '*'
					self.advance()	# Consume '<'
					self.advance()	# Consume '-'
					return Token(MULTIPLY_ASSIGN, '*<-')
				self.advance()
				return Token(MULTIPLY, '*')
			
			if self.current_char == '/':
				if self.peek() == '/':
					self.advance()	# Consume first '/'
					if self.peek() == '<' and self.peek(2) == '-':
						self.advance()	# Consume second '/'
						self.advance()	# Consume '<'
						self.advance()	# Consume '-'
						return Token(FLOOR_DIVIDE_ASSIGN, '//<-')
					self.advance()	# Consume second '/'
					return Token(FLOOR_DIVIDE, '//')
				elif self.peek() == '<' and self.peek(2) == '-':
					self.advance()	# Consume '/'
					self.advance()	# Consume '<'
					self.advance()	# Consume '-'
					return Token(DIVIDE_ASSIGN, '/<-')
				self.advance()
				return Token(DIVIDE, '/')
			
			if self.current_char == '<':
				self.advance()
				if self.current_char == '-':
					self.advance()
					return Token(ASSIGN, '<-')
				elif self.current_char == '=':
					self.advance()
					return Token(LTE, '<=')
				return Token(LT, '<')
			
			if self.current_char == '>':
				self.advance()
				if self.current_char == '=':
					self.advance()
					return Token(GTE, '>=')
				return Token(GT, '>')
			
			if self.current_char == '=':
				self.advance()
				if self.current_char == '=':
					self.advance()
					return Token(EQ, '==')
				raise Exception(f'Lexer error: Invalid character sequence starting with = at position {self.pos - 1}')
			
			if self.current_char == '!':
				self.advance()
				if self.current_char == '=':
					self.advance()
					return Token(NEQ, '!=')
				raise Exception(f'Lexer error: Invalid character sequence starting with ! at position {self.pos - 1}')
			
			if self.current_char == '-':
				if self.peek() == '>':
					self.advance()	# Consume '-'
					self.advance()	# Consume '>'
					return Token(ARROW, '->')
				elif self.peek() == '<' and self.peek(2) == '-':
					self.advance()	# Consume '-'
					self.advance()	# Consume '<'
					self.advance()	# Consume '-'
					return Token(MINUS_ASSIGN, '-<-')
				self.advance()
				return Token(MINUS, '-')
			
			if self.current_char == '+':
				if self.peek() == '<' and self.peek(2) == '-':
					self.advance()	# Consume '+'
					self.advance()	# Consume '<'
					self.advance()	# Consume '-'
					return Token(PLUS_ASSIGN, '+<-')
				self.advance()
				return Token(PLUS, '+')
			
			if self.current_char == '.':
				self.advance()
				if self.current_char == '.':
					self.advance()
					return Token(RANGE_OP, '..')
				raise Exception(f'Lexer error: Invalid character sequence starting with . at position {self.pos - 1}')
			
			# Handle single-character operators and punctuation
			if self.current_char in SINGLE_CHAR_TOKENS:
				tok_type = SINGLE_CHAR_TOKENS[self.current_char]
				char = self.current_char
				self.advance()
				return Token(tok_type, char)
			
			# If none of the above, it's an unexpected character
			raise Exception(f'Lexer error: Invalid character "{self.current_char}" at position {self.pos}')
		return Token(EOF, None)	 # Return EOF when all input is processed
