import sys

# --- Token Types ---
# Define constants for all token types our language will recognize.
EOF = 'EOF'  # End of File
IDENTIFIER = 'IDENTIFIER'  # Variable names (e.g., 'a', 'b', 'result')
INTEGER = 'INTEGER'  # Integer literals (e.g., '10', '42')
STRING = 'STRING'  # String literals (e.g., '"hello"', '" "')

# Arithmetic Operators
PLUS = 'PLUS'  # '+'
MINUS = 'MINUS'	 # '-'
MULTIPLY = 'MULTIPLY'  # '*'
DIVIDE = 'DIVIDE'  # '/'
EXPONENT = 'EXPONENT'  # '**'
FLOOR_DIVIDE = 'FLOOR_DIVIDE'  # '//'

# Assignment and Keywords
ASSIGN = 'ASSIGN'  # '<-'
KEYWORD_INT = 'KEYWORD_INT'  # 'int' (for input type)
KEYWORD_OUT = 'KEYWORD_OUT'  # 'out' (for output statement)

# For Loop Keywords and Operators
KEYWORD_FOR = 'KEYWORD_FOR'  # 'for'
RANGE_OP = 'RANGE_OP'  # '..'
ARROW = 'ARROW'	 # '->'

# Augmented Assignment Operators
PLUS_ASSIGN = 'PLUS_ASSIGN'  # '+<-'
MINUS_ASSIGN = 'MINUS_ASSIGN'  # '-<-'
MULTIPLY_ASSIGN = 'MULTIPLY_ASSIGN'  # '*<-'
DIVIDE_ASSIGN = 'DIVIDE_ASSIGN'	 # '/<-'
EXPONENT_ASSIGN = 'EXPONENT_ASSIGN'  # '**<-'
FLOOR_DIVIDE_ASSIGN = 'FLOOR_DIVIDE_ASSIGN'  # '//<-'

# Boolean Literals and Logical Operators
KEYWORD_TRUE = 'KEYWORD_TRUE'  # 'true'
KEYWORD_FALSE = 'KEYWORD_FALSE'	 # 'false'
KEYWORD_AND = 'KEYWORD_AND'  # 'and'
KEYWORD_OR = 'KEYWORD_OR'  # 'or'
KEYWORD_NOT = 'KEYWORD_NOT'  # 'not'
KEYWORD_WHILE = 'KEYWORD_WHILE'	 # 'while'
KEYWORD_C = 'KEYWORD_C'	 # 'c' for curried functions

# Comparison Operators
EQ = 'EQ'  # '=='
NEQ = 'NEQ'  # '!='
LT = 'LT'  # '<'
GT = 'GT'  # '>'
LTE = 'LTE'  # '<='
GTE = 'GTE'  # '>='

# Punctuation
COMMA = 'COMMA'	 # ','
LPAREN = 'LPAREN'  # '('
RPAREN = 'RPAREN'  # ')'
LBRACE = 'LBRACE'  # '{'
RBRACE = 'RBRACE'  # '}'
SEMICOLON = 'SEMICOLON'	 # ';'

# List Punctuation
LBRACKET = 'LBRACKET'  # '['
RBRACKET = 'RBRACKET'  # ']'


# --- Token Class ---
# Represents a single token found by the lexer.
class Token:
	def __init__(self, type, value):
		self.type = type
		self.value = value
	
	def __str__(self):
		"""String representation of the Token object."""
		return f"Token({self.type}, {repr(self.value)})"
	
	def __repr__(self):
		"""Official string representation (for debugging)."""
		return self.__str__()


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
		"""Parse an identifier or a keyword."""
		result = ''
		while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
			result += self.current_char
			self.advance()
		# Check if the identifier is a reserved keyword
		if result == 'int':
			return Token(KEYWORD_INT, 'int')
		elif result == 'out':
			return Token(KEYWORD_OUT, 'out')
		elif result == 'for':
			return Token(KEYWORD_FOR, 'for')
		elif result == 'true':
			return Token(KEYWORD_TRUE, True)
		elif result == 'false':
			return Token(KEYWORD_FALSE, False)
		elif result == 'and':
			return Token(KEYWORD_AND, 'and')
		elif result == 'or':
			return Token(KEYWORD_OR, 'or')
		elif result == 'not':
			return Token(KEYWORD_NOT, 'not')
		elif result == 'while':
			return Token(KEYWORD_WHILE, 'while')
		elif result == 'c':
			return Token(KEYWORD_C, 'c')
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
			if self.current_char == ',':
				self.advance()
				return Token(COMMA, ',')
			if self.current_char == '(':
				self.advance()
				return Token(LPAREN, '(')
			if self.current_char == ')':
				self.advance()
				return Token(RPAREN, ')')
			if self.current_char == '{':
				self.advance()
				return Token(LBRACE, '{')
			if self.current_char == '}':
				self.advance()
				return Token(RBRACE, '}')
			if self.current_char == ';':
				self.advance()
				return Token(SEMICOLON, ';')
			if self.current_char == '[':
				self.advance()
				return Token(LBRACKET, '[')
			if self.current_char == ']':
				self.advance()
				return Token(RBRACKET, ']')
			
			# If none of the above, it's an unexpected character
			raise Exception(f'Lexer error: Invalid character "{self.current_char}" at position {self.pos}')
		return Token(EOF, None)	 # Return EOF when all input is processed


# --- AST (Abstract Syntax Tree) Nodes ---
# Classes representing different nodes in our program's syntax tree.
class AST:
	pass


class Program(AST):
	def __init__(self, statements):
		self.statements = statements  # A list of statement nodes


class BinOp(AST):
	def __init__(self, left, op, right):
		self.left = left  # Left-hand side expression
		self.op = op  # Operator token (+, -, *, etc.)
		self.right = right  # Right-hand side expression


class UnaryOp(AST):
	def __init__(self, op, right):
		self.op = op  # Operator token (e.g., 'not')
		self.right = right  # Operand


class Num(AST):
	def __init__(self, token):
		self.token = token
		self.value = token.value  # The integer value


class String(AST):
	def __init__(self, token):
		self.token = token
		self.value = token.value  # The string value


class Boolean(AST):
	def __init__(self, token):
		self.token = token
		self.value = token.value  # The boolean value (True/False)


class Var(AST):
	def __init__(self, token):
		self.token = token
		self.value = token.value  # The variable name (string)


class ListLiteral(AST):
	def __init__(self, elements):
		self.elements = elements  # A list of AST nodes (expressions)


class IndexAccess(AST):
	def __init__(self, list_expr, index_expr):
		self.list_expr = list_expr  # The expression that evaluates to a list (e.g., a Var node)
		self.index_expr = index_expr  # The expression that evaluates to the index


class ListAssign(AST):
	def __init__(self, list_var, index_expr, value_expr):
		self.list_var = list_var  # The Var node representing the list variable
		self.index_expr = index_expr  # The expression for the index
		self.value_expr = value_expr  # The expression for the value to assign


class Lambda(AST):
	def __init__(self, params, body, is_curried=False):
		self.params = params  # List of Var nodes for parameters
		self.body = body  # AST node for the lambda's body (expression or Program)
		self.is_curried = is_curried  # Flag for curried function


class Call(AST):
	def __init__(self, func_expr, args):
		self.func_expr = func_expr  # The expression evaluating to the callable
		self.args = args  # List of AST nodes for arguments


class Assign(AST):
	def __init__(self, left, op, right):
		self.left = left  # The variable (Var node) being assigned to
		self.op = op  # The ASSIGN token ('<-')
		self.right = right  # The expression whose value is assigned


class AugmentedAssign(AST):
	def __init__(self, left, op, right):
		self.left = left  # The variable (Var node) being assigned to
		self.op = op  # The augmented assignment token (e.g., '+<-')
		self.right = right  # The expression whose value is used in the operation


class InputStatement(AST):
	def __init__(self, variables, type_token):
		self.variables = variables  # List of Var nodes to store input
		self.type_token = type_token  # The KEYWORD_INT token


class OutputStatement(AST):
	def __init__(self, expressions):  # Now takes a list of expressions
		self.expressions = expressions	# List of expressions to be printed


class ForStatement(AST):
	def __init__(self, variable, start_expr, end_expr, step_expr, body):
		self.variable = variable  # The loop variable (Var node)
		self.start_expr = start_expr  # Expression for the start of the range
		self.end_expr = end_expr  # Expression for the end of the range
		self.step_expr = step_expr  # Expression for the step value (may be None)
		self.body = body  # List of statements in the loop body


class WhileStatement(AST):
	def __init__(self, loop_var, condition, body):
		self.loop_var = loop_var  # The variable for the loop (e.g., 'x')
		self.condition = condition  # The expression representing the loop condition (e.g., 'x < 5')
		self.body = body  # List of statements in the loop body


class ExpressionStatement(AST):	 # New AST node for expressions treated as statements
	def __init__(self, expression):
		self.expression = expression


# --- Parser ---
# Builds the Abstract Syntax Tree from the token stream, respecting operator precedence.
class Parser:
	def __init__(self, lexer):
		self.lexer = lexer
		self.current_token = self.lexer.get_next_token()  # The current token being processed
	
	def error(self, message="Invalid syntax"):
		"""Report a parsing error."""
		raise Exception(f'Parser error: {message} at token {self.current_token}')
	
	def eat(self, token_type):
		"""
		Consume the current token if its type matches the expected type,
		then advance to the next token. Otherwise, raise an error.
		"""
		if self.current_token.type == token_type:
			self.current_token = self.lexer.get_next_token()
		else:
			self.error(f"Expected token type {token_type}, but got {self.current_token.type}")
	
	def program(self):
		"""
		Parses the entire program.
		program ::= (statement (SEMICOLON statement)*)* EOF
		"""
		statements = []
		while self.current_token.type != EOF:
			statements.append(self.statement())
			# Allow multiple statements on one line separated by semicolons
			while self.current_token.type == SEMICOLON:
				self.eat(SEMICOLON)
				if self.current_token.type != EOF and self.current_token.type != RBRACE:  # Don't parse empty statement at end of block
					statements.append(self.statement())
		return Program(statements)
	
	def statement(self):
		"""
		Parses a single statement.
		statement ::= output_statement | for_statement | while_statement |
					  input_statement | list_assignment_statement |
					  assignment_statement | augmented_assignment_statement |
					  expression_statement
		"""
		# Try parsing specific statement types first
		if self.current_token.type == KEYWORD_OUT:
			return self.output_statement()
		elif self.current_token.type == LPAREN:
			# Look ahead for 'for' or 'while' to distinguish loops from expressions
			lexer_pos_backup = self.lexer.pos
			lexer_current_char_backup = self.lexer.current_char
			parser_current_token_backup = self.current_token
			try:
				self.eat(LPAREN)
				# Check for IDENTIFIER followed by KEYWORD_FOR or KEYWORD_WHILE
				if self.current_token.type == IDENTIFIER:
					self.eat(IDENTIFIER)  # Consume loop var
					if self.current_token.type == KEYWORD_FOR:
						# It's a for loop, reset and parse
						self.lexer.pos = lexer_pos_backup
						self.lexer.current_char = lexer_current_char_backup
						self.current_token = parser_current_token_backup
						return self.for_statement()
					elif self.current_token.type == KEYWORD_WHILE:
						# It's a while loop, reset and parse
						self.lexer.pos = lexer_pos_backup
						self.lexer.current_char = lexer_current_char_backup
						self.current_token = parser_current_token_backup
						return self.while_statement()
				# If we get here, it was LPAREN but not a loop structure, so it's an expression.
				# Reset and let the general expression parsing handle it.
				self.lexer.pos = lexer_pos_backup
				self.lexer.current_char = lexer_current_char_backup
				self.current_token = parser_current_token_backup
				return self.expression_statement()  # Fallback to expression statement
			except Exception:
				# If an error occurred during lookahead, it's not a loop.
				# Now try to parse it as a general expression statement.
				self.lexer.pos = lexer_pos_backup
				self.lexer.current_char = lexer_current_char_backup
				self.current_token = parser_current_token_backup
				return self.expression_statement()  # This might still fail if it's truly malformed.
		
		# For IDENTIFIER, try specific assignment/input forms first, then general expression
		elif self.current_token.type == IDENTIFIER:
			# Try list assignment: IDENTIFIER LBRACKET ... <- ...
			lexer_pos_backup = self.lexer.pos
			lexer_current_char_backup = self.lexer.current_char
			parser_current_token_backup = self.current_token
			try:
				temp_var = self.variable()  # Consumes IDENTIFIER
				if self.current_token.type == LBRACKET:
					self.eat(LBRACKET)
					self.expression()  # Consume index expression
					self.eat(RBRACKET)
					if self.current_token.type == ASSIGN:
						self.lexer.pos = lexer_pos_backup
						self.lexer.current_char = lexer_current_char_backup
						self.current_token = parser_current_token_backup
						return self.list_assignment_statement()
				raise Exception("Not a list assignment pattern")  # Force backtrack if not list assign
			except Exception:
				self.lexer.pos = lexer_pos_backup
				self.lexer.current_char = lexer_current_char_backup
				self.current_token = parser_current_token_backup
			
			# Try input statement: IDENTIFIER (COMMA IDENTIFIER)* ASSIGN KEYWORD_INT
			lexer_pos_backup = self.lexer.pos
			lexer_current_char_backup = self.lexer.current_char
			parser_current_token_backup = self.current_token
			try:
				self.variable()	 # Consume first IDENTIFIER
				while self.current_token.type == COMMA:
					self.eat(COMMA)
					self.variable()
				if self.current_token.type == ASSIGN:
					self.eat(ASSIGN)
					if self.current_token.type == KEYWORD_INT:
						self.lexer.pos = lexer_pos_backup
						self.lexer.current_char = lexer_current_char_backup
						self.current_token = parser_current_token_backup
						return self.input_statement()
				raise Exception("Not an input statement pattern")  # Force backtrack
			except Exception:
				self.lexer.pos = lexer_pos_backup
				self.lexer.current_char = lexer_current_char_backup
				self.current_token = parser_current_token_backup
			
			# Try regular or augmented assignment: IDENTIFIER ASSIGN expression / IDENTIFIER OP_ASSIGN expression
			lexer_pos_backup = self.lexer.pos
			lexer_current_char_backup = self.lexer.current_char
			parser_current_token_backup = self.current_token
			try:
				self.variable()	 # Consume IDENTIFIER
				if self.current_token.type == ASSIGN:
					self.lexer.pos = lexer_pos_backup
					self.lexer.current_char = lexer_current_char_backup
					self.current_token = parser_current_token_backup
					return self.assignment_statement()
				elif self.current_token.type in (
				PLUS_ASSIGN, MINUS_ASSIGN, MULTIPLY_ASSIGN, DIVIDE_ASSIGN, EXPONENT_ASSIGN, FLOOR_DIVIDE_ASSIGN):
					self.lexer.pos = lexer_pos_backup
					self.lexer.current_char = lexer_current_char_backup
					self.current_token = parser_current_token_backup
					return self.augmented_assignment_statement()
				raise Exception("Not an assignment pattern")  # Force backtrack
			except Exception:
				self.lexer.pos = lexer_pos_backup
				self.lexer.current_char = lexer_current_char_backup
				self.current_token = parser_current_token_backup
			
			# If none of the specific identifier-starting statements matched, it must be an expression statement.
			return self.expression_statement()
		
		# If it's none of the above, it must be an expression statement (e.g., a literal, a function call)
		else:
			return self.expression_statement()
	
	def input_statement(self):
		"""
		Parses an input statement.
		input_statement ::= IDENTIFIER (COMMA IDENTIFIER)* ASSIGN KEYWORD_INT
		"""
		variables = [self.variable()]
		while self.current_token.type == COMMA:
			self.eat(COMMA)
			variables.append(self.variable())
		
		self.eat(ASSIGN)
		self.eat(KEYWORD_INT)
		return InputStatement(variables, Token(KEYWORD_INT, 'int'))
	
	def output_statement(self):
		"""
		Parses an output statement, now supporting chaining.
		output_statement ::= KEYWORD_OUT ASSIGN expression (ASSIGN expression)*
		"""
		self.eat(KEYWORD_OUT)
		expressions = []
		# The first 'ASSIGN' is mandatory after 'out'
		self.eat(ASSIGN)
		expressions.append(self.expression())
		
		# Subsequent 'ASSIGN' tokens allow chaining
		while self.current_token.type == ASSIGN:
			self.eat(ASSIGN)
			expressions.append(self.expression())
		
		return OutputStatement(expressions)
	
	def assignment_statement(self):
		"""
		Parses an assignment statement.
		assignment_statement ::= IDENTIFIER ASSIGN expression
		"""
		var_node = self.variable()
		self.eat(ASSIGN)
		expr_node = self.expression()
		return Assign(var_node, Token(ASSIGN, '<-'), expr_node)
	
	def list_assignment_statement(self):
		"""
		Parses a list element assignment statement.
		list_assignment_statement ::= IDENTIFIER LBRACKET expression RBRACKET ASSIGN expression
		"""
		list_var_node = self.variable()
		self.eat(LBRACKET)
		index_expr = self.expression()
		self.eat(RBRACKET)
		self.eat(ASSIGN)
		value_expr = self.expression()
		return ListAssign(list_var_node, index_expr, value_expr)
	
	def augmented_assignment_statement(self):
		"""
		Parses an augmented assignment statement.
		augmented_assignment_statement ::= IDENTIFIER (PLUS_ASSIGN | MINUS_ASSIGN | ...) expression
		"""
		var_node = self.variable()
		op_token = self.current_token
		self.eat(op_token.type)	 # Consume the augmented assignment operator
		expr_node = self.expression()
		return AugmentedAssign(var_node, op_token, expr_node)
	
	def for_statement(self):
		"""
		Parses a for loop statement.
		for_statement ::= LPAREN IDENTIFIER KEYWORD_FOR expression RANGE_OP expression (RANGE_OP expression)? RPAREN ARROW LBRACE (statement (SEMICOLON statement)*)* RBRACE
		"""
		self.eat(LPAREN)  # Consume '('
		loop_var = self.variable()  # Get the loop variable (e.g., 'x')
		self.eat(KEYWORD_FOR)  # Consume 'for'
		start_expr = self.expression()	# Get the start expression (e.g., '1')
		self.eat(RANGE_OP)  # Consume '..'
		end_expr = self.expression()  # Get the end expression (e.g., '5')
		step_expr = None
		if self.current_token.type == RANGE_OP:
			self.eat(RANGE_OP)
			step_expr = self.expression()
		self.eat(RPAREN)  # Consume ')'
		self.eat(ARROW)	 # Consume '->'
		self.eat(LBRACE)  # Consume '{'
		
		body_statements = []
		while self.current_token.type != RBRACE:
			body_statements.append(self.statement())
			# Allow multiple statements on one line within the block, separated by semicolons
			while self.current_token.type == SEMICOLON:
				self.eat(SEMICOLON)
				# Only parse another statement if it's not the end of the block or EOF
				if self.current_token.type != RBRACE and self.current_token.type != EOF:
					body_statements.append(self.statement())
		self.eat(RBRACE)  # Consume '}'
		return ForStatement(loop_var, start_expr, end_expr, body_statements, step_expr)
	
	def while_statement(self):
		"""
		Parses a while loop statement.
		while_statement ::= LPAREN IDENTIFIER KEYWORD_WHILE expression RPAREN ARROW LBRACE (statement (SEMICOLON statement)*)* RBRACE
		"""
		self.eat(LPAREN)  # Consume '('
		loop_var = self.variable()  # Consume the loop variable (e.g., 'x')
		self.eat(KEYWORD_WHILE)	 # Consume 'while'
		condition = self.expression()  # Get the condition expression (e.g., 'x < 5')
		self.eat(RPAREN)  # Consume ')'
		self.eat(ARROW)	 # Consume '->'
		self.eat(LBRACE)  # Consume '{'
		
		body_statements = []
		while self.current_token.type != RBRACE:
			body_statements.append(self.statement())
			while self.current_token.type == SEMICOLON:
				self.eat(SEMICOLON)
				if self.current_token.type != RBRACE and self.current_token.type != EOF:
					body_statements.append(self.statement())
		self.eat(RBRACE)  # Consume '}'
		return WhileStatement(loop_var, condition, body_statements)
	
	def expression_statement(self):	 # New: Parses an expression as a statement
		"""Parses an expression as a statement."""
		node = self.expression()
		return ExpressionStatement(node)
	
	# Expression parsing methods, ordered by precedence (lowest to highest)
	def expression(self):
		"""Handles 'or' operator."""
		node = self.logical_and_expression()
		while self.current_token.type == KEYWORD_OR:
			token = self.current_token
			self.eat(KEYWORD_OR)
			node = BinOp(left=node, op=token, right=self.logical_and_expression())
		return node
	
	def logical_and_expression(self):
		"""Handles 'and' operator."""
		node = self.comparison_expression()
		while self.current_token.type == KEYWORD_AND:
			token = self.current_token
			self.eat(KEYWORD_AND)
			node = BinOp(left=node, op=token, right=self.comparison_expression())
		return node
	
	def comparison_expression(self):
		"""Handles comparison operators (==, !=, <, >, <=, >=)."""
		node = self.additive_expression()
		while self.current_token.type in (EQ, NEQ, LT, GT, LTE, GTE):
			token = self.current_token
			self.eat(token.type)
			node = BinOp(left=node, op=token, right=self.additive_expression())
		return node
	
	def additive_expression(self):
		"""Handles '+' and '-' operators."""
		node = self.multiplicative_expression()
		while self.current_token.type in (PLUS, MINUS):
			token = self.current_token
			if token.type == PLUS:
				self.eat(PLUS)
			elif token.type == MINUS:
				self.eat(MINUS)
			node = BinOp(left=node, op=token, right=self.multiplicative_expression())
		return node
	
	def multiplicative_expression(self):
		"""Handles '*', '/', '//' operators."""
		node = self.exponentiation_expression()
		while self.current_token.type in (MULTIPLY, DIVIDE, FLOOR_DIVIDE):
			token = self.current_token
			if token.type == MULTIPLY:
				self.eat(MULTIPLY)
			elif token.type == DIVIDE:
				self.eat(DIVIDE)
			elif token.type == FLOOR_DIVIDE:
				self.eat(FLOOR_DIVIDE)
			node = BinOp(left=node, op=token, right=self.exponentiation_expression())
		return node
	
	def exponentiation_expression(self):
		"""Handles '**' operator (right-associative)."""
		node = self.unary_expression()
		if self.current_token.type == EXPONENT:
			token = self.current_token
			self.eat(EXPONENT)
			node = BinOp(left=node, op=token, right=self.exponentiation_expression())
		return node
	
	def unary_expression(self):
		"""Handles unary 'not' (and potential unary minus)."""
		if self.current_token.type == KEYWORD_NOT:
			token = self.current_token
			self.eat(KEYWORD_NOT)
			return UnaryOp(op=token, right=self.call_or_index_expression())
		# elif self.current_token.type == MINUS: # Uncomment for unary minus
		#     token = self.current_token
		#     self.eat(MINUS)
		#     return UnaryOp(op=token, right=self.call_or_index_expression())
		return self.call_or_index_expression()
	
	def call_or_index_expression(self):  # Handles both indexing and function calls as postfix
		"""
		Parses a primary expression, potentially followed by zero or more index accesses or function calls.
		call_or_index_expression ::= primary (LBRACKET expression RBRACKET | LPAREN (expression (COMMA expression)*)? RPAREN)*
		"""
		node = self.primary()  # Start with a primary (variable, literal, lambda, etc.)
		while self.current_token.type in (LBRACKET, LPAREN):
			if self.current_token.type == LBRACKET:	 # It's an index access
				self.eat(LBRACKET)
				index_expr = self.expression()
				self.eat(RBRACKET)
				node = IndexAccess(node, index_expr)
			elif self.current_token.type == LPAREN:	 # It's a function call
				self.eat(LPAREN)
				args = []
				if self.current_token.type != RPAREN:  # Check if there are arguments
					args.append(self.expression())
					while self.current_token.type == COMMA:
						self.eat(COMMA)
						args.append(self.expression())
				self.eat(RPAREN)
				node = Call(node, args)	 # The node here is the function being called (e.g., Var('square'))
		return node
	
	def primary(self):
		"""
		Parses the highest precedence elements: numbers, variables, strings, booleans, list literals, lambda expressions, or parenthesized expressions.
		primary ::= INTEGER | IDENTIFIER | STRING | BOOLEAN | ListLiteral | Lambda | LPAREN expression RPAREN
		"""
		token = self.current_token
		if token.type == INTEGER:
			self.eat(INTEGER)
			return Num(token)
		elif token.type == IDENTIFIER:
			self.eat(IDENTIFIER)
			return Var(token)
		elif token.type == STRING:
			self.eat(STRING)
			return String(token)
		elif token.type == KEYWORD_TRUE or token.type == KEYWORD_FALSE:
			self.eat(token.type)
			return Boolean(token)
		elif token.type == LBRACKET:
			return self.list_literal()
		elif self.current_token.type == KEYWORD_C or self.current_token.type == LPAREN:
			# Look ahead to distinguish lambda from simple parenthesized expression
			lexer_pos_backup = self.lexer.pos
			lexer_current_char_backup = self.lexer.current_char
			parser_current_token_backup = self.current_token
			is_curried = False
			
			if self.current_token.type == KEYWORD_C:
				is_curried = True
				self.eat(KEYWORD_C)  # Consume 'c'
			
			if self.current_token.type == LPAREN:
				self.eat(LPAREN)
				# Check for `IDENTIFIER` or `RPAREN` immediately after `LPAREN`
				# indicating start of parameters, then `ASSIGN` after `RPAREN`.
				# (param1, param2) <- body
				# () <- body
				if self.current_token.type == IDENTIFIER or self.current_token.type == RPAREN:
					# Parse params and check for ASSIGN token
					if self.current_token.type == IDENTIFIER:
						self.eat(IDENTIFIER)
						while self.current_token.type == COMMA:
							self.eat(COMMA)
							self.eat(IDENTIFIER)
					self.eat(RPAREN)
					if self.current_token.type == ASSIGN:
						# It's a lambda, reset and parse it
						self.lexer.pos = lexer_pos_backup
						self.lexer.current_char = lexer_current_char_backup
						self.current_token = parser_current_token_backup
						return self.lambda_expression()
					else:
						# Not a lambda, assume it's just a parenthesized expression that started with params
						raise Exception("Not a lambda definition")  # Force backtracking to regular expression
				else:
					raise Exception("Not a lambda definition")  # Force backtracking
			else:  # If it was 'c' but not '(', it's an error
				if is_curried:
					self.error("Expected '(' after 'c' for curried lambda.")
				self.error("Expected integer, identifier, string, boolean, list, lambda, or '('")
			
			# If we reach here, it means it was a regular parenthesized expression
			# This part is only reached if the try-except block above failed to identify a lambda.
			# So, we restore state and parse as a regular parenthesized expression.
			self.lexer.pos = lexer_pos_backup
			self.lexer.current_char = lexer_current_char_backup
			self.current_token = parser_current_token_backup
			
			self.eat(LPAREN)  # Consume LPAREN for the expression
			node = self.expression()
			self.eat(RPAREN)
			return node
		else:
			self.error("Expected integer, identifier, string, boolean, list, lambda, or '('")
	
	def list_literal(self):
		"""
		Parses a list literal.
		list_literal ::= LBRACKET (expression (COMMA expression)*)? RBRACKET
		"""
		self.eat(LBRACKET)
		elements = []
		if self.current_token.type != RBRACKET:
			elements.append(self.expression())
			while self.current_token.type == COMMA:
				self.eat(COMMA)
				elements.append(self.expression())
		self.eat(RBRACKET)
		return ListLiteral(elements)
	
	def lambda_expression(self):
		"""
		Parses a lambda expression.
		lambda_expression ::= (KEYWORD_C)? LPAREN (IDENTIFIER (COMMA IDENTIFIER)*)? RPAREN ASSIGN (expression | LBRACE (statement (SEMICOLON statement)*)* RBRACE)
		"""
		is_curried = False
		if self.current_token.type == KEYWORD_C:
			is_curried = True
			self.eat(KEYWORD_C)
		
		self.eat(LPAREN)
		params = []
		if self.current_token.type == IDENTIFIER:
			params.append(self.variable())
			while self.current_token.type == COMMA:
				self.eat(COMMA)
				params.append(self.variable())
		self.eat(RPAREN)
		self.eat(ASSIGN)  # The 'transfer' arrow for lambda body
		
		body = None
		if self.current_token.type == LBRACE:  # Check for block body
			self.eat(LBRACE)
			body_statements = []
			while self.current_token.type != RBRACE:
				body_statements.append(self.statement())
				while self.current_token.type == SEMICOLON:
					self.eat(SEMICOLON)
					if self.current_token.type != RBRACE and self.current_token.type != EOF:
						body_statements.append(self.statement())
			self.eat(RBRACE)
			body = Program(body_statements)	 # Body is a Program node
		else:
			body = self.expression()  # Body is a single expression
		
		return Lambda(params, body, is_curried)
	
	def variable(self):
		"""Parses a variable (identifier)."""
		token = self.current_token
		self.eat(IDENTIFIER)
		return Var(token)


# --- Interpreter ---
# Represents a callable function created from a Lambda AST node.
class Callable:
	def __init__(self, params, body, closure_env, is_curried=False, applied_args=None):
		self.params = params
		self.body = body
		self.closure_env = closure_env	# The scope where the lambda was defined
		self.is_curried = is_curried
		self.applied_args = applied_args if applied_args is not None else []
	
	def __str__(self):
		curry_str = " (curried)" if self.is_curried else ""
		applied_str = f" (applied: {len(self.applied_args)}/{len(self.params)})" if self.applied_args else ""
		return f"<lambda{curry_str} with params {', '.join(p.value for p in self.params)}{applied_str}>"
	
	__repr__ = __str__


# Traverses the AST and executes the program.
class Interpreter:
	def __init__(self, parser=None):
		self.parser = parser
		self.scopes = [{}]  # List of dictionaries, each dict is a scope. Global scope is the first.
	
	@property
	def current_scope(self):
		# Returns the innermost (current) scope
		return self.scopes[-1]
	
	def enter_scope(self):
		# Pushes a new empty scope onto the stack
		self.scopes.append({})
	
	def exit_scope(self):
		# Pops the current scope from the stack, but not the global scope
		if len(self.scopes) > 1:
			self.scopes.pop()
		else:
			raise Exception("Runtime error: Attempted to exit global scope.")
	
	def assign_variable(self, name, value):
		# Assigns to the nearest enclosing scope where variable is found,
		# otherwise creates it in the current (innermost) scope.
		for scope in reversed(self.scopes):
			if name in scope:
				scope[name] = value
				return
		self.current_scope[name] = value  # Create in current scope if not found
	
	def lookup_variable(self, name):
		# Looks up variable from current scope outwards to the global scope.
		for scope in reversed(self.scopes):
			if name in scope:
				return scope[name]
		raise Exception(f"Runtime error: Undefined variable '{name}'")
	
	def visit(self, node):
		"""
		Generic visitor method to dispatch to specific visit_ methods
		based on the AST node's type.
		"""
		method_name = f'visit_{type(node).__name__}'
		visitor = getattr(self, method_name, self.generic_visit)
		return visitor(node)
	
	def generic_visit(self, node):
		"""Fallback for unhandled AST node types."""
		raise Exception(f'No visit_{type(node).__name__} method defined for interpreter.')
	
	def visit_Program(self, node):
		"""Visit all statements in the program."""
		last_result = None
		for statement in node.statements:
			stmt_result = self.visit(statement)
			# For implicit return: if the last statement is an expression, its value is the return.
			# If it's an assignment or output, it might return None.
			last_result = stmt_result
		return last_result  # Return the result of the last statement/expression
	
	def visit_BinOp(self, node):
		"""Evaluate binary operations, including string concatenation and multiplication, and logical/comparison operators."""
		left_val = self.visit(node.left)
		right_val = self.visit(node.right)
		
		if node.op.type == PLUS:
			if isinstance(left_val, str) and isinstance(right_val, str):
				return left_val + right_val  # String concatenation
			elif isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
				return left_val + right_val  # Numeric addition
			else:
				raise Exception(
					f"Runtime error: Cannot perform '+' on types {type(left_val).__name__} and {type(right_val).__name__}")
		
		elif node.op.type == MULTIPLY:
			if isinstance(left_val, str) and isinstance(right_val, int):
				return left_val * right_val  # String repetition
			elif isinstance(left_val, int) and isinstance(right_val, str):
				return right_val * left_val  # String repetition (order doesn't matter)
			elif isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
				return left_val * right_val  # Numeric multiplication
			else:
				raise Exception(
					f"Runtime error: Cannot perform '*' on types {type(left_val).__name__} and {type(right_val).__name__}")
		
		# Logical AND and OR
		elif node.op.type == KEYWORD_AND:
			return bool(left_val) and bool(right_val)
		elif node.op.type == KEYWORD_OR:
			return bool(left_val) or bool(right_val)
		
		# Comparison Operators
		elif node.op.type == EQ:
			return left_val == right_val
		elif node.op.type == NEQ:
			return left_val != right_val
		elif node.op.type == LT:
			return left_val < right_val
		elif node.op.type == GT:
			return left_val > right_val
		elif node.op.type == LTE:
			return left_val <= right_val
		elif node.op.type == GTE:
			return left_val >= right_val
		
		# For other arithmetic operations, ensure both are numbers
		elif not isinstance(left_val, (int, float)) or not isinstance(right_val, (int, float)):
			raise Exception(
				f"Runtime error: Cannot perform arithmetic operation '{node.op.value}' on non-numeric types.")
		
		elif node.op.type == MINUS:
			return left_val - right_val
		elif node.op.type == DIVIDE:
			if right_val == 0:
				raise Exception("Runtime error: Division by zero")
			return left_val / right_val  # Standard float division
		elif node.op.type == FLOOR_DIVIDE:
			if right_val == 0:
				raise Exception("Runtime error: Division by zero")
			return left_val // right_val  # Integer floor division
		elif node.op.type == EXPONENT:
			return left_val ** right_val
	
	def visit_UnaryOp(self, node):
		"""Evaluate unary operations."""
		right_val = self.visit(node.right)
		if node.op.type == KEYWORD_NOT:
			return not bool(right_val)
		else:
			raise Exception(f"Runtime error: Unknown unary operator {node.op.value}")
	
	def visit_Num(self, node):
		"""Return the value of a number literal."""
		return node.value
	
	def visit_String(self, node):
		"""Return the value of a string literal."""
		return node.value
	
	def visit_Boolean(self, node):
		"""Return the value of a boolean literal."""
		return node.value
	
	def visit_Var(self, node):
		"""Retrieve the value of a variable from the current scope chain."""
		return self.lookup_variable(node.value)
	
	def visit_ListLiteral(self, node):
		"""Evaluate a list literal by evaluating its elements."""
		return [self.visit(element_expr) for element_expr in node.elements]
	
	def visit_IndexAccess(self, node):
		"""Access an element in a list using an index."""
		list_obj = self.visit(node.list_expr)
		index_val = self.visit(node.index_expr)
		
		if not isinstance(list_obj, list):
			raise Exception(f"Runtime error: Cannot index non-list type {type(list_obj).__name__}.")
		if not isinstance(index_val, int):
			raise Exception(f"Runtime error: List index must be an integer, got {type(index_val).__name__}.")
		
		try:
			return list_obj[index_val]
		except IndexError:
			raise Exception(f"Runtime error: List index {index_val} out of bounds for list of size {len(list_obj)}.")
	
	def visit_ListAssign(self, node):
		"""Assign a value to an element at a specific index in a list."""
		list_var_name = node.list_var.value
		# Lookup the list object using the scope chain
		list_obj = self.lookup_variable(list_var_name)
		
		if not isinstance(list_obj, list):
			raise Exception(f"Runtime error: Cannot assign to index of non-list variable '{list_var_name}'.")
		
		index_val = self.visit(node.index_expr)
		value_to_assign = self.visit(node.value_expr)
		
		if not isinstance(index_val, int):
			raise Exception(f"Runtime error: List index must be an integer, got {type(index_val).__name__}.")
		
		try:
			list_obj[index_val] = value_to_assign
		except IndexError:
			raise Exception(
				f"Runtime error: List index {index_val} out of bounds for list of size {len(list_obj)} during assignment.")
	
	def visit_Lambda(self, node):
		"""
		When a Lambda node is visited, create a Callable object.
		Crucially, capture the current scope for closure.
		"""
		# A copy of the current scope stack is taken to form the closure environment.
		# This allows the lambda to access variables from where it was defined.
		closure_env = list(self.scopes)	 # Create a copy of the scope stack
		return Callable(node.params, node.body, closure_env, is_curried=node.is_curried)
	
	def visit_Call(self, node):
		"""
		Execute a function call.
		"""
		func_obj = self.visit(node.func_expr)
		
		if not isinstance(func_obj, Callable):
			raise Exception(f"Runtime error: Cannot call non-callable object {func_obj}.")
		
		evaluated_new_args = [self.visit(arg_expr) for arg_expr in node.args]
		combined_args = func_obj.applied_args + evaluated_new_args
		
		if func_obj.is_curried and len(combined_args) < len(func_obj.params):
			# Return a new partially applied callable
			return Callable(func_obj.params, func_obj.body, func_obj.closure_env,
					is_curried=True, applied_args=combined_args)
		
		if len(combined_args) != len(func_obj.params):
			raise Exception(
				f"Runtime error: Expected {len(func_obj.params)} arguments, got {len(combined_args)} for lambda call.")
		
		# Save current scope stack
		original_scopes = list(self.scopes)
		
		# Set scopes to the closure environment + a new local scope for the function call
		self.scopes = list(func_obj.closure_env)  # Start with the closure env
		self.enter_scope()  # Add a new local scope for function parameters and local vars
		
		# Map arguments to parameters in the new function's local scope
		for param_node, arg_val in zip(func_obj.params, combined_args):
			self.current_scope[param_node.value] = arg_val
		
		result = None
		try:
			if isinstance(func_obj.body, Program):	# If body is a block of statements
				# Execute each statement in the block
				# The result of the last expression in the block is the return value
				for statement in func_obj.body.statements:
					result = self.visit(statement)	# Update result with each statement's return
			else:  # Body is a single expression
				result = self.visit(func_obj.body)
		finally:
			# Restore the original scope stack regardless of errors
			self.scopes = original_scopes
		
		return result
	
	def visit_Assign(self, node):
		"""Assign the result of an expression to a variable."""
		var_name = node.left.value
		self.assign_variable(var_name, self.visit(node.right))
	
	def visit_AugmentedAssign(self, node):
		"""Perform augmented assignment (e.g., a +<- 3)."""
		var_name = node.left.value
		current_val = self.lookup_variable(var_name)  # Use lookup_variable
		expr_val = self.visit(node.right)
		
		# Perform the operation based on the augmented assignment type
		new_val = None
		if node.op.type == PLUS_ASSIGN:
			new_val = current_val + expr_val
		elif node.op.type == MINUS_ASSIGN:
			new_val = current_val - expr_val
		elif node.op.type == MULTIPLY_ASSIGN:
			# Handle string repetition for augmented multiplication
			if isinstance(current_val, str) and isinstance(expr_val, int):
				new_val = current_val * expr_val
			elif isinstance(current_val, int) and isinstance(expr_val, str):
				new_val = current_val * expr_val
			elif isinstance(current_val, (int, float)) and isinstance(expr_val, (int, float)):
				new_val = current_val * expr_val
			else:
				raise Exception(
					f"Runtime error: Cannot perform '*<-' on types {type(current_val).__name__} and {type(expr_val).__name__}")
		elif node.op.type == DIVIDE_ASSIGN:
			if expr_val == 0: raise Exception("Runtime error: Division by zero")
			new_val = current_val / expr_val
		elif node.op.type == FLOOR_DIVIDE_ASSIGN:
			if expr_val == 0: raise Exception("Runtime error: Division by zero")
			new_val = current_val // expr_val
		elif node.op.type == EXPONENT_ASSIGN:
			new_val = current_val ** expr_val
		else:
			raise Exception(f"Runtime error: Unknown augmented assignment operator {node.op.value}")
		
		self.assign_variable(var_name, new_val)	 # Use assign_variable
	
	def visit_InputStatement(self, node):
		"""
		Handle input: read a line from stdin, parse comma-separated integers,
		and assign them to the specified variables.
		"""
		try:
			# Read a line from standard input
			input_line = input()
			# Split by comma and convert each part to an integer
			input_values = [int(val.strip()) for val in input_line.split(',')]
		except ValueError:
			raise Exception("Runtime error: Invalid input. Expected comma-separated integers.")
		except EOFError:
			raise Exception("Runtime error: Unexpected end of input during input statement.")
		
		if len(input_values) != len(node.variables):
			raise Exception(f"Runtime error: Expected {len(node.variables)} inputs, but received {len(input_values)}.")
		
		# Assign the parsed values to the variables using assign_variable
		for i, var_node in enumerate(node.variables):
			self.assign_variable(var_node.value, input_values[i])
	
	def visit_OutputStatement(self, node):
		"""
		Handle output: evaluate all expressions and print their values to stdout
		without newlines, supporting chaining.
		"""
		for expr in node.expressions:
			value = self.visit(expr)
			sys.stdout.write(str(value))
	
	def visit_ForStatement(self, node):
		"""
		Handle for loop execution.
		Loops from start_expr to end_expr (inclusive), assigning the current
		iteration value to the loop variable and executing the body statements.
		"""
		loop_var_name = node.variable.value
		start_val = self.visit(node.start_expr)
		end_val = self.visit(node.end_expr)
		step_val = self.visit(node.step_expr) if node.step_expr is not None else 1

		if not isinstance(start_val, int) or not isinstance(end_val, int) or not isinstance(step_val, int):
			raise Exception("Runtime error: For loop range must be integers.")

		if step_val == 0:
			raise Exception("Runtime error: For loop step cannot be 0.")

		# Enter a new scope for the loop body
		self.enter_scope()

		try:
			stop_val = end_val + (1 if step_val > 0 else -1)
			for i in range(start_val, stop_val, step_val):
				self.current_scope[loop_var_name] = i  # Loop var specific to this scope
				for statement in node.body:
					self.visit(statement)
		finally:
			self.exit_scope()  # Ensure scope is exited even if error occurs
	
	def visit_WhileStatement(self, node):
		"""
		Handle while loop execution.
		The loop continues as long as the condition evaluates to true.
		The loop variable (from `loop_var`) is assumed to be defined in an outer scope
		and modified within the body.
		"""
		# No new scope is entered for the while loop itself; its condition and body run
		# in the same scope as where the while loop statement itself is defined.
		# The variable used in the loop (e.g., 'x' in 'x while x < 5') is not local to the loop.
		
		while True:
			condition_result = self.visit(node.condition)
			
			if not isinstance(condition_result, bool):
				raise Exception("Runtime error: While loop condition must evaluate to a boolean.")
			
			if not condition_result:
				break  # Exit loop if condition is false
			
			# Execute loop body
			for statement in node.body:
				self.visit(statement)
	
	def visit_ExpressionStatement(self, node):  # New: Visit method for ExpressionStatement
		"""Evaluates the expression within an ExpressionStatement."""
		return self.visit(node.expression)
	
	def interpret(self):
		"""Start the interpretation process by parsing the program and visiting its AST."""
		if self.parser:
			tree = self.parser.program()
			self.visit(tree)  # Start traversing the AST from the root (Program node)
		else:
			raise Exception("Interpreter error: No parser set.")


# --- Main execution block (REPL) ---
if __name__ == '__main__':
	print("--- Mini Language REPL ---")
	print("Type 'quit' to exit.")
	print("Examples:")
	print("	 a <- 10; b <- 5; out <- a + b")
	print("	 a +<- 2; out <- a")
	print("	 out <- 'Hello' <- ' ' <- 'World!'")
	print("	 out <- 'abc' * 3")
	print("	 out <- 'Line 1\\nLine 2\\tTabbed' <- '\\n'")
	print("	 out <- true and false; out <- '\\n'")
	print("	 out <- not true; out <- '\\n'")
	print("	 (x for 1..3) -> { out <- x out <- ' ' }")
	print("	 x <- 0; (x while x < 5) -> { out <- x; x +<- 1; out <- '\\n' }")
	print("	 my_list <- [10, 'hello', true]; out <- my_list[1]; out <- '\\n'")
	print("	 my_list[0] <- 99; out <- my_list[0]; out <- '\\n'")
	print("	 my_list <- 50; out <- my_list; out <- '\\n'")	# Dynamic typing example
	print("	 square <- (x) <- x * x; out <- square(4); out <- '\\n'")
	print("	 add <- (a, b) <- a + b; out <- add(10, 20); out <- '\\n'")
	print("	 closure_example <- (y) <- (x) <- x + y; f <- closure_example(10); out <- f(5); out <- '\\n'")
	print("	 # New: Lambda with block body (implicit return)")
	print("	 adder_block <- (x) <- { y <- x + 1; y }; out <- adder_block(4); out <- '\\n'")
	print("	 # New: Currying example")
	print("	 curried_sum <- c(a, b) <- a + b; add_five <- curried_sum(5); out <- add_five(3); out <- '\\n'")
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
