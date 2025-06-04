"""Enumeration of all token types used by the language."""

from enum import Enum, auto


class TokenType(Enum):
	"""Enumeration of all token types."""
	
	EOF = auto()  # End of File
	IDENTIFIER = auto()  # Variable names (e.g., 'a', 'b', 'result')
	INTEGER = auto()  # Integer literals (e.g., '10', '42')
	STRING = auto()  # String literals (e.g., '"hello"', '" "')
	
	# Arithmetic Operators
	PLUS = auto()  # '+'
	MINUS = auto()  # '-'
	MULTIPLY = auto()  # '*'
	DIVIDE = auto()  # '/'
	EXPONENT = auto()  # '**'
	FLOOR_DIVIDE = auto()  # '//'
	
	# Assignment and Keywords
	ASSIGN = auto()  # '<-'
	KEYWORD_INT = auto()  # 'int' (for input type)
	KEYWORD_OUT = auto()  # 'out' (for output statement)
	
	# For Loop Keywords and Operators
	KEYWORD_FOR = auto()  # 'for'
	RANGE_OP = auto()  # '..'
	ARROW = auto()  # '->'
	
	# Augmented Assignment Operators
	PLUS_ASSIGN = auto()  # '+<-'
	MINUS_ASSIGN = auto()  # '-<-'
	MULTIPLY_ASSIGN = auto()  # '*<-'
	DIVIDE_ASSIGN = auto()  # '/<-'
	EXPONENT_ASSIGN = auto()  # '**<-'
	FLOOR_DIVIDE_ASSIGN = auto()  # '//<-'
	
	# Boolean Literals and Logical Operators
	KEYWORD_TRUE = auto()  # 'true'
	KEYWORD_FALSE = auto()  # 'false'
	KEYWORD_AND = auto()  # 'and'
	KEYWORD_OR = auto()  # 'or'
	KEYWORD_NOT = auto()  # 'not'
	KEYWORD_WHILE = auto()  # 'while'
	KEYWORD_IF = auto()  # 'if'
	KEYWORD_ELSE = auto()  # 'else'
	KEYWORD_C = auto()  # 'c' for curried functions
	
	# Comparison Operators
	EQ = auto()  # '=='
	NEQ = auto()  # '!='
	LT = auto()  # '<'
	GT = auto()  # '>'
	LTE = auto()  # '<='
	GTE = auto()  # '>='
	
	# Punctuation
	COMMA = auto()  # ','
	LPAREN = auto()  # '('
	RPAREN = auto()  # ')'
	LBRACE = auto()  # '{'
	RBRACE = auto()  # '}'
	SEMICOLON = auto()  # ';'
	
	# List Punctuation
	LBRACKET = auto()  # '['
	RBRACKET = auto()  # ']'


# Export individual names for backwards compatibility
EOF = TokenType.EOF
IDENTIFIER = TokenType.IDENTIFIER
INTEGER = TokenType.INTEGER
STRING = TokenType.STRING
PLUS = TokenType.PLUS
MINUS = TokenType.MINUS
MULTIPLY = TokenType.MULTIPLY
DIVIDE = TokenType.DIVIDE
EXPONENT = TokenType.EXPONENT
FLOOR_DIVIDE = TokenType.FLOOR_DIVIDE
ASSIGN = TokenType.ASSIGN
KEYWORD_INT = TokenType.KEYWORD_INT
KEYWORD_OUT = TokenType.KEYWORD_OUT
KEYWORD_FOR = TokenType.KEYWORD_FOR
RANGE_OP = TokenType.RANGE_OP
ARROW = TokenType.ARROW
PLUS_ASSIGN = TokenType.PLUS_ASSIGN
MINUS_ASSIGN = TokenType.MINUS_ASSIGN
MULTIPLY_ASSIGN = TokenType.MULTIPLY_ASSIGN
DIVIDE_ASSIGN = TokenType.DIVIDE_ASSIGN
EXPONENT_ASSIGN = TokenType.EXPONENT_ASSIGN
FLOOR_DIVIDE_ASSIGN = TokenType.FLOOR_DIVIDE_ASSIGN
KEYWORD_TRUE = TokenType.KEYWORD_TRUE
KEYWORD_FALSE = TokenType.KEYWORD_FALSE
KEYWORD_AND = TokenType.KEYWORD_AND
KEYWORD_OR = TokenType.KEYWORD_OR
KEYWORD_NOT = TokenType.KEYWORD_NOT
KEYWORD_WHILE = TokenType.KEYWORD_WHILE
KEYWORD_IF = TokenType.KEYWORD_IF
KEYWORD_ELSE = TokenType.KEYWORD_ELSE
KEYWORD_C = TokenType.KEYWORD_C
EQ = TokenType.EQ
NEQ = TokenType.NEQ
LT = TokenType.LT
GT = TokenType.GT
LTE = TokenType.LTE
GTE = TokenType.GTE
COMMA = TokenType.COMMA
LPAREN = TokenType.LPAREN
RPAREN = TokenType.RPAREN
LBRACE = TokenType.LBRACE
RBRACE = TokenType.RBRACE
SEMICOLON = TokenType.SEMICOLON
LBRACKET = TokenType.LBRACKET
RBRACKET = TokenType.RBRACKET

__all__ = [name for name in globals() if name.isupper()]
