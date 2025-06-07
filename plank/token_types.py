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
    KEYWORD_INT = auto()  # 'int' (type)
    KEYWORD_STRING = auto()  # 'string' (type)
    KEYWORD_BOOL = auto()  # 'bool' (type)
    KEYWORD_LIST = auto()  # 'list' (type)
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
    KEYWORD_LEN = auto()  # 'len'
    KEYWORD_HEAD = auto()  # 'head'
    KEYWORD_TAIL = auto()  # 'tail'
    KEYWORD_ABS = auto()  # 'abs'
    KEYWORD_MIN = auto()  # 'min'
    KEYWORD_MAX = auto()  # 'max'
    KEYWORD_CLAMP = auto()  # 'clamp'
    KEYWORD_PUSH = auto()  # 'push'
    KEYWORD_POP = auto()  # 'pop'
    KEYWORD_MAP = auto()  # 'map'
    KEYWORD_FILTER = auto()  # 'filter'
    KEYWORD_FOLDL = auto()  # 'foldl'
    KEYWORD_FOLDR = auto()  # 'foldr'
    KEYWORD_SORT = auto()  # 'sort'
    KEYWORD_SPLIT = auto()  # 'split'
    KEYWORD_JOIN = auto()  # 'join'
    KEYWORD_FIND = auto()  # 'find'
    KEYWORD_REPLACE = auto()  # 'replace'
    KEYWORD_ZIP = auto()  # 'zip'
    KEYWORD_ENUMERATE = auto()  # 'enumerate'
    
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

    COLON = auto()  # ':'
    
    # List Punctuation
    LBRACKET = auto()  # '['
    RBRACKET = auto()  # ']'


# Export individual names for backwards compatibility
for _token in TokenType:
    globals()[_token.name] = _token

__all__ = ['TokenType', *TokenType.__members__]
