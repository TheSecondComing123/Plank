# --- Token Types ---
# Define constants for all token types our language will recognize.
EOF = 'EOF'  # End of File
IDENTIFIER = 'IDENTIFIER'  # Variable names (e.g., 'a', 'b', 'result')
INTEGER = 'INTEGER'  # Integer literals (e.g., '10', '42')
STRING = 'STRING'  # String literals (e.g., '"hello"', '" "')

# Arithmetic Operators
PLUS = 'PLUS'  # '+'
MINUS = 'MINUS'  # '-'
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
ARROW = 'ARROW'  # '->'

# Augmented Assignment Operators
PLUS_ASSIGN = 'PLUS_ASSIGN'  # '+<-'
MINUS_ASSIGN = 'MINUS_ASSIGN'  # '-<-'
MULTIPLY_ASSIGN = 'MULTIPLY_ASSIGN'  # '*<-'
DIVIDE_ASSIGN = 'DIVIDE_ASSIGN'  # '/<-'
EXPONENT_ASSIGN = 'EXPONENT_ASSIGN'  # '**<-'
FLOOR_DIVIDE_ASSIGN = 'FLOOR_DIVIDE_ASSIGN'  # '//<-'

# Boolean Literals and Logical Operators
KEYWORD_TRUE = 'KEYWORD_TRUE'  # 'true'
KEYWORD_FALSE = 'KEYWORD_FALSE'  # 'false'
KEYWORD_AND = 'KEYWORD_AND'  # 'and'
KEYWORD_OR = 'KEYWORD_OR'  # 'or'
KEYWORD_NOT = 'KEYWORD_NOT'  # 'not'
KEYWORD_WHILE = 'KEYWORD_WHILE'  # 'while'
KEYWORD_IF = 'KEYWORD_IF'  # 'if'
KEYWORD_ELSE = 'KEYWORD_ELSE'  # 'else'
KEYWORD_C = 'KEYWORD_C'  # 'c' for curried functions

# Comparison Operators
EQ = 'EQ'  # '=='
NEQ = 'NEQ'  # '!='
LT = 'LT'  # '<'
GT = 'GT'  # '>'
LTE = 'LTE'  # '<='
GTE = 'GTE'  # '>='

# Punctuation
COMMA = 'COMMA'  # ','
LPAREN = 'LPAREN'  # '('
RPAREN = 'RPAREN'  # ')'
LBRACE = 'LBRACE'  # '{'
RBRACE = 'RBRACE'  # '}'
SEMICOLON = 'SEMICOLON'  # ';'

# List Punctuation
LBRACKET = 'LBRACKET'  # '['
RBRACKET = 'RBRACKET'  # ']'
