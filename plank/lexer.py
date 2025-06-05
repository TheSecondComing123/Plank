from plank.token_types import *


# --- Token Class ---
# Represents a single token found by the lexer.
from dataclasses import dataclass


@dataclass(slots=True)
class Token:
        type: TokenType
        value: object

        def __repr__(self) -> str:  # pragma: no cover - trivial
                return f"Token({self.type}, {self.value!r})"


# --- Lexer (Tokenizer) ---
# Reads the input text and converts it into a stream of tokens.
class Lexer:
        KEYWORDS = {
                'int': (KEYWORD_INT, 'int'),
                'string': (KEYWORD_STRING, 'string'),
                'bool': (KEYWORD_BOOL, 'bool'),
                'list': (KEYWORD_LIST, 'list'),
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

        MULTI_OPS = {
                '**<-': EXPONENT_ASSIGN,
                '*<-': MULTIPLY_ASSIGN,
                '**': EXPONENT,
                '//<-': FLOOR_DIVIDE_ASSIGN,
                '/<-': DIVIDE_ASSIGN,
                '//': FLOOR_DIVIDE,
                '<-': ASSIGN,
                '->': ARROW,
                '-<-': MINUS_ASSIGN,
                '+<-': PLUS_ASSIGN,
                '<=': LTE,
                '>=': GTE,
                '==': EQ,
                '!=': NEQ,
                '..': RANGE_OP,
        }

        SINGLE_OPS = {
                '*': MULTIPLY,
                '/': DIVIDE,
                '<': LT,
                '>': GT,
                '-': MINUS,
                '+': PLUS,
                ',': COMMA,
                '(': LPAREN,
                ')': RPAREN,
                '{': LBRACE,
                '}': RBRACE,
                ';': SEMICOLON,
                '[': LBRACKET,
                ']': RBRACKET,
        }

        def __init__(self, text):
                self.text = text
                self.pos = 0
                self.current_char = self.text[self.pos] if self.text else None
        
        def advance(self):
                """Move to the next character in the input text."""
                self.pos += 1
                if self.pos < len(self.text):
                        self.current_char = self.text[self.pos]
                else:
                        self.current_char = None  # Indicates end of input
        
        def match(self, string):
                """Return True and consume the string if it is next in the input."""
                if self.text.startswith(string, self.pos):
                        self.pos += len(string)
                        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None
                        return True
                return False
        
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
                quote_char = self.current_char  # Capture the opening quote character
                self.advance()  # Consume the opening quote
                result = ''
                while self.current_char is not None and self.current_char != quote_char:
                        if self.current_char == '\\':  # Handle escape sequences
                                self.advance()  # Consume the backslash
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
                self.advance()  # Consume the closing quote
                return result
        
        def _id(self):
                """Parse an identifier or keyword."""
                result = ''
                while self.current_char and (self.current_char.isalnum() or self.current_char == '_'):
                        result += self.current_char
                        self.advance()
                return Token(*self.KEYWORDS.get(result, (IDENTIFIER, result)))
        

        def get_next_token(self):
                """Return the next token from the input text."""
                while self.current_char is not None:
                        self.skip_whitespace()
                        if self.current_char is None:
                                break

                        if self.current_char.isdigit():
                                return Token(INTEGER, self.integer())

                        if self.current_char in ('"', "'"):
                                return Token(STRING, self.string())

                        if self.current_char.isalpha() or self.current_char == '_':
                                return self._id()

                        for op, ttype in self.MULTI_OPS.items():
                                if self.match(op):
                                        return Token(ttype, op)

                        if self.current_char == '.':
                                raise Exception(
                                        f'Lexer error: Invalid character sequence starting with . at position {self.pos}'
                                )

                        token_type = self.SINGLE_OPS.get(self.current_char)
                        if token_type:
                                ch = self.current_char
                                self.advance()
                                return Token(token_type, ch)

                        raise Exception(
                                f'Lexer error: Invalid character {self.current_char!r} at position {self.pos}'
                        )

                return Token(EOF, None)
