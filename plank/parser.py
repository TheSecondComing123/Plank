from plank.ast_nodes import *
from plank.lexer import Token
from plank.token_types import *


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
        elif self.current_token.type == KEYWORD_BREAK:
            self.eat(KEYWORD_BREAK)
            return BreakStatement()
        elif self.current_token.type == KEYWORD_CONTINUE:
            self.eat(KEYWORD_CONTINUE)
            return ContinueStatement()
        elif self.current_token.type == KEYWORD_RETURN:
            return self.return_statement()
        elif self.current_token.type == KEYWORD_FN:
            return self.function_definition()
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
                self.variable()  # Consume first IDENTIFIER
                while self.current_token.type == COMMA:
                    self.eat(COMMA)
                    self.variable()
                if self.current_token.type == ASSIGN:
                    self.eat(ASSIGN)
                    if self.current_token.type in (KEYWORD_INT, KEYWORD_STRING, KEYWORD_BOOL, KEYWORD_LIST):
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
                self.variable()  # Consume IDENTIFIER
                if self.current_token.type == ASSIGN:
                    self.lexer.pos = lexer_pos_backup
                    self.lexer.current_char = lexer_current_char_backup
                    self.current_token = parser_current_token_backup
                    return self.assignment_statement()
                elif self.current_token.type in (
                        PLUS_ASSIGN, MINUS_ASSIGN, MULTIPLY_ASSIGN, DIVIDE_ASSIGN, MODULUS_ASSIGN,
                        EXPONENT_ASSIGN, FLOOR_DIVIDE_ASSIGN):
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
        input_statement ::= IDENTIFIER (COMMA IDENTIFIER)* ASSIGN TYPE
        where TYPE is one of int, string, bool, list
        """
        variables = [self.variable()]
        while self.current_token.type == COMMA:
            self.eat(COMMA)
            variables.append(self.variable())
        
        self.eat(ASSIGN)
        if self.current_token.type not in (KEYWORD_INT, KEYWORD_STRING, KEYWORD_BOOL, KEYWORD_LIST):
            self.error("Expected type after '<-'")
        type_token = self.current_token
        self.eat(type_token.type)
        return InputStatement(variables, Token(type_token.type, type_token.value))
    
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
        self.eat(op_token.type)  # Consume the augmented assignment operator
        expr_node = self.expression()
        return AugmentedAssign(var_node, op_token, expr_node)

    def return_statement(self):
        """Parses a return statement with an optional expression."""
        self.eat(KEYWORD_RETURN)
        expr = None
        if self.current_token.type not in (SEMICOLON, RBRACE, EOF):
            expr = self.expression()
        return Return(expr)

    def function_definition(self):
        """Parses a function definition."""
        self.eat(KEYWORD_FN)
        name = self.variable()
        self.eat(LPAREN)
        params = []
        if self.current_token.type == IDENTIFIER:
            params.append(self.variable())
            while self.current_token.type == COMMA:
                self.eat(COMMA)
                params.append(self.variable())
        self.eat(RPAREN)
        self.eat(ARROW)
        self.eat(LBRACE)
        body_statements = []
        while self.current_token.type != RBRACE:
            body_statements.append(self.statement())
            while self.current_token.type == SEMICOLON:
                self.eat(SEMICOLON)
                if self.current_token.type != RBRACE and self.current_token.type != EOF:
                    body_statements.append(self.statement())
        self.eat(RBRACE)
        return FunctionDef(name, params, Program(body_statements))
    
    def for_statement(self):
        """
        Parses a for loop statement.
        for_statement ::= LPAREN IDENTIFIER KEYWORD_FOR expression RANGE_OP expression (RANGE_OP expression)? RPAREN ARROW LBRACE (statement (SEMICOLON statement)*)* RBRACE
        """
        self.eat(LPAREN)  # Consume '('
        loop_var = self.variable()  # Get the loop variable (e.g., 'x')
        self.eat(KEYWORD_FOR)  # Consume 'for'
        start_expr = self.expression()  # Get the start expression (e.g., '1')
        self.eat(RANGE_OP)  # Consume '..'
        end_expr = self.expression()  # Get the end expression (e.g., '5')
        step_expr = None
        if self.current_token.type == RANGE_OP:
            self.eat(RANGE_OP)
            step_expr = self.expression()
        self.eat(RPAREN)  # Consume ')'
        self.eat(ARROW)  # Consume '->'
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
        self.eat(KEYWORD_WHILE)  # Consume 'while'
        condition = self.expression()  # Get the condition expression (e.g., 'x < 5')
        self.eat(RPAREN)  # Consume ')'
        self.eat(ARROW)  # Consume '->'
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
    
    def expression_statement(self):  # New: Parses an expression as a statement
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
        while self.current_token.type in (PLUS, MINUS, COLON):
            token = self.current_token
            if token.type == PLUS:
                self.eat(PLUS)
            elif token.type == MINUS:
                self.eat(MINUS)
            elif token.type == COLON:
                self.eat(COLON)
            node = BinOp(left=node, op=token, right=self.multiplicative_expression())
        return node
    
    def multiplicative_expression(self):
        """Handles '*', '/', '%', '//' operators."""
        node = self.exponentiation_expression()
        while self.current_token.type in (MULTIPLY, DIVIDE, MODULUS, FLOOR_DIVIDE):
            token = self.current_token
            if token.type == MULTIPLY:
                self.eat(MULTIPLY)
            elif token.type == DIVIDE:
                self.eat(DIVIDE)
            elif token.type == MODULUS:
                self.eat(MODULUS)
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
            return UnaryOp(op=token, right=self.cast_expression())
        elif self.current_token.type == MINUS:
            token = self.current_token
            self.eat(MINUS)
            return UnaryOp(op=token, right=self.cast_expression())
        return self.cast_expression()

    def cast_expression(self):
        node = self.call_or_index_expression()
        while self.current_token.type == ARROW:
            # Peek ahead to ensure this is a type cast and not part of another construct
            saved_pos = self.lexer.pos
            saved_char = self.lexer.current_char
            saved_token = self.current_token
            next_token = self.lexer.get_next_token()
            self.lexer.pos = saved_pos
            self.lexer.current_char = saved_char
            self.current_token = saved_token

            if next_token.type not in (KEYWORD_INT, KEYWORD_STRING, KEYWORD_BOOL, KEYWORD_LIST):
                break

            self.eat(ARROW)
            type_token = self.current_token
            self.eat(type_token.type)
            node = TypeCast(node, type_token)
        return node
    
    def call_or_index_expression(self):  # Handles both indexing and function calls as postfix
        """
        Parses a primary expression, potentially followed by zero or more index accesses or function calls.
        call_or_index_expression ::= primary (LBRACKET expression RBRACKET | LPAREN (expression (COMMA expression)*)? RPAREN)*
        """
        node = self.primary()  # Start with a primary (variable, literal, lambda, etc.)
        while self.current_token.type in (LBRACKET, LPAREN):
            if self.current_token.type == LBRACKET:  # It's an index access
                self.eat(LBRACKET)
                index_expr = self.expression()
                self.eat(RBRACKET)
                node = IndexAccess(node, index_expr)
            elif self.current_token.type == LPAREN:  # It's a function call
                self.eat(LPAREN)
                args = []
                if self.current_token.type != RPAREN:  # Check if there are arguments
                    args.append(self.expression())
                    while self.current_token.type == COMMA:
                        self.eat(COMMA)
                        args.append(self.expression())
                self.eat(RPAREN)
                node = Call(node, args)  # The node here is the function being called (e.g., Var('square'))
        return node
    
    def primary(self):
        """
        Parses the highest precedence elements: numbers, variables, strings, booleans, list literals, lambda expressions, or parenthesized expressions.
        primary ::= INTEGER | FLOAT | IDENTIFIER | STRING | BOOLEAN | ListLiteral | Lambda | LPAREN expression RPAREN
        """
        token = self.current_token
        if token.type == KEYWORD_IF:
            return self.if_expression()
        if token.type in (KEYWORD_LEN, KEYWORD_HEAD, KEYWORD_TAIL, KEYWORD_ABS, KEYWORD_MIN, KEYWORD_MAX, KEYWORD_CLAMP,
                           KEYWORD_PUSH, KEYWORD_POP, KEYWORD_MAP, KEYWORD_FILTER, KEYWORD_FOLDL, KEYWORD_FOLDR,
                           KEYWORD_SORT, KEYWORD_SPLIT, KEYWORD_JOIN, KEYWORD_FIND, KEYWORD_REPLACE,
                           KEYWORD_ZIP, KEYWORD_ENUMERATE):
            return self.builtin_call()
        if token.type == INTEGER:
            self.eat(INTEGER)
            return Num(token)
        if token.type == FLOAT:
            self.eat(FLOAT)
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
            lexer_pos_backup = self.lexer.pos
            lexer_current_char_backup = self.lexer.current_char
            parser_current_token_backup = self.current_token
            is_curried = False

            if self.current_token.type == KEYWORD_C:
                is_curried = True
                self.eat(KEYWORD_C)

            if self.current_token.type == LPAREN:
                self.eat(LPAREN)
                if self.current_token.type == IDENTIFIER or self.current_token.type == RPAREN:
                    if self.current_token.type == IDENTIFIER:
                        self.eat(IDENTIFIER)
                        while self.current_token.type == COMMA:
                            self.eat(COMMA)
                            self.eat(IDENTIFIER)
                    self.eat(RPAREN)
                    if self.current_token.type == ASSIGN:
                        self.lexer.pos = lexer_pos_backup
                        self.lexer.current_char = lexer_current_char_backup
                        self.current_token = parser_current_token_backup
                        return self.lambda_expression()
                # restore and treat as parenthesized expression
                self.lexer.pos = lexer_pos_backup
                self.lexer.current_char = lexer_current_char_backup
                self.current_token = parser_current_token_backup
                self.eat(LPAREN)
                node = self.expression()
                self.eat(RPAREN)
                return node
            else:
                if is_curried:
                    self.error("Expected '(' after 'c' for curried lambda.")
                self.error("Expected integer, identifier, string, boolean, list, lambda, or '('")
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

    def builtin_call(self):
        token = self.current_token
        self.eat(token.type)

        args = []
        if self.current_token.type == LPAREN:
            self.eat(LPAREN)
            if self.current_token.type != RPAREN:
                args.append(self.expression())
                while self.current_token.type == COMMA:
                    self.eat(COMMA)
                    args.append(self.expression())
            self.eat(RPAREN)
        else:
            args.append(self.expression())
            while self.current_token.type == COMMA:
                self.eat(COMMA)
                args.append(self.expression())

        func_token = Token(IDENTIFIER, token.value)
        return Call(Var(func_token), args)
    
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
            body = Program(body_statements)  # Body is a Program node
        else:
            body = self.expression()  # Body is a single expression
        
        return Lambda(params, body, is_curried)
    
    def if_expression(self):
        """Parses an if expression with optional else/elif branches."""
        self.eat(KEYWORD_IF)
        condition = self.expression()
        self.eat(ARROW)
        if self.current_token.type == LBRACE:
            self.eat(LBRACE)
            body_statements = []
            while self.current_token.type != RBRACE:
                body_statements.append(self.statement())
                while self.current_token.type == SEMICOLON:
                    self.eat(SEMICOLON)
                    if self.current_token.type != RBRACE and self.current_token.type != EOF:
                        body_statements.append(self.statement())
            self.eat(RBRACE)
            then_branch = Program(body_statements)
        else:
            then_branch = self.expression()
        
        else_branch = None
        if self.current_token.type == KEYWORD_ELSE:
            self.eat(KEYWORD_ELSE)
            if self.current_token.type == KEYWORD_IF:
                else_branch = self.if_expression()
            else:
                if self.current_token.type == LBRACE:
                    self.eat(LBRACE)
                    body_statements = []
                    while self.current_token.type != RBRACE:
                        body_statements.append(self.statement())
                        while self.current_token.type == SEMICOLON:
                            self.eat(SEMICOLON)
                            if self.current_token.type != RBRACE and self.current_token.type != EOF:
                                body_statements.append(self.statement())
                    self.eat(RBRACE)
                    else_branch = Program(body_statements)
                else:
                    else_branch = self.expression()
        
        return IfExpression(condition, then_branch, else_branch)
    
    def variable(self):
        """Parses a variable (identifier)."""
        token = self.current_token
        self.eat(IDENTIFIER)
        return Var(token)
