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
		self.expressions = expressions  # List of expressions to be printed


class ForStatement(AST):
	def __init__(self, variable, start_expr, end_expr, body):
		self.variable = variable  # The loop variable (Var node)
		self.start_expr = start_expr  # Expression for the start of the range
		self.end_expr = end_expr  # Expression for the end of the range
		self.body = body  # List of statements in the loop body


class WhileStatement(AST):
	def __init__(self, loop_var, condition, body):
		self.loop_var = loop_var  # The variable for the loop (e.g., 'x')
		self.condition = condition  # The expression representing the loop condition (e.g., 'x < 5')
		self.body = body  # List of statements in the loop body


class ExpressionStatement(AST):  # New AST node for expressions treated as statements
	def __init__(self, expression):
		self.expression = expression
