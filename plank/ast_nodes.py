# --- AST (Abstract Syntax Tree) Nodes ---
# Classes representing different nodes in our program's syntax tree.
from dataclasses import dataclass

from plank.lexer import Token


class AST:
	"""Base class for all AST nodes."""
	pass


@dataclass(slots=True)
class Program(AST):
	statements: list  # List of statement nodes


@dataclass(slots=True)
class BinOp(AST):
	left: AST
	op: Token
	right: AST


@dataclass(slots=True)
class UnaryOp(AST):
	op: Token
	right: AST


@dataclass(slots=True)
class Num(AST):
	token: Token
	value: int | float | None = None
	
	def __post_init__(self):
		if self.value is None:
			self.value = self.token.value


@dataclass(slots=True)
class String(AST):
	token: Token
	value: str | None = None
	
	def __post_init__(self):
		if self.value is None:
			self.value = self.token.value


@dataclass(slots=True)
class Boolean(AST):
	token: Token
	value: bool | None = None
	
	def __post_init__(self):
		if self.value is None:
			self.value = self.token.value


@dataclass(slots=True)
class Var(AST):
	token: Token
	value: str | None = None
	
	def __post_init__(self):
		if self.value is None:
			self.value = self.token.value


@dataclass(slots=True)
class ListLiteral(AST):
        elements: list


@dataclass(slots=True)
class DictLiteral(AST):
        pairs: list  # list of (key, value) tuples


@dataclass(slots=True)
class SetLiteral(AST):
        elements: list


@dataclass(slots=True)
class IndexAccess(AST):
	list_expr: AST
	index_expr: AST


@dataclass(slots=True)
class ListAssign(AST):
	list_var: Var
	index_expr: AST
	value_expr: AST


@dataclass(slots=True)
class Lambda(AST):
        params: list
        body: AST
        is_curried: bool = False


@dataclass(slots=True)
class FunctionDef(AST):
        name: Var
        params: list
        body: Program


@dataclass(slots=True)
class Call(AST):
	func_expr: AST
	args: list


@dataclass(slots=True)
class Assign(AST):
	left: Var
	op: Token
	right: AST


@dataclass(slots=True)
class AugmentedAssign(AST):
	left: Var
	op: Token
	right: AST


@dataclass(slots=True)
class InputStatement(AST):
	variables: list
	type_token: Token


@dataclass(slots=True)
class OutputStatement(AST):
        expressions: list


@dataclass(slots=True)
class ImportStatement(AST):
        path_token: Token


@dataclass(slots=True)
class ForStatement(AST):
	variable: Var
	start_expr: AST
	end_expr: AST
	body: list
	step_expr: AST | None = None


@dataclass(slots=True)
class WhileStatement(AST):
	loop_var: Var
	condition: AST
	body: list


@dataclass(slots=True)
class IfExpression(AST):
	condition: AST
	then_branch: AST
	else_branch: AST | None = None


@dataclass(slots=True)
class ExpressionStatement(AST):  # Expression used as a statement
        expression: AST


@dataclass(slots=True)
class BreakStatement(AST):
        pass


@dataclass(slots=True)
class ContinueStatement(AST):
        pass


@dataclass(slots=True)
class Return(AST):
        value: AST | None = None


@dataclass(slots=True)
class TypeCast(AST):
        expr: AST
        type_token: Token
