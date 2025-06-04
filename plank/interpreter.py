import sys

from plank.token_types import *
from plank.ast_nodes import *


# --- Interpreter ---
# Represents a callable function created from a Lambda AST node.
class Callable:
	def __init__(self, params, body, closure_env, is_curried=False, applied_args=None):
		self.params = params
		self.body = body
		self.closure_env = closure_env  # The scope where the lambda was defined
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
				f"Runtime error: List index {index_val} out of bounds for "
				f"list of size {len(list_obj)} during assignment."
			)
	
	def visit_Lambda(self, node):
		"""
		When a Lambda node is visited, create a Callable object.
		Crucially, capture the current scope for closure.
		"""
		# A copy of the current scope stack is taken to form the closure environment.
		# This allows the lambda to access variables from where it was defined.
		closure_env = list(self.scopes)  # Create a copy of the scope stack
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
			if isinstance(func_obj.body, Program):  # If body is a block of statements
				# Execute each statement in the block
				# The result of the last expression in the block is the return value
				for statement in func_obj.body.statements:
					result = self.visit(statement)  # Update result with each statement's return
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
		try:
			current_val = self.lookup_variable(var_name)  # Use lookup_variable
		except Exception:
			current_val = 0
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
		
		self.assign_variable(var_name, new_val)  # Use assign_variable
	
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
		
		if not isinstance(start_val, int) or not isinstance(end_val, int):
			raise Exception("Runtime error: For loop range must be integers.")
		
		# Enter a new scope for the loop body
		self.enter_scope()
		
		try:
			for i in range(start_val, end_val + 1):
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
	
	def visit_IfExpression(self, node):
		condition_val = self.visit(node.condition)
		if condition_val:
			return self.visit(node.then_branch)
		elif node.else_branch is not None:
			return self.visit(node.else_branch)
		else:
			return None

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
