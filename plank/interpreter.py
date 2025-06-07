import sys
import math
from collections.abc import Iterable, Sequence, MutableSequence, Mapping, MutableMapping, MutableSet

from plank.ast_nodes import *
from plank.token_types import *
from plank.token_types import TokenType


class BreakSignal(Exception):
    pass


class ContinueSignal(Exception):
    pass


class ReturnSignal(Exception):
    def __init__(self, value):
        self.value = value


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
        self._setup_builtins()

    def _call_function(self, func_obj, args):
        """Utility to call a function or builtin with given args."""
        if callable(func_obj) and not isinstance(func_obj, Callable):
            return func_obj(*args)
        if not isinstance(func_obj, Callable):
            raise Exception(f"Runtime error: Cannot call non-callable object {func_obj}.")

        combined_args = func_obj.applied_args + args
        if func_obj.is_curried and len(combined_args) < len(func_obj.params):
            return Callable(func_obj.params, func_obj.body, func_obj.closure_env,
                            is_curried=True, applied_args=combined_args)
        if len(combined_args) != len(func_obj.params):
            raise Exception(
                f"Runtime error: Expected {len(func_obj.params)} arguments, got {len(combined_args)} for lambda call.")

        original_scopes = list(self.scopes)
        self.scopes = list(func_obj.closure_env)
        self.enter_scope()
        for param_node, arg_val in zip(func_obj.params, combined_args):
            self.current_scope[param_node.value] = arg_val
        result = None
        try:
            if isinstance(func_obj.body, Program):
                try:
                    for statement in func_obj.body.statements:
                        result = self.visit(statement)
                except ReturnSignal as rs:
                    result = rs.value
            else:
                try:
                    result = self.visit(func_obj.body)
                except ReturnSignal as rs:
                    result = rs.value
        finally:
            self.scopes = original_scopes
        return result

    def _setup_builtins(self):
        def b_len(x):
            return len(x)

        def b_head(seq):
            if not isinstance(seq, Sequence):
                raise Exception("Runtime error: head expects a sequence")
            if isinstance(seq, (str, bytes)):
                if not seq:
                    raise Exception("Runtime error: head of empty sequence")
                return seq[0]
            if not seq:
                raise Exception("Runtime error: head of empty sequence")
            return seq[0]

        def b_tail(seq):
            if not isinstance(seq, Sequence):
                raise Exception("Runtime error: tail expects a sequence")
            return seq[1:]

        def b_abs(x):
            if not isinstance(x, (int, float)):
                raise Exception("Runtime error: abs expects a number")
            return abs(x)

        def b_min(a, b):
            return a if a < b else b

        def b_max(a, b):
            return a if a > b else b

        def b_clamp(x, lo, hi):
            if not all(isinstance(v, (int, float)) for v in (x, lo, hi)):
                raise Exception("Runtime error: clamp expects numeric arguments")
            return max(lo, min(x, hi))

        def b_push(container, *items):
            if isinstance(container, MutableSequence):
                if len(items) != 1:
                    raise Exception("Runtime error: push on sequence takes one item")
                container.append(items[0])
                return container
            if isinstance(container, MutableSet):
                if len(items) != 1:
                    raise Exception("Runtime error: push on set takes one item")
                container.add(items[0])
                return container
            if isinstance(container, MutableMapping):
                if len(items) != 2:
                    raise Exception("Runtime error: push on dict requires key and value")
                key, value = items
                container[key] = value
                return container
            raise Exception("Runtime error: push expects a mutable sequence, set, or dict")

        def b_pop(container, key=None):
            if isinstance(container, MutableSequence):
                if key is not None:
                    raise Exception("Runtime error: pop on sequence takes no key")
                if not container:
                    raise Exception("Runtime error: pop from empty sequence")
                return container.pop()
            if isinstance(container, MutableSet):
                if key is not None:
                    raise Exception("Runtime error: pop on set takes no key")
                if not container:
                    raise Exception("Runtime error: pop from empty set")
                return container.pop()
            if isinstance(container, MutableMapping):
                if key is None:
                    raise Exception("Runtime error: pop for dict requires a key")
                try:
                    return container.pop(key)
                except KeyError:
                    raise Exception(f"Runtime error: Key {key!r} not found in dictionary.")
            raise Exception("Runtime error: pop expects a mutable sequence, set, or dict")

        def b_contains(container, item):
            return item in container

        def b_remove(container, key_or_item):
            if isinstance(container, MutableSequence):
                if not isinstance(key_or_item, int):
                    raise Exception("Runtime error: remove index must be integer")
                try:
                    del container[key_or_item]
                    return container
                except IndexError:
                    raise Exception("Runtime error: remove index out of range")
            if isinstance(container, MutableSet):
                try:
                    container.remove(key_or_item)
                except KeyError:
                    raise Exception(f"Runtime error: item {key_or_item!r} not found in set")
                return container
            if isinstance(container, MutableMapping):
                try:
                    del container[key_or_item]
                    return container
                except KeyError:
                    raise Exception(f"Runtime error: Key {key_or_item!r} not found in dictionary.")
            raise Exception("Runtime error: remove expects a mutable sequence, set, or dict")

        def b_insert(seq, index, value):
            if not isinstance(seq, MutableSequence):
                raise Exception("Runtime error: insert expects a mutable sequence")
            if not isinstance(index, int):
                raise Exception("Runtime error: insert index must be integer")
            seq.insert(index, value)
            return seq

        def b_extend(container, iterable_or_mapping):
            if isinstance(container, MutableSequence):
                if not isinstance(iterable_or_mapping, Iterable):
                    raise Exception("Runtime error: extend expects an iterable for sequence")
                container.extend(iterable_or_mapping)
                return container
            if isinstance(container, MutableSet):
                if not isinstance(iterable_or_mapping, Iterable):
                    raise Exception("Runtime error: extend expects an iterable for set")
                container.update(iterable_or_mapping)
                return container
            if isinstance(container, MutableMapping):
                if not isinstance(iterable_or_mapping, Mapping):
                    raise Exception("Runtime error: extend expects a mapping for dict")
                container.update(iterable_or_mapping)
                return container
            raise Exception("Runtime error: extend expects a mutable sequence, set, or dict")

        def b_keys(mapping):
            if not isinstance(mapping, Mapping):
                raise Exception("Runtime error: keys expects a mapping")
            return list(mapping.keys())

        def b_values(mapping):
            if not isinstance(mapping, Mapping):
                raise Exception("Runtime error: values expects a mapping")
            return list(mapping.values())

        def b_items(mapping):
            if not isinstance(mapping, Mapping):
                raise Exception("Runtime error: items expects a mapping")
            return [[k, v] for k, v in mapping.items()]

        def b_union(set_a, set_b):
            if not isinstance(set_a, MutableSet) or not isinstance(set_b, MutableSet):
                raise Exception("Runtime error: union expects two sets")
            return set_a | set_b

        def b_intersection(set_a, set_b):
            if not isinstance(set_a, MutableSet) or not isinstance(set_b, MutableSet):
                raise Exception("Runtime error: intersection expects two sets")
            return set_a & set_b

        def b_difference(set_a, set_b):
            if not isinstance(set_a, MutableSet) or not isinstance(set_b, MutableSet):
                raise Exception("Runtime error: difference expects two sets")
            return set_a - set_b

        def b_map(func, iterable):
            if not isinstance(iterable, Iterable):
                raise Exception("Runtime error: map expects an iterable")
            return [self._call_function(func, [x]) for x in iterable]

        def b_filter(func, iterable):
            if not isinstance(iterable, Iterable):
                raise Exception("Runtime error: filter expects an iterable")
            return [x for x in iterable if self._call_function(func, [x])]

        def b_foldl(func, init, iterable):
            if not isinstance(iterable, Iterable):
                raise Exception("Runtime error: foldl expects an iterable")
            acc = init
            for x in iterable:
                acc = self._call_function(func, [acc, x])
            return acc

        def b_foldr(func, init, iterable):
            if not isinstance(iterable, Iterable):
                raise Exception("Runtime error: foldr expects an iterable")
            acc = init
            seq = list(iterable)
            for x in reversed(seq):
                acc = self._call_function(func, [x, acc])
            return acc

        def b_sort(iterable):
            if not isinstance(iterable, Iterable):
                raise Exception("Runtime error: sort expects an iterable")
            try:
                return sorted(iterable)
            except TypeError:
                raise Exception("Runtime error: elements not comparable for sort")

        def b_split(string, delim):
            if not isinstance(string, str) or not isinstance(delim, str):
                raise Exception("Runtime error: split expects a string and delimiter")
            return string.split(delim)

        def b_join(delim, lst):
            if not isinstance(lst, list) or not all(isinstance(s, str) for s in lst):
                raise Exception("Runtime error: join expects a list of strings")
            if not isinstance(delim, str):
                raise Exception("Runtime error: join expects a string delimiter")
            return delim.join(lst)

        def b_find(string, sub):
            if not isinstance(string, str) or not isinstance(sub, str):
                raise Exception("Runtime error: find expects two strings")
            return string.find(sub)

        def b_replace(string, old, new):
            if not isinstance(string, str) or not isinstance(old, str) or not isinstance(new, str):
                raise Exception("Runtime error: replace expects strings")
            return string.replace(old, new)


        def b_zip(a, b):
            if not isinstance(a, Iterable) or not isinstance(b, Iterable):
                raise Exception("Runtime error: zip expects two iterables")
            return [[x, y] for x, y in zip(a, b)]

        def b_enumerate(iterable):
            if not isinstance(iterable, Iterable):
                raise Exception("Runtime error: enumerate expects an iterable")
            return [[i, v] for i, v in enumerate(iterable)]

        def b_round(x):
            if not isinstance(x, (int, float)):
                raise Exception("Runtime error: round expects a number")
            return round(x)

        def b_floor(x):
            if not isinstance(x, (int, float)):
                raise Exception("Runtime error: floor expects a number")
            return math.floor(x)

        def b_ceil(x):
            if not isinstance(x, (int, float)):
                raise Exception("Runtime error: ceil expects a number")
            return math.ceil(x)

        def b_sum(iterable):
            if not isinstance(iterable, Iterable):
                raise Exception("Runtime error: sum expects an iterable of numbers")
            seq = list(iterable)
            if not all(isinstance(i, (int, float)) for i in seq):
                raise Exception("Runtime error: sum expects a list of numbers")
            return sum(seq)

        def b_average(iterable):
            if not isinstance(iterable, Iterable):
                raise Exception("Runtime error: average expects a non-empty iterable of numbers")
            seq = list(iterable)
            if not seq or not all(isinstance(i, (int, float)) for i in seq):
                raise Exception("Runtime error: average expects a non-empty iterable of numbers")
            return sum(seq) / len(seq)

        def b_range(start, end, step=None):
            if step is None:
                step = 1 if start <= end else -1
            if not all(isinstance(v, int) for v in (start, end, step)):
                raise Exception("Runtime error: range expects integer arguments")
            stop = end + (1 if step > 0 else -1)
            return list(range(start, stop, step))

        builtins = {
            'len': b_len,
            'head': b_head,
            'tail': b_tail,
            'abs': b_abs,
            'min': b_min,
            'max': b_max,
            'clamp': b_clamp,
            'push': b_push,
            'pop': b_pop,
            'contains': b_contains,
            'remove': b_remove,
            'insert': b_insert,
            'extend': b_extend,
            'keys': b_keys,
            'values': b_values,
            'items': b_items,
            'union': b_union,
            'intersection': b_intersection,
            'difference': b_difference,
            'map': b_map,
            'filter': b_filter,
            'foldl': b_foldl,
            'foldr': b_foldr,
            'sort': b_sort,
            'split': b_split,
            'join': b_join,
            'find': b_find,
            'replace': b_replace,
            'zip': b_zip,
            'enumerate': b_enumerate,
            'round': b_round,
            'floor': b_floor,
            'ceil': b_ceil,
            'sum': b_sum,
            'average': b_average,
            'range': b_range,
        }
        self.scopes[0].update(builtins)
    
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
        """Evaluate binary operations and logical/comparison operators."""
        left_val = self.visit(node.left)
        right_val = self.visit(node.right)
        
        match node.op.type:
            case TokenType.PLUS:
                if isinstance(left_val, str) and isinstance(right_val, str):
                    return left_val + right_val
                if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
                    return left_val + right_val
                raise Exception(
                    f"Runtime error: Cannot perform '+' on types {type(left_val).__name__} and {type(right_val).__name__}")
            
            case TokenType.MULTIPLY:
                if isinstance(left_val, str) and isinstance(right_val, int):
                    return left_val * right_val
                if isinstance(left_val, int) and isinstance(right_val, str):
                    return right_val * left_val
                if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
                    return left_val * right_val
                raise Exception(
                    f"Runtime error: Cannot perform '*' on types {type(left_val).__name__} and {type(right_val).__name__}")
            
            case TokenType.KEYWORD_AND:
                return bool(left_val) and bool(right_val)
            case TokenType.KEYWORD_OR:
                return bool(left_val) or bool(right_val)
            
            case TokenType.EQ:
                return left_val == right_val
            case TokenType.NEQ:
                return left_val != right_val
            case TokenType.LT:
                return left_val < right_val
            case TokenType.GT:
                return left_val > right_val
            case TokenType.LTE:
                return left_val <= right_val
            case TokenType.GTE:
                return left_val >= right_val
            
            case TokenType.MINUS:
                if not isinstance(left_val, (int, float)) or not isinstance(right_val, (int, float)):
                    raise Exception(
                        f"Runtime error: Cannot perform arithmetic operation '-' on non-numeric types.")
                return left_val - right_val
            case TokenType.DIVIDE:
                if right_val == 0:
                    raise Exception("Runtime error: Division by zero")
                return left_val / right_val
            case TokenType.MODULUS:
                if right_val == 0:
                    raise Exception("Runtime error: Division by zero")
                return left_val % right_val
            case TokenType.FLOOR_DIVIDE:
                if right_val == 0:
                    raise Exception("Runtime error: Division by zero")
                return left_val // right_val
            case TokenType.EXPONENT:
                return left_val ** right_val
            case TokenType.COLON:
                if isinstance(left_val, list) and isinstance(right_val, list):
                    return left_val + right_val
                if isinstance(left_val, list):
                    new_list = list(left_val)
                    new_list.append(right_val)
                    return new_list
                if isinstance(right_val, list):
                    return [left_val] + right_val
                raise Exception("Runtime error: ':' operator requires at least one list operand")
            case _:
                raise Exception(f"Runtime error: Unknown binary operator {node.op.value}")
    
    def visit_UnaryOp(self, node):
        """Evaluate unary operations."""
        right_val = self.visit(node.right)
        if node.op.type == KEYWORD_NOT:
            return not bool(right_val)
        elif node.op.type == MINUS:
            if not isinstance(right_val, (int, float)):
                raise Exception("Runtime error: Unary minus requires a numeric operand.")
            return -right_val
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

    def visit_DictLiteral(self, node):
        """Evaluate a dictionary literal."""
        return {self.visit(k): self.visit(v) for k, v in node.pairs}

    def visit_SetLiteral(self, node):
        """Evaluate a set literal."""
        return {self.visit(e) for e in node.elements}
    
    def visit_IndexAccess(self, node):
        """Access an element in a list or dictionary using an index/key."""
        container = self.visit(node.list_expr)
        key = self.visit(node.index_expr)

        if isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
            if not isinstance(key, int):
                raise Exception(
                    f"Runtime error: Sequence index must be an integer, got {type(key).__name__}."
                )
            try:
                return container[key]
            except IndexError:
                raise Exception(
                    f"Runtime error: Index {key} out of bounds for sequence of size {len(container)}."
                )
        elif isinstance(container, Mapping):
            try:
                return container[key]
            except KeyError:
                raise Exception(f"Runtime error: Key {key!r} not found in dictionary.")
        else:
            raise Exception(
                f"Runtime error: Cannot index non-sequence type {type(container).__name__}."
            )
    
    def visit_ListAssign(self, node):
        """Assign a value to an element at a specific index in a list."""
        list_var_name = node.list_var.value
        # Lookup the list object using the scope chain
        list_obj = self.lookup_variable(list_var_name)
        
        if not isinstance(list_obj, MutableSequence):
            raise Exception(
                f"Runtime error: Cannot assign to index of non-sequence variable '{list_var_name}'."
            )
        
        index_val = self.visit(node.index_expr)
        value_to_assign = self.visit(node.value_expr)
        
        if not isinstance(index_val, int):
            raise Exception(
                f"Runtime error: Sequence index must be an integer, got {type(index_val).__name__}."
            )
        
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
        evaluated_new_args = [self.visit(arg_expr) for arg_expr in node.args]
        return self._call_function(func_obj, evaluated_new_args)
    
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
        elif node.op.type == MODULUS_ASSIGN:
            if expr_val == 0: raise Exception("Runtime error: Division by zero")
            new_val = current_val % expr_val
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
        Handle input for any supported type. Values are whitespace separated
        similar to C++'s cin operator.
        """
        try:
            input_line = input()
            if node.type_token.value == 'list' and len(node.variables) == 1:
                parts = [input_line.strip()]
            else:
                parts = input_line.strip().split()
        except EOFError:
            raise Exception("Runtime error: Unexpected end of input during input statement.")

        if len(parts) != len(node.variables):
            raise Exception(
                f"Runtime error: Expected {len(node.variables)} inputs, but received {len(parts)}.")

        def convert(val: str):
            t = node.type_token.value
            if t == 'int':
                return int(val)
            if t == 'string':
                return val
            if t == 'bool':
                if val.lower() in ('true', '1', 't'):
                    return True
                if val.lower() in ('false', '0', 'f'):
                    return False
                raise ValueError
            if t in ('list', 'dict', 'set'):
                import ast
                return ast.literal_eval(val)
            return val

        try:
            input_values = [convert(p) for p in parts]
        except (ValueError, SyntaxError):
            raise Exception(f"Runtime error: Invalid input for type {node.type_token.value}.")

        for i, var_node in enumerate(node.variables):
            self.assign_variable(var_node.value, input_values[i])

    def visit_TypeCast(self, node):
        value = self.visit(node.expr)
        t = node.type_token.value
        try:
            if t == 'int':
                return int(value)
            if t == 'string':
                return str(value)
            if t == 'bool':
                return bool(value)
            if t == 'list':
                return list(value)
            if t == 'dict':
                return dict(value)
            if t == 'set':
                return set(value)
        except Exception:
            raise Exception(f"Runtime error: Cannot convert to {t}.")
        raise Exception(f"Runtime error: Unknown type {t} in cast")
    
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
        step_val = self.visit(node.step_expr) if node.step_expr is not None else (1 if start_val <= end_val else -1)
        
        if not isinstance(start_val, int) or not isinstance(end_val, int) or not isinstance(step_val, int):
            raise Exception("Runtime error: For loop range must be integers.")
        
        # Enter a new scope for the loop body
        self.enter_scope()
        
        try:
            stop = end_val + (1 if step_val > 0 else -1)
            for i in range(start_val, stop, step_val):
                self.current_scope[loop_var_name] = i  # Loop var specific to this scope
                try:
                    for statement in node.body:
                        self.visit(statement)
                except ContinueSignal:
                    continue
                except BreakSignal:
                    break
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
            continue_outer = False
            try:
                for statement in node.body:
                    self.visit(statement)
            except ContinueSignal:
                continue_outer = True
            except BreakSignal:
                break

            if continue_outer:
                continue
    
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

    def visit_BreakStatement(self, node):
        raise BreakSignal()

    def visit_ContinueStatement(self, node):
        raise ContinueSignal()

    def visit_Return(self, node):
        value = self.visit(node.value) if node.value is not None else None
        raise ReturnSignal(value)
    
    def interpret(self):
        """Start the interpretation process by parsing the program and visiting its AST."""
        if self.parser:
            tree = self.parser.program()
            self.visit(tree)  # Start traversing the AST from the root (Program node)
        else:
            raise Exception("Interpreter error: No parser set.")
