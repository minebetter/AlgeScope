# factory.py
import random
from abc import ABC, abstractmethod
from typing import Optional, List
import algebrarium.core as core


class Leaf:
    """Represents a leaf node (a value) in an expression tree."""

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Leaf({self.value})"


class Node:
    """Represents an internal node (an operation) in an expression tree."""

    def __init__(self, action: core.Action, children: list):
        self.action = action
        self.children = children

    def __repr__(self):
        return f"Node({self.action.symbol}, {self.children})"


class QuestionGenerator(ABC):
    """Abstract base class for generating questions from an algebraic system."""

    def __init__(self,
                 algebra_sys: core.AlgebraicSystem,
                 max_visible_elements: int = 100):
        self._system = algebra_sys
        self._set = algebra_sys.set
        self._signature = algebra_sys.signature
        self._operations = [op for op in self._signature if op.arity > 0]

    @abstractmethod
    def generate(self, num_operate: int) -> dict:
        """Generates a question with a specific complexity (number of operations)."""
        pass


class ExpressionQuestionGenerator(QuestionGenerator):
    """
    Generates expression evaluation questions (e.g., 'a * b = ?').
    The complexity is determined by the number of operations in the expression.
    """

    def _distribute_ops(self, total_ops: int, num_children: int) -> List[int]:
        """
        Randomly distributes a budget of operations among children.
        Unlike leaves, a child can receive 0 operations (meaning it becomes a Leaf).
        """
        counts = [0] * num_children
        for _ in range(total_ops):
            child_index = random.randrange(num_children)
            counts[child_index] += 1
        return counts

    def _generate_tree(self, num_operate: int, leaves_collection: list):
        # vvv --- MODIFIED LOGIC --- vvv
        # Base case: If no operations budget is left, we must create a Leaf (value).
        if num_operate <= 0:
            leaf_value = self._set.sampler(k=1)[0]
            leaf = Leaf(leaf_value)
            leaves_collection.append(leaf)
            return leaf

        # Recursive step: Choose an operation.
        possible_actions = [
            op for op in self._operations
            # Note: We can generally use any operation if num_operate >= 1.
            # If strictly num_operate < op.arity is a concern for specific depth logic,
            # it is usually not an issue here because we distribute ops, not leaves.
        ]

        if not possible_actions:
            # Fallback if no operations exist, return leaf
            return self._generate_tree(0, leaves_collection)

        weights = [50 if op.arity >= 2 else 1 for op in possible_actions]
        action = random.choices(possible_actions, weights=weights, k=1)[0]

        # We consume 1 operation for this current Node.
        remaining_ops = num_operate - 1

        # Distribute the remaining operations among the children.
        child_op_counts = self._distribute_ops(remaining_ops, action.arity)

        children = [
            self._generate_tree(count, leaves_collection)
            for count in child_op_counts
        ]
        return Node(action, children)
        # ^^^ --- END OF MODIFIED LOGIC --- ^^^

    def _evaluate_tree(self, node):
        """Recursively evaluates the expression tree to find the result."""
        if isinstance(node, Leaf):
            return node.value

        child_values = [self._evaluate_tree(child) for child in node.children]
        result = node.action.apply(*child_values)

        # Runtime Closure Validation
        if not self._system.validate_element(result):
            raise ValueError(
                f"Closure Error: The result '{result}' of applying operation '{node.action.symbol}' "
                f"to {child_values} is not a valid element of the set.")
        return result

    def _tree_to_string(self,
                        node,
                        unknown_leaf: Optional[Leaf] = None) -> str:
        if isinstance(node, Leaf):
            value_str = str(node.value)
            display_val = f"'{value_str}'" if not value_str.isdigit() and not (
                value_str.startswith('-')
                and value_str[1:].isdigit()) else value_str
            return 'x' if node is unknown_leaf else display_val

        action = node.action
        if action.arity == 0:
            return f"'{action.apply()}'"

        child_strs = [
            self._tree_to_string(c, unknown_leaf) for c in node.children
        ]
        if action.arity == 2 and action.precedence > 0:
            return f"({child_strs[0]} {action.symbol} {child_strs[1]})"
        else:
            return f"{action.symbol}({', '.join(child_strs)})"

    def _cleanup_expression_string(self, expr_str: str) -> str:
        if expr_str.startswith('(') and expr_str.endswith(')'):
            return expr_str[1:-1]
        return expr_str

    def generate(self, num_operate: int) -> dict:
        # Changed validation: 0 operations is valid (just a constant),
        # but usually we want at least 1 for a question.
        if num_operate < 1:
            # Allow 0 for simple "eval this constant" but typically >= 1
            if num_operate < 0:
                raise ValueError("Number of operations must be non-negative.")

        if not self._operations and num_operate > 0:
            raise ValueError(
                "Cannot generate expressions: the algebraic system has no operations."
            )

        leaves = []
        tree = self._generate_tree(num_operate, leaves)
        question_str_raw = self._tree_to_string(tree)
        question_str = self._cleanup_expression_string(question_str_raw)
        answer_val = self._evaluate_tree(tree)

        answer_str = str(answer_val)
        if isinstance(answer_val, str) and not answer_str.isdigit():
            answer_str = f"'{answer_val}'"

        return {
            "question": f"{question_str} = ?",
            "answer": answer_str,
            "num_operate": num_operate,  # Key renamed
            "n_variables": 1,
            "category": "Expression"
        }


class EquationQuestionGenerator(ExpressionQuestionGenerator):
    """Generates equation solving questions (e.g., 'a * x = c', solve for x)."""

    def generate(self, num_operate: int) -> dict:
        if num_operate < 1:
            raise ValueError(
                "Number of operations must be at least 1 for a meaningful equation."
            )
        if not self._operations:
            raise ValueError(
                "Cannot generate equations: the algebraic system has no operations."
            )

        leaves = []
        full_tree = self._generate_tree(num_operate, leaves)

        # If the tree ended up having no leaves (e.g. 0-arity ops only), fallback
        if not leaves:
            return super().generate(num_operate)

        result = self._evaluate_tree(full_tree)
        unknown_leaf = random.choice(leaves)
        answer = unknown_leaf.value
        question_str_raw = self._tree_to_string(full_tree,
                                                unknown_leaf=unknown_leaf)
        question_str = self._cleanup_expression_string(question_str_raw)

        result_str = str(result)
        if isinstance(result, str) and not result_str.isdigit():
            result_str = f"'{result}'"

        answer_str = str(answer)
        if isinstance(answer, str) and not answer_str.isdigit():
            answer_str = f"'{answer}'"

        return {
            "question": f"{question_str} = {result_str}, x = ?",
            "answer": f"x = {answer_str}",
            "num_operate": num_operate,
            "n_variables": 1,
            "category": "Equation"
        }


class SystemOfGroupEquationsQuestionGenerator(QuestionGenerator):
    """
    Generates a solvable system of `n_variables` equations.
    """

    def __init__(self, algebra_sys, n_variables: int = 2, **kwargs):
        super().__init__(algebra_sys, **kwargs)
        if n_variables < 1:
            raise ValueError(
                "System of equations requires at least 1 variable.")
        self.n_variables = n_variables
        self._op, self._inv_op = self._find_group_operations()

    def _find_group_operations(self):
        binary_op = next((op for op in self._signature if op.arity == 2), None)
        inverse_op = next((op for op in self._signature if op.symbol == "inv"),
                          None)
        if not binary_op:
            raise ValueError(
                "Cannot generate group equations: No binary operation found.")
        if not inverse_op:
            raise ValueError(
                "Cannot generate group equations: No inverse operation with symbol 'inv' found."
            )
        return binary_op, inverse_op

    def _generate_variable_names(self, n: int) -> List[str]:
        common_names = [
            'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν',
            'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω'
        ]
        if n <= len(common_names):
            return common_names[:n]
        return [f'v_{i+1}' for i in range(n)]

    def generate(self, num_operate: int) -> dict:
        """
        Args:
            num_operate (int): The total number of operations in the generated equation.
                             In a binary structure, Operands = Operations + 1.
        """
        # Ensure we have enough operations to cover the structure of the variables.
        # For the i-th equation (0-indexed), we use i+1 variables.
        # To combine i+1 variables, we need at least i binary operations.
        # The largest equation has self.n_variables variables, needing self.n_variables - 1 ops.
        min_ops = self.n_variables - 1
        if num_operate < min_ops:
            # Adjust or warn. For simplicity, we ensure at least min_ops
            # but ideally the user should provide enough complexity.
            # We will proceed but the equation might look dense.
            pass

        ordered_vars = self._generate_variable_names(self.n_variables)
        solutions = self._set.sampler(k=self.n_variables)
        solution_map = {
            name: val
            for name, val in zip(ordered_vars, solutions)
        }

        question_equations = []
        shuffle_vars = ordered_vars[:]
        random.shuffle(shuffle_vars)

        for i in range(self.n_variables):
            core_variables = shuffle_vars[:i + 1]

            # vvv --- MODIFIED LOGIC --- vvv
            # Determine total operands required.
            # Binary tree property: Leaves (Operands) = Internal Nodes (Operations) + 1
            # If input num_operate is the target for the equation:
            target_total_operands = num_operate + 1

            if target_total_operands < len(core_variables):
                raise ValueError(
                    f"num_operate ({num_operate}) is too small to support {len(core_variables)} variables. "
                    f"Needs at least {len(core_variables) - 1} operations.")

            # Construct LHS operands
            lhs_operands = core_variables[:]

            # Fill the rest with constants
            num_constants = target_total_operands - len(lhs_operands)
            constants = self._set.sampler(k=num_constants)
            lhs_operands.extend(constants)
            # ^^^ --- END OF MODIFIED LOGIC --- ^^^

            random.shuffle(lhs_operands)

            if not lhs_operands: continue

            def _format_operand(op_val):
                op_str = str(op_val)
                if op_str in shuffle_vars:
                    return op_str
                return f"'{op_str}'"

            rhs_result = solution_map.get(lhs_operands[0], lhs_operands[0])
            lhs_str = _format_operand(lhs_operands[0])

            for op in lhs_operands[1:]:
                op_val = solution_map.get(op, op)
                rhs_result = self._op.apply(rhs_result, op_val)
                lhs_str = f"({lhs_str} {self._op.symbol} {_format_operand(op)})"

            result_str = f"'{str(rhs_result)}'"
            question_equations.append(f"{lhs_str} = {result_str}")

        random.shuffle(question_equations)
        full_question = "\n".join(question_equations)

        sorted_solutions = sorted(solution_map.items())
        answer_parts = [f"{name} = '{val}'" for name, val in sorted_solutions]
        full_answer = ", ".join(answer_parts)

        return {
            "question":
            f"Solve the following system for {', '.join(ordered_vars)}:\n{full_question}",
            "answer": full_answer,
            "num_operate": num_operate,
            "n_variables": self.n_variables,
            "category": "System of Group Equations"
        }


class MembershipQuestionGenerator(QuestionGenerator):
    """Generates questions asking if an element belongs to the set."""

    def _generate_non_element(self):
        element = self._set.sampler(k=1)[0]
        bad_char = random.choice(['$', '#', '!', '@', '&'])
        non_element_str = str(element) + bad_char

        try:
            if isinstance(element, int) or isinstance(float(element), float):
                if self._set.is_finite() and all(
                        isinstance(e, (int, float))
                        for e in self._set.elements):
                    max_val = max(self._set.elements)
                    return max_val + random.randint(1, 100)
                else:
                    return 999999999
        except (ValueError, TypeError, AttributeError):
            pass

        return non_element_str

    def generate(self, num_operate: int = 1) -> dict:
        # num_operate is ignored here, just kept for interface consistency
        is_member = random.choice([True, False])

        if is_member:
            element = self._set.sampler(k=1)[0]
            answer = "Yes"
        else:
            element = self._generate_non_element()
            while self._system.validate_element(element):
                element = self._generate_non_element()
            answer = "No"

        element_str = str(element)
        if isinstance(element, str) and not element_str.isdigit():
            element_str = f"'{element}'"

        set_name = self._set.name if self._set.name else "the set"

        return {
            "question": f"Is the element {element_str} in {set_name}?",
            "answer": answer,
            "num_operate":
            0,  # Membership check involves 0 algebraic operations
            "n_variables": 0,
            "category": "Membership"
        }


class StepByStepExpressionGenerator(ExpressionQuestionGenerator):
    """
    Generates expression questions where substeps use the explicit calculated values.
    """

    def _format_value(self, val) -> str:
        val_str = str(val)
        if isinstance(val, str) and not val_str.isdigit() and not (
                val_str.startswith('-') and val_str[1:].isdigit()):
            return f"'{val_str}'"
        return val_str

    def _evaluate_with_subquestions(self, node, steps: list):
        if isinstance(node, Leaf):
            val = node.value
            return val, self._format_value(val)

        child_results = []
        child_strs = []

        for child in node.children:
            val, val_str = self._evaluate_with_subquestions(child, steps)
            child_results.append(val)
            child_strs.append(val_str)

        try:
            result = node.action.apply(*child_results)
        except Exception as e:
            raise ValueError(f"Error calculating {node.action.symbol}: {e}")

        if not self._system.validate_element(result):
            raise ValueError(
                f"Closure Error: result '{result}' is not in set.")

        action = node.action
        if action.arity == 2 and action.precedence > 0:
            question_expr = f"{child_strs[0]} {action.symbol} {child_strs[1]}"
        elif action.arity == 1:
            question_expr = f"{action.symbol}({child_strs[0]})"
        else:
            question_expr = f"{action.symbol}({', '.join(child_strs)})"

        answer_str = self._format_value(result)

        sub_step = {
            "question": f"{question_expr} = ?",
            "answer": answer_str,
            "category": "Substep",
            # This substep itself represents 1 operation
            "num_operate": 1
        }
        steps.append(sub_step)

        return result, answer_str

    def generate(self, num_operate: int) -> dict:
        if num_operate < 1:
            raise ValueError("Number of operations must be >= 1")

        leaves = []
        tree = self._generate_tree(num_operate, leaves)

        question_str_raw = self._tree_to_string(tree)
        question_str = self._cleanup_expression_string(question_str_raw)

        substeps = []
        final_val, final_str = self._evaluate_with_subquestions(tree, substeps)

        return {
            "question": f"{question_str} = ?",
            "answer": final_str,
            "num_operate": num_operate,
            "n_variables": 1,
            "category": "Reasoning Expression",
            "substeps": substeps
        }
