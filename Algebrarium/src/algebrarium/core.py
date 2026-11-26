# core.py

from abc import ABC, abstractmethod
from typing import Callable, Any, List, Dict


class Set:
    """
    A container for the carrier set's properties, defined by rules rather than a fixed list.
    It holds a validator to check for membership and a sampler to generate elements.
    """

    def __init__(self, definition: str, validator: Callable[[str], bool],
                 sampler: Callable[[int], List[str]]):
        self.definition_str = definition
        self.validator = validator  # Rule to check if a symbol is in the set
        self.sampler = sampler  # Rule to generate sample elements from the set

    @property
    def elements(self) -> List[Any]:
        """
        Warning: This property is for backward compatibility or for systems
        that are known to be finite and small. It samples a default number of elements.
        For infinite sets, this will only be a small subset.
        """
        print(
            "Warning: Accessing .elements on a rule-based Set. Sampling 10 elements."
        )
        return self.sampler(k=10)


class Action:
    """
    A purely symbolic operation defined by a set of computational rules.
    It operates directly on the elements of the carrier set (strings).
    """

    def __init__(self, symbol: str, arity: int, name: str, precedence: int,
                 calculation_logic: Callable[..., Any]):
        self.symbol = symbol
        self.arity = arity
        self.name = name
        self.precedence = precedence
        # The logic is a pure algorithm operating on symbols
        self._logic = calculation_logic

    def apply(self, *args: Any) -> Any:
        """Applies the action's algorithmic logic directly to the given symbolic arguments."""
        if len(args) != self.arity:
            raise ValueError(
                f"Action '{self.symbol}' expects {self.arity} arguments, but got {len(args)}."
            )
        return self._logic(*args)

    def definition(self) -> str:
        """Generates the definition string for this purely symbolic operation."""
        arity_map = {
            0: "Constant",
            1: "Unary Operation",
            2: "Binary Operation"
        }
        op_type = arity_map.get(self.arity, f"{self.arity}-ary Operation")
        return f"'{self.symbol}' ({op_type}): An operation named '{self.name}'."


class AlgebraicSystem:
    """
    A self-contained, purely formal algebraic system defined by a set of symbols and rules.
    The principle is "Rules as Truth": the system is nothing more or less than
    what is described by its set and operations.
    """

    def __init__(self, name: str, description: str, rules_template: str,
                 set_obj: Set, signature: List[Action],
                 format_context: Dict[str, Any], raw_config: Dict[str, Any],
                 identity_map: Dict[Action,
                                    Action], inverse_map: Dict[Action,
                                                               Action]):
        self.name = name
        self._description_template = description
        self._rules_template = rules_template
        self._set = set_obj
        self._signature = sorted(signature,
                                 key=lambda a: a.precedence,
                                 reverse=True)
        self._format_context = format_context
        self._raw_config = raw_config
        self.actions: Dict[str, Action] = {
            action.symbol: action
            for action in signature
        }
        self.identity_map = identity_map
        self.inverse_map = inverse_map

    @property
    def set(self) -> Set:
        return self._set

    def validate_element(self, element: Any) -> bool:
        """Checks if a given symbolic element belongs to the carrier set using its validator rule."""
        # Ensure we are always validating the string representation of an element
        return self._set.validator(str(element))

    @property
    def signature(self) -> List[Action]:
        return self._signature

    def _format_string(self, text: str) -> str:
        return text.format_map(self._format_context)

    @property
    def description(self) -> str:
        return self._format_string(self._description_template)

    def get_full_definition(self) -> str:
        """Presents the system's definition in its pure, formal structure."""
        parts = [
            f"--- Full Algebraic System Definition ---",
            f"System: {self.description}",
            f"1. Carrier Set (S): {self.set.definition_str}",
            "2. Signature (Operations):"
        ]
        for action in self.signature:
            parts.append(f"   - {action.definition()}")
        parts.append("----------------------------------------")
        return "\n".join(parts)

    def get_computation_rules(self) -> str:
        """Dynamically generates rules based on their high-level descriptions."""
        # This method is now simpler, as it relies on the descriptive text from the template.
        # The complex table generation is kept for finite systems that still define it.
        rules_parts = [self._format_string(self._rules_template)]

        # Add semantic explanations (Identity, Inverse) - This logic remains valuable
        if self.identity_map:
            rules_parts.append("\n--- Identity Element Properties ---")
            for op, identity in self.identity_map.items():
                rules_parts.append(
                    f"- The constant '{identity.symbol}' is the identity for the '{op.symbol}' operation."
                )
        if self.inverse_map:
            rules_parts.append("\n--- Inverse Properties ---")
            for op, inv_op in self.inverse_map.items():
                identity = self.identity_map.get(op)
                if identity:
                    rules_parts.append(
                        f"- The function '{inv_op.symbol}(a)' finds the inverse for 'a' regarding the '{op.symbol}' operation."
                    )

        # Display computation tables ONLY if they are explicitly defined in the raw config.
        # This handles backward compatibility for finite, table-based systems.
        if 'set' in self._raw_config and 'elements' in self._raw_config['set']:
            element_names = self._raw_config['set']['elements']
            for action_config in self._raw_config.get('signature', []):
                logic_config = action_config.get('logic', {})
                if not isinstance(logic_config, dict): continue

                #symbol = action_config['symbol']
                #if logic_config.get('type') == 'table_lookup_2d':
                #    table_int = logic_config['params']['table']
                #    col_width = max(
                #        max(len(str(name)) for name in element_names), 4) + 2
                #    header = f"{'':<{col_width}}|" + "".join(
                #        [f"{e:^{col_width}}" for e in element_names])
                #    divider = "-" * len(header)
                #    rules_parts.extend([
                #        f"\n--- Computation Table for '{symbol}' (for reference) ---",
                #        header, divider
                #    ])
                #    for i, row in enumerate(table_int):
                #        row_name = element_names[i]
                #        row_str_results = [element_names[res] for res in row]
                #        row_str = f"{row_name:<{col_width}}|" + "".join(
                #            [f"{res:^{col_width}}" for res in row_str_results])
                #        rules_parts.append(row_str)

        return "\n".join(rules_parts)
