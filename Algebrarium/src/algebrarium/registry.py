# registry.py
from typing import Dict, Callable, Any


class ComponentRegistry:
    """A centralized, robust registry for all pluggable components."""

    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {
            "action_logics": {},
            "set_generators": {},  # Kept for backward compatibility if needed
            "set_validators": {},  # NEW: For membership rules
            "set_samplers": {},  # NEW: For generating example elements
            "param_generators": {},
        }

    def register(self, category: str, name: str) -> Callable:
        """A decorator to register a component."""
        if category not in self._registry:
            raise KeyError(
                f"Invalid component category: '{category}'. Valid categories are {list(self._registry.keys())}"
            )

        def decorator(cls_or_func: Any) -> Any:
            if name in self._registry[category]:
                raise ValueError(
                    f"Component '{name}' already registered in category '{category}'."
                )

            self._registry[category][name] = cls_or_func
            print(f"register {category} {name}")
            return cls_or_func

        return decorator

    def get(self, category: str, name: str) -> Any:
        """Retrieve a component by its category and name."""
        try:
            return self._registry[category][name]
        except KeyError:
            raise ValueError(
                f"Component '{name}' not found in category '{category}'.")


# Create a global instance for the application to use
registry = ComponentRegistry()
