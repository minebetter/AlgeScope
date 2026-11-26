# builder.py
import yaml
from copy import deepcopy
from typing import Dict, Any, List

from .registry import registry
from .core import Set, Action, AlgebraicSystem
import json


def _try_convert_numeric(value: str) -> Any:
    """Helper to convert string values to int or float if possible."""
    if not isinstance(value, str):
        return value
    if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
        return int(value)
    try:
        return float(value)
    except (ValueError, TypeError):
        return value


class SystemFactory:
    """
    Builds an AlgebraicSystem instance from a meta-template and instance parameters.
    The template declaratively defines the system's symbols and computational rules.
    """

    def __init__(self, template_path: str):
        with open(template_path, 'r') as f:
            self.template = yaml.safe_load(f)

    def create_instance(self, instance_params: Dict[str,
                                                    Any]) -> AlgebraicSystem:
        """
        Creates a concrete instance of a formal algebraic system based on rules.

        Process:
        1. Resolve all generative parameters into a concrete config.
        2. Build the Set object by creating validator and sampler functions from the config.
        3. Build the Actions by creating algorithmic calculation logic from the config.
        4. Assemble the final AlgebraicSystem.
        """
        # Step 1: Create a fully resolved, concrete configuration object
        config_template = deepcopy(self.template)
        resolved_config = self._resolve_config(config_template,
                                               instance_params)

        format_context = {**resolved_config, **instance_params}

        # --- Step 2: Build the Set from rules ---
        set_config = resolved_config['set']

        # Build Validator
        validator_conf = set_config['validator']
        validator_builder = registry.get('set_validators',
                                         validator_conf['type'])
        validator_func = validator_builder(**validator_conf.get('params', {}))

        # Build Sampler
        sampler_conf = set_config['sampler']
        sampler_builder = registry.get('set_samplers', sampler_conf['type'])
        sampler_func = sampler_builder(**sampler_conf.get('params', {}))

        set_obj = Set(definition=set_config['definition'],
                      validator=validator_func,
                      sampler=sampler_func)

        # --- Step 3: Build the Signature (Actions) from algorithms ---
        signature: List[Action] = []
        for action_config in resolved_config.get('signature', []):
            print(action_config)
            logic_config = action_config['logic']
            logic_func_builder = registry.get('action_logics',
                                              logic_config['type'])

            # Pass only the specific params the logic needs.
            # This decouples logic from the entire system definition.
            # For table_lookup, 'elements' and 'table' will be in the params.
            # For integer_addition, params will be empty.
            logic_params = logic_config.get('params', {})
            calculation_logic = logic_func_builder(**logic_params)

            action = Action(symbol=action_config['symbol'],
                            arity=action_config['arity'],
                            name=action_config['name'],
                            precedence=action_config.get('precedence', 0),
                            calculation_logic=calculation_logic)
            signature.append(action)

        # --- Step 4: Assemble the AlgebraicSystem ---
        # Note: Identity/Inverse maps need to be configured manually or via new logic
        # This implementation assumes they are handled by the template's structure.
        system = AlgebraicSystem(
            name=resolved_config['name'],
            description=resolved_config['description'],
            rules_template=resolved_config.get('rules_template', ''),
            set_obj=set_obj,
            signature=signature,
            format_context=format_context,
            raw_config=resolved_config,
            identity_map={},  # This part might need further evolution
            inverse_map={})

        return system

    def _resolve_config(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Recursively traverses the config data, resolving generator blocks
        and formatting string templates.
        """
        if isinstance(data, dict):
            if 'generator' in data:
                gen_name = data['generator']
                gen_func = registry.get('param_generators', gen_name)

                gen_params = self._resolve_config(data.get('params', {}),
                                                  context)
                #try:
                #    gen_params = json.loads(gen_params)
                #except:
                #    pass
                # Convert resolved string params back to numbers if they look like them
                for key, val in gen_params.items():
                    gen_params[key] = _try_convert_numeric(val)
                return gen_func(**gen_params)
            else:
                return {
                    key: self._resolve_config(value, context)
                    for key, value in data.items()
                }
        elif isinstance(data, list):
            return [self._resolve_config(item, context) for item in data]
        elif isinstance(data, str):
            try:
                # Use a temporary context to avoid modifying the main one
                temp_context = context.copy()
                return _try_convert_numeric(data.format_map(temp_context))
            except (KeyError, ValueError):
                return data
        else:
            return data
