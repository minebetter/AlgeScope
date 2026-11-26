import os
import random
import json
import traceback
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from simple_parsing import ArgumentParser, field

from algebrarium.builder import SystemFactory
from algebrarium.core import AlgebraicSystem
from algebrarium.factory import ExpressionQuestionGenerator, EquationQuestionGenerator, QuestionGenerator, SystemOfGroupEquationsQuestionGenerator, StepByStepExpressionGenerator
import algebrarium.components  # 确保 components 被注册
#import algebrarium.generators


# vvv --- NEW: Command-line Argument Configuration --- vvv
@dataclass
class ScriptArguments:
    """Configuration for the data generation script."""
    # Input/Output paths
    template_dir: str = field(
        default="./templates",
        alias="-t",
        help="Root directory containing the .yaml template files.")
    output_dir: str = field(default="./data/",
                            alias="-o",
                            help="Path to the output JSONL file.")

    # Generation parameters
    num_questions_per_instance: int = field(
        default=3,
        alias="-n",
        help=
        "Number of questions to generate for EACH 'demo_instance' in a template."
    )
    min_difficulty: int = field(
        default=3,
        alias="-m",
        help=
        "Minimum complexity (number of elements) for the easiest questions.")
    max_difficulty: int = field(
        default=5,
        alias="-M",
        help=
        "Maximum complexity (number of elements) for the hardest questions.")
    max_attempts_factor: int = field(
        default=2,
        alias="--max-attempts",
        help=
        "Maximum complexity (number of elements) for the hardest questions.")

    sys_of_eq_num: int = field(
        default=2,
        alias="--num-equations-for-system",
        help=
        "Maximum number of equations in a system, defining the complexity of the hardest questions."
    )
    type: str = field(default="deductive",
                      help="type of reasoning, inductive or deductive")
    num_inductive_per_case: int = field(
        default=10, help="type of reasoning, inductive or deductive")
    num_inductive_total: int = field(
        default=1000, help="type of reasoning, inductive or deductive")


def generate_unique_questions(generator: QuestionGenerator,
                              num_target: int,
                              min_elements: int = 3,
                              max_elements: int = 5,
                              max_attempts_factor: int = 30) -> List[Dict]:
    """Generates a specified number of unique questions using the given generator."""
    unique_questions = []
    seen_question_strings = set()
    max_attempts = num_target * max_attempts_factor

    for _ in range(max_attempts):
        if len(unique_questions) >= num_target:
            break

        num_elements = random.randint(min_elements, max_elements)
        q_dict = generator.generate(num_elements)
        question_str = q_dict['question']

        if question_str not in seen_question_strings:
            seen_question_strings.add(question_str)
            unique_questions.append(q_dict)

    return unique_questions


def generate_deductive_questions_for_system(system: AlgebraicSystem,
                                            background: str,
                                            gen_config: dict) -> List[Dict]:
    """
    Generates a set of questions for a given system based on the dynamic difficulty config.
    """
    all_questions = []

    if not any(action.arity > 0 for action in system.signature):
        print(
            "    - No operations with arity > 0. Skipping question generation."
        )
        return []

    expr_gen = ExpressionQuestionGenerator(system)
    eqn_gen = EquationQuestionGenerator(system)
    sys_eqn_gen = SystemOfGroupEquationsQuestionGenerator(
        system, gen_config['sys_of_eq_num'])
    #rea_gen = StepByStepExpressionGenerator(system)
    gen_cycle = [expr_gen, eqn_gen, sys_eqn_gen]

    for generator in gen_cycle:
        new_questions = generate_unique_questions(
            generator,
            num_target=gen_config['num_questions_per_instance'],
            min_elements=gen_config['min_difficulty'],
            max_elements=gen_config['max_difficulty'],
            max_attempts_factor=gen_config['max_attempts_factor'],
        )
        for q in new_questions:
            q['background'] = background
            q['inductive_cases'] = None
            q['template_path'] = getattr(system, 'template_path', 'N/A')
        all_questions.extend(new_questions)

    return all_questions


def generate_inductive_questions_for_system(system: AlgebraicSystem,
                                            background: str,
                                            gen_config: dict) -> List[Dict]:
    """
    Generates a set of questions for a given system based on the dynamic difficulty config.
    """
    all_questions = []

    if not any(action.arity > 0 for action in system.signature):
        print(
            "    - No operations with arity > 0. Skipping question generation."
        )
        return []

    expr_gen = ExpressionQuestionGenerator(system)
    eqn_gen = EquationQuestionGenerator(system)
    sys_eqn_gen = SystemOfGroupEquationsQuestionGenerator(
        system, gen_config['sys_of_eq_num'])
    gen_cycle = [expr_gen, eqn_gen, sys_eqn_gen]

    for generator in gen_cycle:
        new_questions = generate_unique_questions(
            generator,
            num_target=gen_config['num_questions_per_instance'],
            min_elements=gen_config['min_difficulty'],
            max_elements=gen_config['max_difficulty'],
            max_attempts_factor=gen_config['max_attempts_factor'],
        )
        inductive_cases = generate_unique_questions(
            generator,
            num_target=gen_config['num_inductive_total'],
            min_elements=2,
            max_elements=3,
            max_attempts_factor=gen_config['max_attempts_factor'],
        )
        inductive_questions = set(map(lambda x: x['question'],
                                      inductive_cases))
        gen_questions = list(
            filter(lambda x: x['question'] not in inductive_questions,
                   new_questions))

        for q in gen_questions:
            q['background'] = background
            q['inductive_cases'] = random.choices(
                inductive_cases, k=gen_config['num_inductive_per_case'])
            q['template_path'] = getattr(system, 'template_path', 'N/A')
        all_questions.extend(gen_questions)

    return all_questions


def process_instance(system: AlgebraicSystem, params: dict,
                     gen_config: dict) -> List[Dict]:
    """Processes a single system instance to generate a set of questions."""
    title = system._format_string(system.name)
    print(f"  - System: {title}")
    print(f"  - Params: {params}")

    if gen_config['type'] == 'deductive':
        background = ("[MATHEMATICAL DEFINITION]\n"
                      f"{system.get_full_definition()}\n\n"
                      "[COMPUTATION RULES]\n"
                      f"{system.get_computation_rules()}")

        questions = generate_deductive_questions_for_system(
            system, background, gen_config)
        print(
            f"  --> Generated {len(questions)} total deductive questions for this instance."
        )
        return questions
    elif gen_config['type'] == 'inductive':
        background = ("[MATHEMATICAL DEFINITION]\n"
                      f"{system.get_full_definition()}\n\n")

        questions = generate_inductive_questions_for_system(
            system, background, gen_config)
        print(
            f"  --> Generated {len(questions)} total inductive questions for this instance."
        )
        return questions


def process_template(template_path: str, gen_config: dict) -> List[Dict]:
    """Loads a template and generates questions for all its demo instances."""
    print("\n" + "#" * 80)
    print(f"PROCESSING TEMPLATE: {template_path}")
    all_questions_for_template = []

    try:
        factory = SystemFactory(template_path)
        instance_params_list = factory.template.get("demo_instances", [])

        if not instance_params_list:
            print(f"  - No 'demo_instances' found. Skipping.")
            return []
        instance_params_list = instance_params_list[:1]
        print(f"  - Found {len(instance_params_list)} instance(s) to process.")
        for i, params in enumerate(instance_params_list, 1):
            print(f"\n  --- Instance {i}/{len(instance_params_list)} ---")
            system_instance = factory.create_instance(params)
            system_instance.template_path = template_path.split("/")[-1]
            system_instance.params = params
            instance_questions = process_instance(system_instance, params,
                                                  gen_config)
            all_questions_for_template.extend(instance_questions)

    except Exception as e:
        print(f"\n  FATAL ERROR processing template file {template_path}: {e}")
        traceback.print_exc()

    return all_questions_for_template


def write_jsonl(output_file: str, data: List[Dict], mode: str = "w"):
    """Writes a list of dictionaries to a JSONL file."""
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, mode, encoding='utf-8') as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    except IOError as e:
        print(f"Error writing to file {output_file}: {e}")


def main():

    parser = ArgumentParser()
    parser.add_arguments(ScriptArguments, dest="args")
    args = parser.parse_args().args
    print(args)
    gen_config = asdict(args)

    print("--- SCRIPT CONFIGURATION ---")
    print(f"Template Directory: {args.template_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Questions per Instance: {args.num_questions_per_instance}")
    print(
        f"Difficulty Range (Complexity): {args.min_difficulty} to {args.max_difficulty}"
    )
    print("--------------------------\n")

    if not os.path.isdir(args.template_dir):
        print(f"Error: Template directory '{args.template_dir}' not found.")
        return

    print(
        f"Starting question generation from templates in '{args.template_dir}'..."
    )

    template_files = []
    for root, _, files in os.walk(args.template_dir):
        for filename in files:
            if filename.endswith((".yaml", ".yml")):
                template_files.append(os.path.join(root, filename))

    template_files.sort()
    for template_path in template_files:
        questions_from_file = process_template(template_path, gen_config)
        outpath = os.path.join(
            args.output_dir,
            template_path.split("/")[-1].replace('.yaml', '.jsonl'))
        print(f"Writing all questions to '{outpath}'...")
        write_jsonl(output_file=outpath, data=questions_from_file, mode='w')


if __name__ == "__main__":
    main()
