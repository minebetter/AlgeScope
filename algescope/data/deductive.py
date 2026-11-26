"""Construct deductive warmup/train/validation or export frozen historical splits."""
import argparse
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import random
import sys

from .common import digest, save
from .io import load, unpack, new_output, seed_process, validate_splits, write_split

ROOT = Path(__file__).resolve().parents[2]
PIN = 'de97a57b8dc7536f29af11e5a7e7704cbd590691'
INSTRUCTION = 'Solve the problem step by step. Give the final answer in \\boxed{}.'


def key(row):
    return (Path(row['template_path']).name, ' '.join(row['question'].split()))


def systems():
    sys.path.insert(0, str(ROOT / 'Algebrarium/src'))
    with contextlib.redirect_stdout(io.StringIO()):
        from algebrarium import components
        from algebrarium.builder import SystemFactory
        from algebrarium.factory import (ExpressionQuestionGenerator, EquationQuestionGenerator,
                                        SystemOfGroupEquationsQuestionGenerator)
    names = json.loads((ROOT / 'configs/algebra_templates.json').read_text())
    paths = {p.name: p for p in (ROOT / 'Algebrarium/templates').rglob('*.yaml')}
    for name in names:
        factory = SystemFactory(str(paths[name]))
        if name == 'tree_rotation_Z1_rank.yaml':
            # Supply the already-defined inverse map; leave the fixed source intact.
            factory.template['set']['sampler']['params']['inverse_map'] = factory.template['set']['inverse_map']
        with contextlib.redirect_stdout(io.StringIO()):
            system = factory.create_instance(factory.template['demo_instances'][0])
        generators = [ExpressionQuestionGenerator(system), EquationQuestionGenerator(system),
                      SystemOfGroupEquationsQuestionGenerator(system, 2)]
        yield name, system, generators


def normalize(row):
    row = dict(row)
    row['template_path'] = Path(row['template_path']).name
    row.update(benchmark='deductive', task=row['template_path'].removesuffix('.yaml'),
               sample_id='algebra:' + digest(key(row)))
    if 'num_elements' in row:
        row.update(difficulty=row['num_elements'], difficulty_field='num_elements')
    else:
        row.update(difficulty=row['num_operate'], difficulty_field='num_operate')
    return row


def generate(count, low, high, name, system, generators, seen, max_attempts=300):
    rows = []
    background = ('[MATHEMATICAL DEFINITION]\n' + system.get_full_definition()
                  + '\n\n[COMPUTATION RULES]\n' + system.get_computation_rules())
    for index in range(count):
        difficulty = low + index % (high-low+1)
        for attempt in range(max_attempts):
            row = random.choice(generators).generate(difficulty)
            row.update(template_path=name, background=background)
            if key(row) in seen:
                continue
            seen.add(key(row))
            row['prompt'] = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                             {'role': 'user', 'content': background + '\n\n' + row['question'] + '\n' + INSTRUCTION}]
            rows.append(normalize(row))
            break
        else:
            raise RuntimeError(f'Unique generation exhausted: {name}, budget={difficulty}, row={index}')
    return rows


def build(args):
    if args.source_dir:
        splits = {s: [normalize(unpack(r)) for r in load(args.source_dir / (s + '.parquet'))]
                  for s in ['train', 'validation', 'warmup']}
        if any('[COMPUTATION RULES]' not in '\n'.join(m['content'] for m in r['prompt']).upper()
               for rows in splits.values() for r in rows):
            raise ValueError('Frozen source must contain deductive prompts with explicit computation rules')
        # Legacy warmup need not be a subset, but must remain outside validation.
        validate_splits(splits, warmup_subset=None)
        return splits, 'frozen_source_export'
    splits = {s: [] for s in ['train', 'validation', 'warmup']}
    seen = set()
    for name, system, generators in systems():
        validation = generate(args.validation_per_template, 2, 17, name, system, generators, seen)
        train = generate(args.train_per_template, 2, 5, name, system, generators, seen)
        easy = [r for r in train if r['difficulty'] == 2]
        if len(easy) < args.warmup_per_template:
            raise ValueError('Not enough difficulty-2 training targets for the requested warmup count')
        splits['validation'].extend(validation)
        splits['train'].extend(train)
        splits['warmup'].extend(random.sample(easy, args.warmup_per_template))
        print(f'{name}: train={len(train)}, validation={len(validation)}', flush=True)
    validate_splits(splits)
    return splits, 'fresh_num_operate_not_historical_num_elements'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-dir', type=Path)
    p.add_argument('--train-per-template', type=int, default=3000)
    p.add_argument('--validation-per-template', type=int, default=100)
    p.add_argument('--warmup-per-template', type=int, default=300)
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()
    if min(args.train_per_template, args.validation_per_template, args.warmup_per_template) < 1:
        p.error('Counts must be positive')
    seed_process(args.seed)
    out = new_output(args.output)
    splits, mode = build(args)
    manifest = {'mode': mode, 'seed': args.seed, 'upstream_commit': PIN, 'splits': {}}
    for split, rows in splits.items():
        random.Random(args.seed).shuffle(rows)
        manifest['splits'][split] = dict(write_split(out, split, rows),
            difficulty=dict(Counter(str(r['difficulty']) for r in rows)))
    save(out / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
