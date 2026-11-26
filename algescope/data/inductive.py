"""Create paired inductive inputs: same targets, ten distinct Q/A examples, no rules."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from .common import save
from .deductive import key, systems
from .io import load, unpack, new_output, seed_process, validate_splits, write_split
from algescope.rewards.algebra_semantics import check


def acceptable_example(row):
    result = check(row, row['answer'])
    return result['status'] == 'correct' or (
        result['status'] == 'unsupported' and Path(row['template_path']).name
        in {'image_transforms_d4.yaml', 'rubiks_cube_group.yaml'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deductive-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--example-source', type=Path, help='Frozen historical examples, JSONL/parquet, including inductive_cases')
    p.add_argument('--pool-size', type=int, default=100)
    p.add_argument('--examples', type=int, default=10)
    p.add_argument('--seed', type=int, default=20261009)
    args = p.parse_args()
    if not 1 <= args.examples <= args.pool_size:
        p.error('Require 1 <= examples <= pool-size')
    seed_process(args.seed)
    out = new_output(args.output)
    splits = {s: [unpack(r) for r in load(args.deductive_dir / (s + '.parquet'))]
              for s in ['train', 'validation', 'warmup']}
    # Supports legacy warmup that intersects training without requiring full inclusion.
    val_ids = {r['sample_id'] for r in splits['validation']}
    if val_ids & {r['sample_id'] for s in ['train', 'warmup'] for r in splits[s]}:
        raise ValueError('Validation leakage in the source splits')
    excluded = {key(r) for rows in splits.values() for r in rows}
    inverse_groups = {(r['template_path'], r['category']) for rows in splits.values()
                      for r in rows if 'inv(' in r['question']}
    pools = defaultdict(dict)
    if args.example_source:
        for source in load(args.example_source):
            row = unpack(source)
            for candidate in [row, *(row.get('inductive_cases') or [])]:
                candidate = dict(candidate, template_path=row['template_path'])
                if key(candidate) in excluded:
                    continue
                if candidate.get('num_elements', candidate.get('num_operate')) not in [2, 3]:
                    continue
                if not acceptable_example(candidate):
                    continue
                pools[(Path(row['template_path']).name, candidate['category'])][key(candidate)] = candidate
    else:
        if any(r['difficulty_field'] != 'num_operate' for rows in splits.values() for r in rows):
            raise ValueError('Historical targets require --example-source; do not reinterpret old difficulty labels')
        for name, system, generators in systems():
            for generator in generators:
                pool = {}
                for attempt in range(args.pool_size * 300):
                    row = dict(generator.generate(random.choice([2, 3])), template_path=name)
                    if key(row) not in excluded and acceptable_example(row):
                        pool[key(row)] = row
                    has_inverse = any('inv(' in c['question'] for c in pool.values())
                    if len(pool) >= args.pool_size and (
                            (name, row['category']) not in inverse_groups or has_inverse):
                        break
                else:
                    raise RuntimeError(f'Insufficient distinct examples: {name}/{type(generator).__name__}')
                pools[(name, row['category'])] = pool
    manifest = {'mode': 'paired_inductive', 'seed': args.seed, 'examples_per_target': args.examples, 'splits': {}}
    for split, rows in splits.items():
        converted = []
        for row in rows:
            candidates = list(pools[(row['template_path'], row['category'])].values())
            if len(candidates) < args.examples:
                raise ValueError('Insufficient distinct examples for ' + row['template_path'])
            for attempt in range(1000):
                examples = random.sample(candidates, args.examples)
                if 'inv(' not in row['question'] or any('inv(' in c['question'] for c in examples):
                    break
            else:
                raise ValueError('No inverse example for ' + row['template_path'] + '/' + row['category'])
            user = row['prompt'][1]['content']
            if '[COMPUTATION RULES]' not in user:
                raise ValueError('Source is not a deductive prompt')
            definition = user.split('[COMPUTATION RULES]', 1)[0].rstrip()
            cases = '\n\n'.join(f"Q: {c['question']}\nA: {c['answer']}" for c in examples)
            prompt = [row['prompt'][0], {'role': 'user', 'content': definition + '\n\n[COMPUTATION CASES]\n'
                      + cases + '\n\n[TARGET PROBLEM]\n' + row['question']
                      + '\nInfer the rule from the examples. Give the final answer in \\boxed{}.'}]
            converted.append(dict(row, benchmark='inductive', prompt=prompt,
                examples_json=json.dumps([{'question': c['question'], 'answer': c['answer']} for c in examples], ensure_ascii=False)))
        manifest['splits'][split] = write_split(out, split, converted)
    save(out / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
