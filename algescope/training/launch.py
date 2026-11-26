"""Render or launch the common VERL GRPO recipe; does not allocate cluster resources."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from algescope.data.common import save
from algescope.data.io import load, unpack

ROOT = Path(__file__).resolve().parents[2]
VERL_PIN = '8fdc4d3f202f41461f4de9f42a637228e342668b'


def command(args):
    values = yaml.safe_load((ROOT / 'configs/grpo.yaml').read_text())
    models = yaml.safe_load((ROOT / 'configs/models.yaml').read_text())
    if args.model not in models:
        raise ValueError('Choose a model label from configs/models.yaml')
    if args.gpus < 1 or args.nodes < 1 or args.tp < 1 or args.gpus % args.tp:
        raise ValueError('GPUs must be a positive multiple of rollout TP')
    if (8 * values['actor_rollout_ref.rollout.n']) % (args.gpus * args.nodes):
        raise ValueError('PPO minibatch responses must be divisible by the GPU count')
    values.update({
        'actor_rollout_ref.model.path': args.checkpoint,
        'actor_rollout_ref.rollout.tensor_model_parallel_size': args.tp,
        'data.train_files': str(args.data.resolve() / 'train.parquet'),
        'data.val_files': str(args.data.resolve() / 'validation.parquet'),
        'data.custom_cls.path': str(ROOT / 'algescope/training/dataset.py'),
        'data.custom_cls.name': 'AlgeScopeDataset',
        'custom_reward_function.path': str(ROOT / 'algescope/rewards/compute.py'),
        'custom_reward_function.name': 'compute_score',
        'trainer.n_gpus_per_node': args.gpus, 'trainer.nnodes': args.nodes,
        'trainer.total_training_steps': args.steps,
        'trainer.experiment_name': args.model + '-' + args.benchmark,
        'trainer.default_local_dir': str(args.output.resolve() / 'checkpoints'),
        'trainer.validation_data_dir': str(args.output.resolve() / 'validation'),
        'trainer.rollout_data_dir': str(args.output.resolve() / 'rollouts'),
        'hydra.run.dir': str(args.output.resolve() / 'hydra'),
        'algorithm.norm_adv_by_std_in_grpo': args.benchmark == 'synlogic',
    })
    overrides = [k + '=' + json.dumps(v, ensure_ascii=False, separators=(',', ':')) for k, v in values.items()]
    return [sys.executable, '-u', '-m', 'verl.trainer.main_ppo', *overrides]


def preflight(args):
    import importlib.util
    from transformers import AutoTokenizer
    from algescope.training.prompts import prompt_ids
    spec = importlib.util.find_spec('verl')
    if spec is None:
        raise ValueError('Install the pinned VERL checkout first')
    repo = Path(spec.origin).resolve().parent.parent
    pin = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if pin != VERL_PIN:
        raise ValueError(f'Expected VERL {VERL_PIN}; found {pin}')
    subprocess.run(['git', '-C', str(repo), 'apply', '--reverse', '--check',
                    str(ROOT / 'patches/verl_v050_trace.patch')], check=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=False)
    summary, split_ids = {}, {}
    for split in ['train', 'validation']:
        rows = load(args.data / (split + '.parquet'))
        if not rows:
            raise ValueError('Empty dataset: ' + split)
        records = [unpack(r) for r in rows]
        if any(r['benchmark'] != args.benchmark for r in records):
            raise ValueError('Benchmark argument does not match the actual data')
        split_ids[split] = {r['sample_id'] for r in records}
        if len(split_ids[split]) != len(records):
            raise ValueError('Duplicate sample IDs')
        lengths = [len(prompt_ids(tokenizer, r['prompt'])) for r in rows]
        if max(lengths) > 8192:
            raise ValueError('An input exceeds 8192 tokens; no rows will be filtered or truncated')
        summary[split] = {'rows': len(rows), 'max_prompt_tokens': max(lengths)}
    if split_ids['train'] & split_ids['validation']:
        raise ValueError('Training/validation leakage')
    if args.benchmark == 'synlogic' and not os.environ.get('SYNLOGIC_SOURCE'):
        raise ValueError('Set SYNLOGIC_SOURCE to the pinned patched upstream checkout')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--benchmark', choices=['deductive', 'inductive', 'synlogic'], required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpus', type=int, default=8)
    p.add_argument('--nodes', type=int, default=1)
    p.add_argument('--tp', type=int, default=4)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--execute', action='store_true', help='Without this flag, only print the command')
    args = p.parse_args()
    if args.steps < 1:
        p.error('steps must be positive')
    cmd = command(args)
    if not args.execute:
        print(json.dumps(cmd, indent=2))
        return
    if args.output.exists():
        p.error('Output exists; use a new directory')
    report = preflight(args)
    args.output.mkdir(parents=True)
    save(args.output / 'preflight.json', report)
    save(args.output / 'command.json', cmd)
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    started = time.time()
    with (args.output / 'console.log').open('w') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        code = proc.wait()
    save(args.output / 'exit.json', {'exit_code': code, 'elapsed_seconds': time.time()-started})
    raise SystemExit(code)


if __name__ == '__main__':
    main()
