"""Regenerate the twelve SynLogic task families used in our cross-benchmark study."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import contextlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess

from . import synlogic_adapter as adapter
from .common import digest, save
from .io import load, new_output, seed_process, validate_splits, write_split

PIN = 'd8c527fd17edb739172619efb9b681805fc74b8d'


def deadline(signum, frame):
    raise TimeoutError('SynLogic generation deadline')


def worker(spec):
    task, source, root, seed, targets = spec
    import numpy as np
    adapter.setup(source)
    seen, prompts, result = set(), set(), {}
    rejections = Counter()
    folder = Path(root) / 'parts' / task
    folder.mkdir(parents=True)
    for split, count in targets.items():
        records = []
        attempt = 0
        while len(records) < count:
            if attempt >= max(20000, count * 100):
                raise RuntimeError(f'Generation exhausted: {task}/{split}, rejections={dict(rejections)}')
            local_seed = int(digest([seed, 'synlogic', task, split, attempt])[:8], 16)
            attempt += 1
            random.seed(local_seed)
            np.random.seed(local_seed)
            signal.signal(signal.SIGALRM, deadline)
            signal.setitimer(signal.ITIMER_REAL, 15)
            try:
                with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                    row = adapter.generate(task, split, local_seed)
            except (ImportError, FileNotFoundError):
                raise
            except Exception as exc:
                rejections[type(exc).__name__ + ':' + str(exc)[:120]] += 1
                continue
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            if row['canonical_instance_hash'] in seen or row['prompt_sha256'] in prompts:
                rejections['duplicate'] += 1
                continue
            seen.add(row['canonical_instance_hash'])
            prompts.add(row['prompt_sha256'])
            row['generation_attempt'] = attempt - 1
            records.append(row)
        target = folder / (split + '.jsonl')
        target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
        result[split] = len(records)
        print(f'{task}/{split}: {len(records)}', flush=True)
    return {'task': task, 'counts': result, 'rejections': dict(rejections)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Pinned SynLogic checkout with the English patch applied')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--warmup-per-task', type=int, default=750)
    p.add_argument('--train-per-task', type=int, default=7500)
    p.add_argument('--validation-per-task', type=int, default=250)
    p.add_argument('--seed', type=int, default=20261016)
    args = p.parse_args()
    if min(args.workers, args.warmup_per_task, args.train_per_task, args.validation_per_task) < 1:
        p.error('Worker and sample counts must be positive')
    seed_process(args.seed)
    source = args.source.resolve()
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != PIN:
        raise ValueError(f'Expected SynLogic {PIN}; found {actual}')
    patch = Path(__file__).resolve().parents[2] / 'patches/synlogic_english.patch'
    subprocess.run(['git', '-C', str(source), 'apply', '--reverse', '--check', str(patch)], check=True)
    out = new_output(args.output)
    # Historical order: easier warmup, then train, then independent validation.
    targets = {'warmup': args.warmup_per_task, 'train': args.train_per_task, 'validation': args.validation_per_task}
    specs = [(task, str(source), str(out), args.seed, targets) for task in adapter.TASKS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        reports = list(pool.map(worker, specs))
    splits = {}
    for split in targets:
        splits[split] = [row for task in adapter.TASKS for row in load(out / 'parts' / task / (split + '.jsonl'))]
        random.Random(1234).shuffle(splits[split])
    validate_splits(splits, warmup_subset=False)
    manifest = {'upstream_commit': PIN, 'seed': args.seed, 'task_reports': reports,
                'validation_protocol': 'independent instances from the training difficulty distribution',
                'splits': {s: write_split(out, s, rows) for s, rows in splits.items()}}
    save(out / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
