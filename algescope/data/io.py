"""Portable dataset IO and VERL records; generated content stays outside Git."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys

from .common import digest, save


def seed_process(seed):
    value = str(seed % (2**32))
    if os.environ.get('PYTHONHASHSEED') != value:
        os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]],
                  dict(os.environ, PYTHONHASHSEED=value))
    random.seed(seed)
    import numpy as np
    np.random.seed(int(value))


def load(path):
    path = Path(path)
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        rows = pq.read_table(path).to_pylist()
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows


def unpack(row):
    truth = row.get('reward_model', {}).get('ground_truth')
    if isinstance(truth, str) and truth.startswith('{'):
        metadata = json.loads(truth)
        if metadata.get('schema') == 'algescope-v1':
            return dict(metadata['record'], prompt=row['prompt'])
    return row


def pack(row):
    metadata = {k: v for k, v in row.items() if k != 'prompt'}
    return {
        'data_source': row['benchmark'] + '/' + row['task'],
        'prompt': row['prompt'],
        'reward_model': {'style': 'rule', 'ground_truth': json.dumps(
            {'schema': 'algescope-v1', 'record': metadata}, ensure_ascii=False)},
        'extra_info': {'sample_id': row['sample_id'], 'task': row['task']},
    }


def write_split(folder, name, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = Path(folder) / (name + '.parquet')
    if path.exists():
        raise FileExistsError(path)
    if not rows:
        raise ValueError('Cannot publish an empty split')
    ids = [row['sample_id'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate sample IDs in ' + name)
    tmp = path.with_suffix('.partial')
    writer = None
    try:
        for start in range(0, len(rows), 256):
            table = pa.Table.from_pylist([pack(r) for r in rows[start:start+256]])
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression='zstd')
            writer.write_table(table)
    finally:
        if writer:
            writer.close()
    if pq.read_metadata(tmp).num_rows != len(rows):
        raise ValueError('Incomplete parquet write')
    tmp.replace(path)
    return {'rows': len(rows), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def validate_splits(splits, warmup_subset=True):
    ids = {s: {r['sample_id'] for r in rows} for s, rows in splits.items()}
    for s, rows in splits.items():
        if len(ids[s]) != len(rows):
            raise ValueError(f'Duplicate questions in {s}')
    if ids['validation'] & (ids['train'] | ids['warmup']):
        raise ValueError('Validation targets overlap training/warmup; supply cleaned source files')
    if warmup_subset and not ids['warmup'] <= ids['train']:
        raise ValueError('Warmup must be a subset of training targets')
    if warmup_subset is False and ids['warmup'] & ids['train']:
        raise ValueError('Warmup and training targets overlap')
