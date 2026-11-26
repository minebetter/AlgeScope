"""Aggregate saved traces with explicit complete-coverage and sampling checks."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path

from algescope.data.io import load, unpack


def summarize(rows, expected):
    grouped = defaultdict(list)
    for row in rows:
        if row['sample_id'] not in expected:
            raise ValueError('Unknown sample ID in traces')
        if row['score'] not in [0, 1, 0.0, 1.0]:
            raise ValueError('Expected binary outcome scores')
        if not isinstance(row.get('finish_reason'), str):
            raise ValueError('Missing actual termination metadata')
        grouped[row['sample_id']].append(row)
    if set(grouped) != set(expected):
        raise ValueError('Incomplete question coverage; do not report a partial file as a full evaluation')
    sizes = {len(v) for v in grouped.values()}
    if len(sizes) != 1:
        raise ValueError('Unequal sample counts per question')
    k = sizes.pop()
    if not k:
        raise ValueError('No samples')
    buckets = defaultdict(list)
    for key, samples in grouped.items():
        reference = expected[key]
        if any('draw' in r for r in samples):
            if {r.get('draw') for r in samples} != set(range(k)):
                raise ValueError('Missing or duplicated draw indices')
        difficulty = reference.get('difficulty')
        field = reference.get('difficulty_field', 'task_specific')
        band = ('ID' if 2 <= difficulty <= 5 else 'OOD' if 6 <= difficulty <= 17 else 'other') if isinstance(difficulty, int) else 'task_specific'
        for template, group_band in [('ALL', 'ALL'), (reference['template_path'], 'ALL'), ('ALL', band), (reference['template_path'], band)]:
            buckets[(template, group_band, field)].append(samples)
    result = []
    for (template, band, field), questions in sorted(buckets.items()):
        scores = [r['score'] for samples in questions for r in samples]
        result.append({'template': template, 'band': band, 'difficulty_field': field,
                       'questions': len(questions), 'samples_per_question': k,
                       'average_at_k': sum(scores)/len(scores),
                       'pass_at_k': sum(any(r['score'] for r in samples) for samples in questions)/len(questions),
                       'length_terminated_fraction': sum(r.get('finish_reason') == 'length' for samples in questions for r in samples)/len(scores)})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--traces', type=Path, required=True, help='One JSONL or a VERL validation directory')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output exists')
    records = [unpack(r) for r in load(args.data)]
    expected = {r['sample_id']: r for r in records}
    if len(expected) != len(records):
        raise ValueError('Duplicate dataset IDs')
    files = sorted(args.traces.glob('*.jsonl'), key=lambda p: int(p.stem)) if args.traces.is_dir() else [args.traces]
    if not files:
        raise ValueError('No trace files found')
    results = []
    for file in files:
        results.extend(dict(step=file.stem, **r) for r in summarize(load(file), expected))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == '__main__':
    main()
