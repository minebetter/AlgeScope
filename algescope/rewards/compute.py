"""Shared outcome reward for training, validation and standalone evaluation."""
import json
import os
import re

from algescope.rewards.algebra_semantics import check, scalar
from algescope.rewards.answer_format import extract_answer


def last_box(text):
    start = text.rfind('\\boxed{')
    if start < 0:
        return None
    start += len('\\boxed{')
    depth = 1
    for i in range(start, len(text)):
        depth += (text[i] == '{') - (text[i] == '}')
        if depth == 0:
            return text[start:i]
    return None


def ordered_answer(text):
    text = text.replace('\\alpha', 'α').replace('\\beta', 'β').strip()
    pairs = []
    for part in re.split(r',\s*(?=[αβx]\s*=)', text):
        match = re.fullmatch(r'\s*([αβx])\s*=\s*(.*?)\s*', part)
        if not match:
            return scalar(text)
        pairs.append((match[1], scalar(match[2])))
    return tuple(sorted(pairs)) if len({k for k, _ in pairs}) == len(pairs) else None


def score_record(row, text, finish_reason):
    if finish_reason != 'stop':
        return 0.0, 'unfinished'
    if '<think>' in text and '</think>' not in text:
        return 0.0, 'unclosed_think'
    if row['benchmark'] == 'synlogic':
        from algescope.data.synlogic_adapter import setup, judge
        source = os.environ.get('SYNLOGIC_SOURCE')
        if source:
            setup(source)
        answer = extract_answer(text)
        if answer is None:
            return 0.0, 'invalid_final_answer'
        ok = judge(row['task'], answer, row['answer'], json.loads(row['verification_metadata_json']))
        return float(ok), 'correct' if ok else 'incorrect'
    if row['benchmark'] not in ['deductive', 'inductive']:
        raise ValueError('Unsupported benchmark: ' + row['benchmark'])
    answer = last_box(text.split('</think>')[-1])
    if answer is None:
        return 0.0, 'no_final_boxed'
    verdict = check(row, answer)
    if verdict['status'] == 'unsupported':
        # D4 / Rubik: preserve order; do not silently sort noncommuting operations.
        candidate = ordered_answer(answer)
        ok = candidate is not None and candidate == ordered_answer(row['answer'])
        return float(ok), 'reference_match' if ok else 'reference_mismatch'
    return float(verdict['status'] == 'correct'), 'semantic_' + verdict['status']


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    truth = json.loads(ground_truth)
    if truth.get('schema') != 'algescope-v1':
        raise ValueError('Use an AlgeScope data preparation entry point')
    row = truth['record']
    if data_source != row['benchmark'] + '/' + row['task']:
        raise ValueError('Dataset/reward metadata mismatch')
    info = extra_info or {}
    if 'finish_reason' not in info:
        raise ValueError('Missing generation termination metadata; apply the pinned VERL trace patch')
    score, status = score_record(row, solution_str, info['finish_reason'])
    result = {'score': score, 'acc': score, 'status': status, 'sample_id': row['sample_id'],
              'template': row['template_path'], 'difficulty': str(row.get('difficulty', 'task_specific')),
              'difficulty_field': row.get('difficulty_field', 'task_specific')}
    result.update(info.get('_trace_record', {}))
    return result
