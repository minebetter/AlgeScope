"""Greedy or sampled vLLM evaluation with complete per-question trace records."""
import argparse
import json
import os
from pathlib import Path
import time

from algescope.data.common import save
from algescope.data.io import load, unpack
from algescope.rewards.compute import score_record
from algescope.training.prompts import prompt_ids
from algescope.training.trace import decode_response


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tp', type=int, default=4)
    p.add_argument('--samples', type=int, default=1)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--top-k', type=int, default=-1)
    p.add_argument('--max-response-tokens', type=int, default=8192)
    p.add_argument('--max-prompt-tokens', type=int, default=8192)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()
    if min(args.tp, args.samples, args.batch_size, args.max_response_tokens, args.max_prompt_tokens) < 1:
        p.error('Counts must be positive')
    if args.temperature < 0 or not 0 < args.top_p <= 1 or args.top_k == 0 or args.top_k < -1:
        p.error('Invalid sampling parameters')
    if args.temperature == 0 and args.samples != 1:
        p.error('Greedy evaluation uses exactly one response per question')
    if args.output.exists():
        p.error('Output exists; use a fresh directory')
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=False)
    rows = [unpack(r) for r in load(args.data)]
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Dataset is empty or contains duplicate sample IDs')
    if any(r['benchmark'] == 'synlogic' for r in rows) and not os.environ.get('SYNLOGIC_SOURCE'):
        raise ValueError('Set SYNLOGIC_SOURCE before evaluating SynLogic')
    tokens = [prompt_ids(tokenizer, r['prompt']) for r in rows]
    if max(map(len, tokens)) > args.max_prompt_tokens:
        raise ValueError('A prompt exceeds the configured limit; nothing was truncated')
    args.output.mkdir(parents=True)
    save(args.output / 'config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    engine = LLM(model=args.checkpoint, tensor_parallel_size=args.tp, trust_remote_code=False,
                 max_model_len=args.max_prompt_tokens + args.max_response_tokens, seed=args.seed)
    params = SamplingParams(n=args.samples, temperature=args.temperature, top_p=args.top_p,
                            top_k=args.top_k, max_tokens=args.max_response_tokens, seed=args.seed)
    path = args.output / 'traces.jsonl'
    started = time.time()
    total = 0
    with path.open('w') as stream:
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start:start+args.batch_size]
            outputs = engine.generate([{'prompt_token_ids': t} for t in tokens[start:start+args.batch_size]], params)
            if len(outputs) != len(chunk):
                raise RuntimeError('Generation returned incomplete question coverage')
            for offset, (row, output) in enumerate(zip(chunk, outputs, strict=True)):
                if len(output.outputs) != args.samples:
                    raise RuntimeError('Generation returned an incomplete sample group')
                for draw, completion in enumerate(output.outputs):
                    text = decode_response(tokenizer, completion.token_ids, completion.stop_reason)
                    score, status = score_record(row, text, completion.finish_reason)
                    record = {'sample_id': row['sample_id'], 'draw': draw, 'template': row['template_path'],
                              'difficulty': row.get('difficulty', 'task_specific'),
                              'difficulty_field': row.get('difficulty_field', 'task_specific'),
                              'input': tokenizer.decode(tokens[start+offset], skip_special_tokens=False),
                              'output': text, 'score': score, 'status': status,
                              'output_with_special_tokens': tokenizer.decode(completion.token_ids, skip_special_tokens=False),
                              'prompt_token_ids': tokens[start+offset], 'response_token_ids': list(completion.token_ids),
                              'response_token_count': len(completion.token_ids),
                              'finish_reason': completion.finish_reason, 'stop_reason': completion.stop_reason,
                              'termination_metadata_source': 'vllm.CompletionOutput'}
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                    total += 1
            stream.flush()
    save(args.output / 'complete.json', {'questions': len(rows), 'responses': total,
                                        'elapsed_seconds': time.time()-started})


if __name__ == '__main__':
    main()
