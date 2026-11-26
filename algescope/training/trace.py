"""Bridge real vLLM termination metadata to rewards and VERL JSONL archives."""
import json


def decode_response(tokenizer, response_ids, stop_reason=None):
    body = list(response_ids)
    transport = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    if isinstance(stop_reason, int):
        transport.add(stop_reason)
    while body and body[-1] in transport:
        body.pop()
    return tokenizer.decode(body, skip_special_tokens=False)


def prepare_score_inputs(tokenizer, prompt_ids, response_ids, fields):
    metadata = fields.get('algescope_trace')
    if metadata is None:
        raise ValueError('Apply patches/verl_v050_trace.patch to the pinned VERL checkout')
    trace = json.loads(metadata)
    count = trace['response_token_count']
    if not 0 <= count <= len(response_ids):
        raise ValueError('Invalid response length in vLLM metadata')
    prompt = prompt_ids.cpu().tolist()
    response = response_ids[:count].cpu().tolist()
    text = decode_response(tokenizer, response, trace.get('stop_reason'))
    info = dict(fields.get('extra_info', {}))
    info['finish_reason'] = trace['finish_reason']
    # JSON strings keep variable-length arrays out of VERL's numeric metric reducer.
    info['_trace_record'] = {
        'finish_reason': trace['finish_reason'], 'stop_reason': json.dumps(trace['stop_reason']),
        'prompt_token_ids': json.dumps(prompt), 'response_token_ids': json.dumps(response),
        'prompt_token_count': len(prompt), 'response_token_count': count,
        'output_with_special_tokens': tokenizer.decode(response, skip_special_tokens=False),
        'termination_metadata_source': 'vllm.CompletionOutput',
    }
    return text, info
