"""Frozen dataset records and bounded, non-executing numeric parsing."""
import ast
from fractions import Fraction
import hashlib
import json
import operator
from pathlib import Path
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def numeric(text):
    text = str(text).strip().strip('$').replace('−', '-')
    text = re.sub(r'\\(?:d?frac)\{(-?\d+)\}\{(-?\d+)\}', r'(\1)/(\2)', text)
    if len(text) > 200 or not re.fullmatch(r'[0-9eE.+*/%()\s-]+', text):
        raise ValueError('Not a bounded numeric expression')
    tree = ast.parse(text, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 80:
        raise ValueError('Expression too large')
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Mod: operator.mod}
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = Fraction(str(node.value))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            result = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in ops:
            result = ops[type(node.op)](visit(node.left), visit(node.right))
        else:
            raise ValueError('Unsupported expression')
        if result.numerator.bit_length() > 512 or result.denominator.bit_length() > 512:
            raise ValueError('Numeric result too large')
        return result
    return visit(tree.body)


SYSTEM = ('Solve the problem carefully. Explain your reasoning, then give one final answer '
          'inside <answer>...</answer>. The answer block may contain a number, sequence, '
          'equation, list, or grid as appropriate. Do not put alternative answers in the block.')


def record(benchmark, task, split, seed, question, answer, metadata, canonical, parameters, gold_trace=''):
    key = digest([benchmark, task, canonical])
    sample_id = benchmark + ':' + task + ':' + key[:24]
    return dict(benchmark=benchmark, task=task, split=split, sample_id=sample_id,
                template_path=benchmark + '/' + task, question=question, answer=str(answer),
                prompt=[dict(role='system', content=SYSTEM), dict(role='user', content=question)],
                difficulty_parameters_json=json.dumps(parameters, ensure_ascii=False, sort_keys=True),
                canonical_instance_hash=key, prompt_sha256=digest(question),
                generator_seed=seed, verification_metadata_json=json.dumps(metadata, ensure_ascii=False),
                gold_reasoning_trace=gold_trace,
                extra_info=dict(sample_id=sample_id, benchmark=benchmark, task=task,
                                canonical_instance_hash=key))
