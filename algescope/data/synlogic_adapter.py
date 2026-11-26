"""SynLogic fresh-12: source generators, explicit final-answer contracts, safe judges."""
import ast
import contextlib
from functools import lru_cache
import importlib
import json
import math
import operator
import os
from pathlib import Path
import random
import re
import signal
import sys

from .common import numeric, record

TASKS = ['operation', 'dyck_language', 'dyck_language_errors', 'dyck_language_reasoning_errors',
         'web_of_lies', 'object_properties', 'goods_exchange', 'space_reasoning',
         'space_reasoning_tree', 'sudoku', 'cryptarithm', 'arrow_maze']
CLASSES = dict(zip(TASKS, ['Operation', 'DyckLanguage', 'DyckLanguageErrors',
    'DyckLanguageReasoningErrors', 'WebOfLies', 'ObjectProperties', 'GoodsExchange',
    'SpaceReasoning', 'SpaceReasoningTree', 'Sudoku', 'Cryptarithm', 'ArrowMaze']))
PAIRS = dict(zip(')]}>', '([{<'))


class GeneratorDeadline(BaseException):
    """Bypass broad Exception catches inside the upstream expression generator."""


def safe_generated_expression(text):
    """Replay generated arithmetic without eval, bounding costly exponentiation."""
    tree=ast.parse(text.replace('Abs(', 'abs('),mode='eval')
    if sum(1 for _ in ast.walk(tree))>150: raise ValueError('generated_expression_too_large')
    ops={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,
         ast.Mod:operator.mod,ast.Pow:operator.pow}
    def visit(node):
        if isinstance(node,ast.Constant) and type(node.value) in [int,float]: result=node.value
        elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
            result=visit(node.operand)*(-1 if isinstance(node.op,ast.USub) else 1)
        elif isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='abs' and len(node.args)==1 and not node.keywords:
            result=abs(visit(node.args[0]))
        elif isinstance(node,ast.BinOp) and type(node.op) in ops:
            a,b=visit(node.left),visit(node.right)
            if isinstance(node.op,ast.Pow) and abs(b)>32: raise ValueError('generated_exponent_too_large')
            result=ops[type(node.op)](a,b)
        else: raise ValueError('unsupported_generated_expression')
        if type(result) not in [int,float] or not math.isfinite(result) or abs(result)>1e30:
            raise ValueError('generated_intermediate_too_large')
        return result
    return visit(tree.body)


def setup(root):
    root = str(Path(root).resolve())
    if root not in sys.path: sys.path.insert(0, root)


def first_error(sequence):
    stack = []
    for i, char in enumerate(sequence, 1):
        if char in PAIRS.values(): stack.append(char)
        elif not stack or stack.pop() != PAIRS[char]: return i
    return len(sequence) + 1 if stack else -1


def thoughts_errors(metadata):
    stack, errors = [], []
    for i, char in enumerate(metadata['dyck_sequence']):
        if char in PAIRS.values(): stack.append(char)
        else:
            assert stack.pop() == PAIRS[char]
        thought = metadata['thoughts'][i + 2]
        claimed = thought.split(';')[-1].split(':', 1)[-1].strip()
        if claimed in ['空', 'empty']: claimed = ''
        if claimed != ''.join(stack): errors.append(i + 3)
    end = metadata['thoughts'][-1]
    if '不为空' in end or 'not empty' in end: errors.append(len(metadata['thoughts']))
    return errors


def literal(text):
    text = text.strip()
    if len(text) > 20000: raise ValueError('Answer too large')
    text = re.sub(r'^```(?:python|json)?\s*|\s*```$', '', text).strip()
    return ast.literal_eval(text)


def sudoku_valid(grid, puzzle):
    if not isinstance(grid, (list, tuple)) or len(grid) != 9: return False
    if not all(isinstance(row, (list, tuple)) and len(row) == 9 for row in grid): return False
    if not all(type(x) is int and 1 <= x <= 9 for row in grid for x in row): return False
    target = set(range(1, 10))
    if any(set(row) != target for row in grid): return False
    if any({grid[r][c] for r in range(9)} != target for c in range(9)): return False
    if any({grid[r+i][c+j] for i in range(3) for j in range(3)} != target
           for r in (0,3,6) for c in (0,3,6)): return False
    return all(puzzle[r][c] == 'X' or puzzle[r][c] == grid[r][c] for r in range(9) for c in range(9))


def cryptarithm_valid(answer, metadata):
    answer = re.sub(r'\s+', '', answer)
    if not re.fullmatch(r'\d+(?:[+*-]\d+)+=\d+', answer): return False
    numbers = re.findall(r'\d+', answer)
    ops = re.findall(r'[+*-]', answer)
    words = metadata['letter_words']
    if ops != metadata['operators'] or len(numbers) != len(words): return False
    mapping = {}
    for word, number in zip(words, numbers):
        if len(word) != len(number) or len(number)>1 and number[0]=='0': return False
        for letter, digit in zip(word, number):
            if letter in mapping and mapping[letter] != digit: return False
            mapping[letter] = digit
    if len(set(mapping.values())) != len(mapping): return False
    left, right = answer.split('=')
    return numeric(left) == numeric(right)


@lru_cache(None)
def arrow_verifier():
    from games.tasks.arrow_maze.scripts.arrow_maze_verifier import ArrowMazeVerifier
    return ArrowMazeVerifier()


def judge(task, answer, gold, metadata):
    """Judge only the extracted final answer. Never eval model-generated code."""
    if len(answer) > 20000: return False
    try:
        if task=='operation':
            a,b=numeric(answer),numeric(gold)
            return abs(float(a-b)) <= 1e-9*max(1,abs(float(b)))
        if task in ['object_properties', 'dyck_language_errors']:
            return numeric(answer) == numeric(gold)
        if task == 'dyck_language':
            text = ''.join(answer.split()).strip('"\'')
            return text == metadata['full_sequence'] and first_error(text) == -1
        if task == 'dyck_language_reasoning_errors':
            text = answer.strip().strip('"')
            if not re.fullmatch(r'\d*(?:\s*,\s*\d+)*', text): return False
            values = [] if not text else [int(s) for s in text.split(',')]
            return len(set(values)) == len(values) and set(values) == set(metadata['error_indices'])
        if task == 'web_of_lies':
            values = [v.strip().lower() for v in answer.split(',')]
            expected = [v.strip().lower() for v in gold.split(',')]
            mapping = {'yes':True,'true':True,'no':False,'false':False}
            return [mapping[v] for v in values] == [mapping[v] for v in expected]
        if task == 'goods_exchange':
            pairs = literal(answer)
            if not isinstance(pairs,(tuple,list)) or len(pairs) != len(metadata['names']): return False
            if not all(isinstance(p,(list,tuple)) and len(p)==2 and all(isinstance(x,str) for x in p) for p in pairs): return False
            if len(set(p[0] for p in pairs)) != len(pairs): return False
            return dict(pairs) == dict(literal(gold))
        if task in ['space_reasoning', 'space_reasoning_tree']:
            values = [x.strip() for x in answer.split(',')]
            return len(set(values)) == len(values) and set(values) == set(x.strip() for x in gold.split(','))
        if task == 'sudoku': return sudoku_valid(literal(answer), metadata['original_sudoku'])
        if task == 'cryptarithm': return cryptarithm_valid(answer, metadata)
        if task == 'arrow_maze':
            grid = literal(answer)
            if not isinstance(grid,(list,tuple)): return False
            from base.data import Data
            # The source arrow verifier only JSON-parses and checks constraints.
            with open(os.devnull,'w') as quiet, contextlib.redirect_stdout(quiet):
                return bool(arrow_verifier().verify(Data('',gold,metadata=metadata),json.dumps(grid)))
    except (ValueError, TypeError, KeyError, SyntaxError, ZeroDivisionError, OverflowError, IndexError):
        return False
    raise ValueError('Unregistered task ' + task)


def specifications(task, split, rng):
    warm = split == 'warmup'
    count = {'num_of_questions':1,'max_attempts':100}
    if task == 'operation':
        ns = rng.choice([1,2] if warm else [1,2,3])
        return dict(num_symbols_range=(ns,ns),max_operands=4 if warm else 5,
                    definition_num_symbols_max=3 if warm else 4,language='en'),dict(count,max_attempts=1)
    if task.startswith('dyck_language'):
        params = dict(count,n_types=rng.choice([3,4]),total_length=rng.choice([20,24] if warm else [24,32,40]))
        if task == 'dyck_language': params['to_fill_length'] = rng.choice([4,6])
        if task == 'dyck_language_reasoning_errors': params['n_errors'] = rng.choice([0,1,2] if warm else [0,1,2,3])
        return {},params
    if task == 'web_of_lies': return {},dict(count,num_person=rng.choice([6,8] if warm else [8,10,12]),difficulty=3,statement_type=0)
    if task == 'object_properties': return dict(num_range=[10,15],transformation_range=[2,3] if warm else [2,4],language_type='english'),count
    if task == 'goods_exchange': return {},dict(count,num_people=rng.choice([6,8]),operator_num=rng.choice([6,8] if warm else [8,12,16]))
    if task == 'space_reasoning': return dict(n=8),dict(num_of_data=1,num_waypoints=rng.choice([3,5] if warm else [3,5,7]),unknown_node_num=1,language='en')
    if task == 'space_reasoning_tree': return dict(min_nodes=50,max_nodes=75 if warm else 100),dict(num_of_data=1,language='en')
    if task == 'sudoku': return {},dict(count,difficulty=1 if warm else rng.choice([2,3]),unique_solution=True)
    if task == 'cryptarithm': return {},dict(count,num_letter=rng.choice([3,4]),operator_num=rng.choice([1,2]),operator_level=2)
    if task == 'arrow_maze':
        size=6 if warm else rng.choice([6,7,8])
        return {},dict(count,width=size,height=size,arrow_fill_rate_min=.3,arrow_fill_rate_max=.5)
    raise ValueError(task)


def replace_names(text, names):
    # Replace simultaneously, so a token generated by replacement cannot be replaced again.
    names = list(dict.fromkeys(names))
    mapping = {name:f'ENTITY_{i}' for i,name in enumerate(names)}
    if not names: return text
    return re.sub('|'.join(re.escape(x) for x in sorted(names,key=len,reverse=True)),
                  lambda m:mapping[m.group()],text)


def canonical(task, question, metadata):
    m=metadata
    if task=='dyck_language': return m['question_sequence']
    if task=='dyck_language_errors': return m['bracket_string']
    if task=='dyck_language_reasoning_errors': return [m['dyck_sequence'],m['thoughts']]
    if task=='sudoku':
        # Digit-renamed copies stay in one split.
        mapping={}; result=[]
        for row in m['original_sudoku']:
            for v in row:
                if v=='X': result.append('X')
                else:
                    mapping.setdefault(v,len(mapping)+1); result.append(mapping[v])
        return result
    if task=='arrow_maze': return m['maze']
    if task=='cryptarithm':
        letters={}
        words=[]
        for word in m['letter_words']:
            ids=[]
            for char in word:
                letters.setdefault(char,len(letters)); ids.append(letters[char])
            words.append(ids)
        return [words,m['operators']]
    if task=='operation':
        text=json.dumps([m['expression'],m['symbol_definitions']],ensure_ascii=False)
        return replace_names(text,list(m['symbol_definitions']))
    if task=='goods_exchange':
        return replace_names(json.dumps([m['owns_before'],m['operations']],ensure_ascii=False),m['names']+m['objects'])
    if task=='web_of_lies':
        return replace_names(question,[x[k] for x in m['people'] for k in ['name','location']])
    if task=='space_reasoning_tree':
        return {'parents':m['audit_tree']['parents'],'target':m['audit_tree']['target']}
    if task=='space_reasoning':
        return replace_names(question,m['audit_graph']['items'])
    if task=='object_properties': return [m['story'],m['question']]
    raise ValueError(task)


def normalize_prompt(task, row):
    m=row.metadata
    if task=='dyck_language':
        return ('Complete this bracket sequence by appending the shortest possible suffix. '
                'Return the entire completed sequence, including the given prefix.\nPrefix: '+m['question_sequence'])
    if task=='dyck_language_errors':
        return ('Find the first invalid bracket position (1-based). If only unclosed brackets remain '
                'at the end, answer length+1. If the sequence is valid, answer -1.\nSequence: '+m['bracket_string'])
    if task=='dyck_language_reasoning_errors':
        thoughts=[s.replace('我们应该逐个处理输入并跟踪栈的配置。','Process characters in order and track the stack.')
                    .replace('现在，我们已经到达结尾。最终栈是不为空的。','At the end the stack is not empty.')
                    .replace('现在，我们已经到达结尾。最终栈是空的。','At the end the stack is empty.')
                    .replace('栈: ','Stack: ').replace('空','empty') for s in m['thoughts']]
        m['thoughts']=thoughts
        return ('Inspect the proposed stack trace for this valid bracket sequence. Judge each displayed '
                'stack against the correctly processed prefix, independently of previous displayed errors. '
                'List all incorrect Thought numbers separated by commas; answer an empty string if none.\n'
                +'Sequence: '+m['dyck_sequence']+'\n'+'\n'.join(thoughts))
    if task=='sudoku':
        return ('Complete this standard 9x9 Sudoku (X denotes an empty cell). Each row, column and '
                '3x3 box must contain digits 1–9 exactly once. Preserve all given clues. Return the '
                'complete grid as a Python list or tuple of nine rows.\n'+json.dumps(m['original_sudoku']))
    if task=='cryptarithm':
        words=m['letter_words']; ops=m['operators']
        equation=' '.join(sum(([w,o] for w,o in zip(words,ops)),[])+[words[-2],'=',words[-1]])
        return ('Each letter represents one digit, different letters represent different digits, and '
                'leading digits cannot be zero. Replace letters with digits to give a valid numerical '
                'equation. Standard arithmetic precedence applies.\n'+equation)
    question=row.question
    # Retain substantive task instructions; remove only the trailing original answer-format request.
    markers={
      'operation':[', and put your numerical answer in'],
      'goods_exchange':[' Please output',' Please provide','\nPlease output',' Please end',' Please use'],
      'web_of_lies':['Please think step by step','Please analyze step by step','Please end your response','Think step by step, end your response'],
      'arrow_maze':['\nAdditional requirement:'],
      'space_reasoning':['Your answer must be inside','Your answer should be'],
      'space_reasoning_tree':['Your answer should be in'],
      'object_properties':['Your answer should be in','Your answer must be inside','Give your final answer as:'],
    }
    for marker in markers.get(task,[]):
        if marker in question: question=question.split(marker,1)[0].rstrip()
    # Explicitly replace residual boxed syntax with the common final-answer container.
    question=re.sub(r'\\boxed\{\}', '<answer>...</answer>',question)
    return question


def generate(task, split, seed):
    rng=random.Random(seed)
    init,params=specifications(task,split,rng)
    module=importlib.import_module(f'games.tasks.{task}.scripts.{task}')
    game=getattr(module,CLASSES[task])(**init)
    captured={}
    if task=='operation':
        def bounded(expression, definitions):
            simplified=game._simplify_mix_expression(expression,definitions)
            result=safe_generated_expression(simplified)
            return int(result) if isinstance(result,float) and result.is_integer() else result,simplified
        game._evaluate_expression=bounded
    if task=='space_reasoning_tree':
        original=game.build_five_layer_tree
        def build(*args,**kwargs):
            result=original(*args,**kwargs);captured['nodes']=result[1];return result
        game.build_five_layer_tree=build
        find=game.find_valid_cousin_target
        def target(*args,**kwargs):
            result=find(*args,**kwargs);captured['target']=result;return result
        game.find_valid_cousin_target=target
    if task=='space_reasoning':
        original=game.generate_question
        def query(nodes,start_idx,end_idx,seen_node,language='en'):
            result=original(nodes,start_idx,end_idx,seen_node,language)
            captured.update(nodes=nodes,start=start_idx,end=end_idx,seen=seen_node)
            return result
        game.generate_question=query
    if task=='operation':
        def deadline(signum,frame): raise GeneratorDeadline('operation_cpu_deadline')
        signal.signal(signal.SIGVTALRM,deadline)
        signal.setitimer(signal.ITIMER_VIRTUAL,1)
    try: rows=game.generate(**params)
    except GeneratorDeadline: raise TimeoutError('operation_cpu_deadline') from None
    finally:
        if task=='operation': signal.setitimer(signal.ITIMER_VIRTUAL,0)
    if not rows: raise ValueError('generator_returned_no_rows')
    row=rows[0]; m=row.metadata
    m.pop('trace_id',None)
    if task=='space_reasoning_tree':
        nodes=captured['nodes']; ids={n:i for i,n in enumerate(nodes)}
        m['audit_tree']=dict(items=[n.item for n in nodes],parents=[ids.get(n.parent,-1) for n in nodes],target=ids[captured['target']])
        parents=m['audit_tree']['parents']; t=m['audit_tree']['target']; parent=parents[t]; grandparent=parents[parent]
        cousins=[nodes[i].item for i,p in enumerate(parents) if p>=0 and p!=parent and parents[p]==grandparent]
        assert set(row.answer.split(', '))==set(cousins),'tree_answer_mismatch'
    if task=='space_reasoning':
        nodes=captured['nodes'];ids={n:i for i,n in enumerate(nodes)}
        m['audit_graph']=dict(items=[n.item for n in nodes],
            connections=[{k:ids[v] for k,v in n.connections.items()} for n in nodes],
            start=captured['start'],end=captured['end'],seen=captured['seen'])
        assert row.answer==nodes[captured['end']].item,'graph_answer_mismatch'
    if task=='dyck_language_errors': assert int(row.answer)==first_error(m['bracket_string']),'bracket_label_mismatch'
    if task=='dyck_language_reasoning_errors': assert set(m['error_indices'])==set(thoughts_errors(m)),'trace_error_label_mismatch'
    question=normalize_prompt(task,row)
    assert not re.search(r'[\u4e00-\u9fff]',question),'non_english_prompt'
    gold=str(row.answer)
    assert judge(task,gold,gold,m),'gold_rejected_by_safe_judge'
    assert not judge(task,'__INVALID_ANSWER__',gold,m),'invalid_answer_accepted'
    native_input='\\boxed{'+gold+'}' if task in ['operation','object_properties','space_reasoning','space_reasoning_tree'] else gold
    if task=='goods_exchange': native_input='```python\n'+gold+'\n```'
    assert game.verify(row,native_input),'gold_rejected_by_upstream_verifier'
    m['upstream_gold_roundtrip']=True
    return record('synlogic',task,split,seed,question,gold,m,canonical(task,question,m),
                  dict(initialization=init,generation=params))
