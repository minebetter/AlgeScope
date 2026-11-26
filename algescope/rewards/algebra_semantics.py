"""Interpret stated algebra rules independently of benchmark solvers and math-verify.

28 templates are supported. D4 (orientation conventions) and Rubik (inconsistent
canonicalization examples) are deliberately left unclassified.
Only explicit AST nodes are interpreted; no evaluation of arbitrary source code.
"""
import ast, operator, re
from decimal import Decimal

def scalar(s):
    s=s.strip().strip('$').strip()
    s=s.replace('\\_', '_').replace('\\,','').replace('\\!','')
    s=re.sub(r'\\(?:text|mathrm|mathtt)\{([^{}]*)\}',r'\1',s)
    if len(s)>=2 and s[0]==s[-1] and s[0] in "'\"":s=s[1:-1]
    return s.strip()
def vector(s,n=None):
    v=tuple(map(int,s.strip('()').split(',')))
    if n is not None and len(v)!=n:raise ValueError('wrong vector dimension')
    return v
def reduce_word(s,pairs):
    result=[]
    for c in s:
        if c not in pairs:raise ValueError('invalid word character')
        if result and result[-1]==pairs[c]:result.pop()
        else:result.append(c)
    return ''.join(result)
def permutation(s):
    if not re.fullmatch(r'(?:\([012 ]*\))+',s):raise ValueError('invalid cycle syntax')
    result=list(range(3))
    for match in reversed(re.findall(r'\(([^()]*)\)',s)):
        cycle=list(map(int,match.split()))
        if len(cycle)!=len(set(cycle)):raise ValueError('duplicate cycle element')
        perm=dict(zip(cycle,cycle[1:]+cycle[:1]))
        result=[perm.get(x,x) for x in result]
    return tuple(result)
def algebra(template):
    t=template.split('.')[0]
    identity=lambda a:a
    if t.startswith('unix_permission'):
        def atom(s):
            if not re.fullmatch('[r-][w-][x-]',s):raise ValueError('invalid permission mask')
            return sum(b for c,b in zip(s,[4,2,1]) if c!='-')
        return atom,operator.xor,identity
    if t.startswith(('xor_scrambler','ui_feature')):
        chars='01' if t.startswith('xor') else 'FT'
        def atom(s):
            if len(s)!=8 or set(s)-set(chars):raise ValueError('invalid bit string')
            return int(s.translate(str.maketrans(chars,'01')),2)
        return atom,operator.xor,identity
    if t.startswith('rgb_'):
        def atom(s):
            v=vector(s,3)
            if any(x<0 or x>255 for x in v):raise ValueError('invalid RGB channel')
            return v
        return atom,lambda a,b:tuple(x^y for x,y in zip(a,b)),identity
    if t.startswith(('social_network','css_class')):
        return lambda s:frozenset(s.split(',')) if s else frozenset(),operator.xor,identity
    if t.startswith('debate_'):
        return lambda s:{'KEEP':0,'FLIP':1}[s],operator.xor,identity
    for prefix,alphabet,inverses in [('knitting','kpKP','KPkp'),('robot_path','FBLR','BFRL'),('sliding','UDLR','DURL')]:
        if t.startswith(prefix):
            pairs=dict(zip(alphabet,inverses))
            return lambda s:reduce_word(s,pairs),lambda a,b:reduce_word(a+b,pairs),lambda a:''.join(pairs[c] for c in a[::-1])
    if t.startswith('tree_rotation'):
        def atom(s):
            if s and not re.fullmatch('[LR](;[LR])*',s):raise ValueError('invalid tree rotation')
            return s.count('R')-s.count('L')
        return atom,operator.add,operator.neg
    if t.startswith('browser_'):
        def atom(s):
            if not s:return 0
            m=re.fullmatch(r'(FWD|BACK|STAY)\((\d+)\)',s)
            if not m:raise ValueError('invalid browser operation')
            return int(m[2])*{'FWD':1,'BACK':-1,'STAY':0}[m[1]]
        return atom,operator.add,operator.neg
    if t.startswith('ledger_'):
        def atom(s):
            if s=='NO_OP_0':return Decimal(0)
            m=re.fullmatch(r'(DEPOSIT|WITHDRAW)_([0-9]+(?:\.[0-9]+)?)',s)
            if not m:raise ValueError('invalid transaction')
            return Decimal(m[2])*(1 if m[1]=='DEPOSIT' else -1)
        return atom,operator.add,operator.neg
    if t.startswith('chess_'):
        return lambda s:vector(s,2),lambda a,b:tuple(x+y for x,y in zip(a,b)),lambda a:tuple(-x for x in a)
    for prefix,mod,name in [('caesar',26,'Shift_'),('fsm_',4,'Shift_'),('card_',52,'cut_'),('musical_',12,'UP_'),('timezone',24,'UTC')]:
        if t.startswith(prefix):
            def atom(s):
                if not s.startswith(name):raise ValueError('invalid modular element')
                return int(s[len(name):])%mod
            return atom,lambda a,b:(a+b)%mod,lambda a:-a%mod
    if t.startswith('cube_top'):
        return lambda s:{'ID':0,'U':1,'U2':2,'U_prime':3}[s],lambda a,b:(a+b)%4,lambda a:-a%4
    if t.startswith(('enigma','vigenere')):
        n=3 if t.startswith('enigma') else 5
        def atom(s):
            s=s.replace(',','')
            if len(s)!=n or not re.fullmatch('[A-Z]+',s):raise ValueError('invalid cipher state')
            return tuple(ord(c)-65 for c in s)
        return atom,lambda a,b:tuple((x+y)%26 for x,y in zip(a,b)),lambda a:tuple(-x%26 for x in a)
    if t.startswith(('coffee','poetry')):
        keys=['espresso','foam','milk','sugar','water'] if t.startswith('coffee') else ['long','short','stressed','unstressed']
        def atom(s):
            pairs=[part.split(':') for part in s.split(',')] if s else []
            d={k:int(v) for k,v in pairs}
            if set(d)-set(keys):raise ValueError('unknown component')
            return tuple(d.get(k,0) for k in keys)
        return atom,lambda a,b:tuple(x+y for x,y in zip(a,b)),lambda a:tuple(-x for x in a)
    if t.startswith('cron_'):
        keys=['M','H','d','m','w'];mods=[60,24,31,12,7]
        def atom(s):
            parts=s.split('_') if s else []
            if any(not re.fullmatch(r'[MHdmw][+-]\d+',x) for x in parts):raise ValueError('invalid cron field')
            d={x[0]:int(x[1:]) for x in parts}
            return tuple(d.get(k,0)%m for k,m in zip(keys,mods))
        return atom,lambda a,b:tuple((x+y)%m for x,y,m in zip(a,b,mods)),lambda a:tuple(-x%m for x,m in zip(a,mods))
    if t.startswith('hearthstone'):
        keys=['ATK','HP','DIVINE_SHIELD','POISONOUS','TAUNT']
        def atom(s):
            vals=dict.fromkeys(keys,0)
            for part in s.split(';') if s else []:
                if part.endswith('_TOGGLE'):vals[part[:-7]]^=1
                else:
                    m=re.fullmatch(r'(ATK|HP)([+-]\d+)',part)
                    if not m:raise ValueError('invalid buff')
                    vals[m[1]]+=int(m[2])
            return tuple(vals[k] for k in keys)
        return atom,lambda a,b:tuple(x+y if i<2 else x^y for i,(x,y) in enumerate(zip(a,b))),lambda a:tuple(-x if i<2 else x for i,x in enumerate(a))
    if t.startswith('symmetric_'):
        return permutation,lambda a,b:tuple(a[x] for x in b),lambda a:tuple(a.index(x) for x in range(3))
    if t.startswith('affine_'):
        def atom(s):
            a,b=vector(s,2);pow(a,-1,26);return a%26,b%26
        return atom,lambda a,b:((a[0]*b[0])%26,(a[0]*b[1]+a[1])%26),lambda a:(pow(a[0],-1,26),-pow(a[0],-1,26)*a[1]%26)
    raise NotImplementedError('unsupported template: '+t)

def parse(expression):
    chunks=re.split(r"('[^']*')",expression.strip())
    for i in range(0,len(chunks),2):
        chunks[i]=re.sub(r'\b[01]{8}\b',lambda m:repr(m.group()),chunks[i])
        chunks[i]=re.sub(r'>>|->|\$|⊎|🎨|∘|⊕|@|#|&|\b(?:sync|then|chmod|toggleClass|ปรับ)\b','+',chunks[i])
    return ast.parse(''.join(chunks),mode='eval').body
def execute(node,env,ops):
    atom,op,inv=ops
    if isinstance(node,ast.Constant) and isinstance(node.value,str):return atom(node.value)
    if isinstance(node,ast.Name) and node.id in env:return atom(env[node.id])
    if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):return op(execute(node.left,env,ops),execute(node.right,env,ops))
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='inv' and len(node.args)==1:return inv(execute(node.args[0],env,ops))
    raise ValueError('unsupported expression syntax')
def check(r,answer):
    if answer is None:return {'status':'no_boxed'}
    try:
        ops=algebra(r['template_path'])
        q=r['question'];lines=q.splitlines()[1:] if q.startswith('Solve the following') else q.splitlines()
        eqs=[]
        for line in lines:
            line=re.sub(r',\s*x\s*=\s*\?\s*$','',line.strip())
            a,b=line.rsplit('=',1);eqs.append((parse(a),None if b.strip()=='?' else parse(b)))
        variables=sorted({n.id for a,b in eqs for node in [a,b] if node is not None for n in ast.walk(node)
                          if isinstance(n,ast.Name) and n.id in ['α','β','x']})
        if variables:
            text=answer.replace('\\alpha','α').replace('\\beta','β')
            parts=re.split(r',\s*(?=[αβx]\s*=)',text)
            env={}
            for part in parts:
                m=re.fullmatch(r'\s*([αβx])\s*=\s*(.*?)\s*',part)
                if not m:raise ValueError('unparsed assignment')
                env[m[1]]=scalar(m[2])
            if set(env)!=set(variables):raise ValueError('missing assignments')
            checks=[execute(a,env,ops)==execute(b,env,ops) for a,b in eqs]
        else:checks=[execute(eqs[0][0],{},ops)==ops[0](scalar(answer))]
        return {'status':'correct' if all(checks) else 'incorrect','equations':checks}
    except NotImplementedError as e:return {'status':'unsupported','reason':str(e)}
    except Exception as e:return {'status':'unparsed','reason':type(e).__name__+': '+str(e)}
