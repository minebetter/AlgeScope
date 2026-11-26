"""Final-answer extraction used by the SynLogic experiments."""
import re


def _terminal_answer(body):
    """Extract one closed, non-nested final answer without interpreting its content."""
    body = body.strip()
    if '<answer>' not in body and '</answer>' not in body:
        if body.count('\\boxed{') == 1:
            start=body.index('\\boxed{')+len('\\boxed{');depth=1
            for i in range(start,len(body)):
                depth+=(body[i]=='{')-(body[i]=='}')
                if depth==0:
                    if re.fullmatch(r'(?:[\s.$]|\\[\]\)])*',body[i+1:]):return body[start:i].strip()
                    return None
        fences=list(re.finditer(r'```(?:python|json)?\s*\n(.*?)\n```',body,re.S))
        if len(fences)==1 and not body[fences[0].end():].strip():
            return fences[0].group(1).strip()
        return None
    matches = list(re.finditer(r'<answer>(.*?)</answer>', body, re.S))
    if len(matches) != 1 or body.count('<answer>') != 1 or body.count('</answer>') != 1:
        return None
    match = matches[0]
    if body[match.end():].strip():
        return None
    return match.group(1).strip()


def extract_answer(text):
    """Reject multi-turn continuation; inspect only the single final response."""
    if text.count('<think>') > 1 or text.count('</think>') > 1:
        return None
    body = text.split('</think>')[-1].strip()
    return _terminal_answer(body)
