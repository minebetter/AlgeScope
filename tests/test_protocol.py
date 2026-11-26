import json
import unittest

from algescope.rewards.compute import score_record, ordered_answer
from algescope.training.summarize import summarize
from algescope.training.trace import prepare_score_inputs


class ProtocolTests(unittest.TestCase):
    def test_semantic_reward_and_termination(self):
        row = {'benchmark': 'deductive', 'template_path': 'caesar_cipher_Zn.yaml',
               'question': "('Shift_3' ∘ x) = 'Shift_8', x = ?", 'answer': "x = 'Shift_5'"}
        self.assertEqual(score_record(row, "\\boxed{x='Shift_5'}", 'stop')[0], 1)
        self.assertEqual(score_record(row, "\\boxed{x='Shift_6'}", 'stop')[0], 0)
        self.assertEqual(score_record(row, "\\boxed{x='Shift_5'}", 'length')[0], 0)
        self.assertEqual(score_record(row, "<think>\\boxed{x='Shift_5'}", 'stop')[0], 0)

    def test_noncommuting_reference_order(self):
        row = {'benchmark': 'deductive', 'template_path': 'rubiks_cube_group.yaml',
               'question': "'R' >> 'U' = ?", 'answer': "'R U'"}
        self.assertEqual(score_record(row, "\\boxed{'R U'}", 'stop')[0], 1)
        self.assertEqual(score_record(row, "\\boxed{'U R'}", 'stop')[0], 0)
        self.assertIsNone(ordered_answer("x='R', x='U'"))

    def test_summary_coverage_and_passk(self):
        expected = {name: {'template_path': 'toy', 'difficulty': 3 if name == 'a' else 8,
                          'difficulty_field': 'num_elements'} for name in ['a', 'b']}
        rows = [{'sample_id': name, 'draw': draw, 'score': score, 'finish_reason': 'stop'}
                for name, scores in [('a', [0, 1]), ('b', [1, 1])] for draw, score in enumerate(scores)]
        result = summarize(rows, expected)
        total = next(r for r in result if r['template'] == 'ALL' and r['band'] == 'ALL')
        self.assertEqual(total['average_at_k'], .75)
        self.assertEqual(total['pass_at_k'], 1)
        with self.assertRaises(ValueError):
            summarize(rows[:2], expected)
        with self.assertRaises(ValueError):
            summarize(rows + [rows[0]], expected)

    def test_trace_uses_real_length_not_padding(self):
        class Tensor(list):
            def __getitem__(self, key):
                value = super().__getitem__(key)
                return Tensor(value) if isinstance(key, slice) else value
            def cpu(self): return self
            def tolist(self): return list(self)
        class Tokenizer:
            eos_token_id = 2
            pad_token_id = 0
            def decode(self, ids, skip_special_tokens=False):
                return '|'.join(map(str, ids))
        text, info = prepare_score_inputs(Tokenizer(), Tensor([8, 9]), Tensor([4, 2, 0, 0]),
            {'algescope_trace': json.dumps({'finish_reason': 'stop', 'stop_reason': None, 'response_token_count': 2})})
        self.assertEqual(text, '4')
        self.assertEqual(json.loads(info['_trace_record']['response_token_ids']), [4, 2])
        with self.assertRaises(ValueError):
            prepare_score_inputs(Tokenizer(), Tensor([8]), Tensor([4]), {})


if __name__ == '__main__':
    unittest.main()
