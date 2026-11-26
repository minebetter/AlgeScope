import unittest
from unittest.mock import patch

from algescope.rewards.answer_format import extract_answer
from algescope.rewards.compute import score_record


class AnswerFormatTests(unittest.TestCase):
    def test_single_terminal_answer(self):
        for text in [
            '<answer>42</answer>',
            'The calculation is complete. <answer> 42 </answer>\n',
            '<think>Try a different value.</think><answer>42</answer>',
        ]:
            with self.subTest(text=text):
                self.assertEqual(extract_answer(text), '42')

    def test_generic_terminal_fallbacks(self):
        for text, answer in [
            (r'The result is \boxed{42}.', '42'),
            (r'\boxed{{1, 2}}', '{1, 2}'),
            ('Result:\n```json\n[1, 2]\n```', '[1, 2]'),
        ]:
            with self.subTest(text=text):
                self.assertEqual(extract_answer(text), answer)

    def test_nested_answers_are_rejected(self):
        for text in [
            '<answer>Explanation. <answer>42</answer></answer>',
            '<think>Reasoning.</think><answer>Explanation. <answer>42</answer>',
        ]:
            with self.subTest(text=text):
                self.assertIsNone(extract_answer(text))

    def test_answer_content_is_not_reinterpreted(self):
        for content in [r'Explanation. \boxed{42}', '```json\n[1, 2]\n```']:
            with self.subTest(content=content):
                self.assertEqual(extract_answer('<answer>' + content + '</answer>'), content)

    def test_ambiguous_or_incomplete_answers_are_rejected(self):
        for text in [
            '<answer>42</answer><answer>43</answer>',
            '<answer>42</answer>More text.',
            '<answer>42',
            '<think><answer>42</answer></think>No final answer.',
            '<think>One.</think><think>Two.</think><answer>42</answer>',
            r'\boxed{42} or \boxed{43}',
            '```json\n42\n```\n```json\n43\n```',
        ]:
            with self.subTest(text=text):
                self.assertIsNone(extract_answer(text))

    def test_reward_uses_only_the_completed_final_answer(self):
        row = {'benchmark': 'synlogic', 'task': 'operation', 'answer': '42',
               'verification_metadata_json': '{}'}
        cases = [
            ('<answer>42</answer>', 'stop', (1.0, 'correct')),
            ('<think><answer>42</answer></think><answer>41</answer>',
             'stop', (0.0, 'incorrect')),
            ('<answer>Explanation. <answer>42</answer></answer>',
             'stop', (0.0, 'invalid_final_answer')),
            ('<think>Reasoning.</think><answer>Explanation. <answer>42</answer>',
             'stop', (0.0, 'invalid_final_answer')),
            (r'<answer>Explanation. \boxed{42}</answer>', 'stop', (0.0, 'incorrect')),
            ('<answer>42</answer>', 'length', (0.0, 'unfinished')),
        ]
        with patch.dict('os.environ', {'SYNLOGIC_SOURCE': ''}):
            for text, finish_reason, expected in cases:
                with self.subTest(text=text, finish_reason=finish_reason):
                    self.assertEqual(score_record(row, text, finish_reason), expected)


if __name__ == '__main__':
    unittest.main()
