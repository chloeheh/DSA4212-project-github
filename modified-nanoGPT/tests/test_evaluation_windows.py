"""Ensure final evaluation scores every eligible next token once without piece leakage."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import piece_windows


class EvaluationWindowTests(unittest.TestCase):
    def test_short_pieces_and_tail_targets_are_included_exactly_once(self):
        pieces = [{'start': 0, 'length': 3}, {'start': 3, 'length': 10}]
        windows = list(piece_windows(pieces, 4))
        actual = []
        for piece_id, start, count in windows:
            piece = pieces[piece_id]
            self.assertGreaterEqual(start, piece['start'])
            self.assertLess(start + count, piece['start'] + piece['length'])
            actual.extend(range(start + 1, start + count + 1))
        self.assertEqual(actual, [1, 2, *range(4, 13)])
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(windows[-1][-1], 1)

    def test_invalid_piece_offsets_rejected(self):
        with self.assertRaises(ValueError):
            list(piece_windows([{'start': 1, 'length': 4}], 2))

    def test_context_must_be_positive(self):
        with self.assertRaises(ValueError):
            list(piece_windows([], 0))


if __name__ == '__main__':
    unittest.main()
