"""Boundary regression checks, independent of PyTorch and the full dataset."""
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_loader import MaestroData


class PieceAwareSamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        with (self.root / 'meta.pkl').open('wb') as f:
            pickle.dump({'vocab_size': 32}, f)
        # One too-short piece followed by pieces with one and four valid windows.
        pieces = [{'start': 0, 'length': 3}, {'start': 3, 'length': 5},
                  {'start': 8, 'length': 8}]
        self.manifest = {'train': pieces, 'validation': pieces}
        (self.root / 'piece_offsets.json').write_text(json.dumps(self.manifest))
        for split in ('train', 'val'):
            np.arange(16, dtype=np.uint16).tofile(self.root / f'{split}.bin')

    def test_only_valid_windows_and_all_endpoints_reachable(self):
        loader = MaestroData(self.root, block_size=4)
        for split in ('train', 'val'):
            starts = loader.sample_starts(split, 10000, np.random.default_rng(7))
            self.assertEqual(set(starts), {3, 8, 9, 10, 11})
            x, y = loader.batch(split, 100, np.random.default_rng(8))
            np.testing.assert_array_equal(x + 1, y)
            self.assertEqual(x.shape, (100, 4))

    def test_evaluation_generator_does_not_advance_training_generator(self):
        loader = MaestroData(self.root, block_size=4)
        training = np.random.default_rng(1337)
        control = np.random.default_rng(1337)
        loader.batch('val', 20, np.random.default_rng(1339))
        np.testing.assert_array_equal(loader.batch('train', 20, training)[0],
                                      loader.batch('train', 20, control)[0])

    def test_rejects_mismatched_offsets(self):
        self.manifest['train'][1]['start'] = 4
        (self.root / 'piece_offsets.json').write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, 'offsets'):
            MaestroData(self.root, block_size=4)

    def test_rejects_token_outside_vocabulary(self):
        np.full(16, 32, dtype=np.uint16).tofile(self.root / 'train.bin')
        with self.assertRaisesRegex(ValueError, 'vocabulary'):
            MaestroData(self.root, block_size=4)

    def test_rejects_context_larger_than_every_piece(self):
        with self.assertRaisesRegex(ValueError, 'no piece'):
            MaestroData(self.root, block_size=16)


if __name__ == '__main__':
    unittest.main()
