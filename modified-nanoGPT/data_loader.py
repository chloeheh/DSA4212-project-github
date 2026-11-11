"""Sample next-token windows without crossing MAESTRO piece boundaries."""
import json
import pickle
from pathlib import Path

import numpy as np


class MaestroData:
    def __init__(self, data_dir, block_size):
        self.root = Path(data_dir)
        self.block_size = block_size
        if block_size < 1:
            raise ValueError('block_size must be positive')
        # Only load metadata produced by our own tokenizer (pickle is not safe for untrusted files).
        with (self.root / 'meta.pkl').open('rb') as f:
            self.meta = pickle.load(f)
        self.vocab_size = int(self.meta['vocab_size'])
        manifest = json.loads((self.root / 'piece_offsets.json').read_text())
        self.arrays, self.starts, self.cumulative = {}, {}, {}
        for split, manifest_key in [('train', 'train'), ('val', 'validation')]:
            path = self.root / f'{split}.bin'
            if path.stat().st_size % np.dtype(np.uint16).itemsize:
                raise ValueError(f'{path}: incomplete uint16 token')
            data = np.memmap(path, dtype=np.uint16, mode='r')
            pieces = manifest[manifest_key]
            cursor = 0
            for piece in pieces:
                if piece['start'] != cursor or piece['length'] < 2:
                    raise ValueError(f'{split}: invalid piece offsets')
                cursor += piece['length']
            if cursor != len(data):
                raise ValueError(f'{split}: offsets do not match binary file length')
            for offset in range(0, len(data), 1_000_000):
                if int(data[offset:offset + 1_000_000].max()) >= self.vocab_size:
                    raise ValueError(f'{split}: token outside vocabulary')
            eligible = [p for p in pieces if p['length'] > block_size]
            if not eligible:
                raise ValueError(f'{split}: no piece has {block_size + 1} tokens')
            self.arrays[split] = data
            self.starts[split] = np.array([p['start'] for p in eligible], dtype=np.int64)
            # A piece of length L has L - block_size valid next-token windows.
            self.cumulative[split] = np.cumsum(
                [p['length'] - block_size for p in eligible], dtype=np.int64)

    def sample_starts(self, split, batch_size, rng):
        cumulative = self.cumulative[split]
        draws = rng.integers(0, cumulative[-1], size=batch_size)
        pieces = np.searchsorted(cumulative, draws, side='right')
        previous = np.where(pieces == 0, 0, cumulative[np.maximum(pieces - 1, 0)])
        return self.starts[split][pieces] + draws - previous

    def batch(self, split, batch_size, rng):
        starts = self.sample_starts(split, batch_size, rng)
        data, size = self.arrays[split], self.block_size
        # Gather each window once, including its final target token.
        windows = data[starts[:, None] + np.arange(size + 1)].astype(np.int64)
        x = np.ascontiguousarray(windows[:, :-1])
        y = np.ascontiguousarray(windows[:, 1:])
        return x, y
