"""Full-coverage, piece-aware evaluation of a trusted nanoGPT checkpoint."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
import time

import numpy as np


def piece_windows(pieces, block_size):
    """Yield (piece index, input start, target count), scoring each next token once."""
    if block_size < 1:
        raise ValueError('block_size must be positive')
    cursor = 0
    for index, piece in enumerate(pieces):
        length = piece['length']
        if piece['start'] != cursor or length < 2:
            raise ValueError('Invalid or non-contiguous piece offsets')
        for offset in range(0, length - 1, block_size):
            yield index, cursor + offset, min(block_size, length - 1 - offset)
        cursor += length


def event_family(token):
    for family in ('NOTE_ON', 'NOTE_OFF', 'TIME_SHIFT', 'VELOCITY'):
        if token.startswith(family + '_'):
            return family
    return 'PEDAL' if token in ('PEDAL_ON', 'PEDAL_OFF') else token


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoint(checkpoint_path, data_dir, output_dir, device='cuda', batch_size=16):
    import torch
    from torch.nn import functional as F
    from contextlib import nullcontext
    from model import GPT, GPTConfig

    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    started = time.monotonic()
    checkpoint_path, data_dir, output_dir = map(Path, (checkpoint_path, data_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / 'meta.pkl').open('rb') as f:
        meta = pickle.load(f)
    # Only load project-generated metadata/checkpoints; these contain trusted pickle data.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    args = checkpoint['model_args']
    if args['vocab_size'] != meta['vocab_size']:
        raise ValueError('Checkpoint vocabulary does not match tokenized data')
    model = GPT(GPTConfig(**args))
    state = {key.removeprefix('_orig_mod.'): value for key, value in checkpoint['model'].items()}
    model.load_state_dict(state)
    provenance = dict(model_args=args, selected_step=checkpoint['iter_num'],
                      selected_elapsed_seconds=checkpoint.get('elapsed_seconds'),
                      selection_val_loss=checkpoint['best_val_loss'],
                      tokens_seen_in_training=checkpoint.get('tokens_seen'))
    del state, checkpoint
    is_cuda = device.startswith('cuda')
    if is_cuda:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Evaluation requires a BF16 CUDA GPU, or device=cpu for a local check')
        torch.cuda.set_device(torch.device(device) if ':' in device else 0)
        torch.backends.cuda.matmul.allow_tf32 = True
    model = model.to(device).eval()
    manifest = json.loads((data_dir / 'piece_offsets.json').read_text())
    digest = hashlib.sha256()
    with checkpoint_path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    report = dict(checkpoint=str(checkpoint_path), checkpoint_sha256=digest.hexdigest(),
                  **provenance, splits={}, gpu=torch.cuda.get_device_name() if is_cuda else device)
    for split, filename in [('validation', 'val.bin'), ('test', 'test.bin')]:
        pieces = manifest[split]
        path = data_dir / filename
        if path.stat().st_size % 2:
            raise ValueError(f'{filename}: partial uint16 token')
        data = np.memmap(path, dtype=np.uint16, mode='r')
        if sum(p['length'] for p in pieces) != len(data) or int(data.max()) >= meta['vocab_size']:
            raise ValueError(f'{split}: binary/metadata mismatch or invalid token IDs')
        windows = list(piece_windows(pieces, args['block_size']))
        piece_nll = np.zeros(len(pieces), dtype=np.float64)
        piece_counts = np.zeros(len(pieces), dtype=np.int64)
        token_nll = np.zeros(meta['vocab_size'], dtype=np.float64)
        token_counts = np.zeros(meta['vocab_size'], dtype=np.int64)
        with torch.inference_mode():
            for first in range(0, len(windows), batch_size):
                batch = windows[first:first + batch_size]
                width = max(length for _, _, length in batch)
                x = np.zeros((len(batch), width), dtype=np.int64)
                y = np.full_like(x, -1)
                for row, (_, start, length) in enumerate(batch):
                    x[row, :length] = data[start:start + length]
                    y[row, :length] = data[start + 1:start + length + 1]
                inputs = torch.from_numpy(x).to(device)
                context = torch.amp.autocast('cuda', dtype=torch.bfloat16) if is_cuda else nullcontext()
                with context:
                    logits, _ = model(inputs, torch.from_numpy(y).to(device))
                # Float32 CE; retain individual losses for exact token weighting and diagnostics.
                losses = F.cross_entropy(logits.float().transpose(1, 2), torch.from_numpy(y).to(device),
                                         reduction='none', ignore_index=-1).cpu().numpy()
                for row, (piece_id, _, length) in enumerate(batch):
                    values = losses[row, :length].astype(np.float64)
                    targets = y[row, :length]
                    piece_nll[piece_id] += values.sum()
                    piece_counts[piece_id] += length
                    token_nll += np.bincount(targets, weights=values, minlength=meta['vocab_size'])
                    token_counts += np.bincount(targets, minlength=meta['vocab_size'])
                if first % (batch_size * 100) == 0:
                    print(f'{split}: evaluated {min(first + batch_size, len(windows))}/{len(windows)} windows', flush=True)
        expected = sum(p['length'] - 1 for p in pieces)
        if int(piece_counts.sum()) != expected or int(token_counts.sum()) != expected:
            raise AssertionError('Not all eligible targets were scored exactly once')
        rows = [dict(midi_filename=p['midi_filename'], scored_tokens=int(piece_counts[i]),
                     cross_entropy=float(piece_nll[i] / piece_counts[i]),
                     perplexity=math.exp(piece_nll[i] / piece_counts[i])) for i, p in enumerate(pieces)]
        write_csv(output_dir / f'{split}_per_piece.csv', rows)
        families = {}
        for token_id in range(meta['vocab_size']):
            token = meta['itos'][token_id]
            family = event_family(token)
            totals = families.setdefault(family, [0, 0.0])
            totals[0] += int(token_counts[token_id])
            totals[1] += float(token_nll[token_id])
        family_rows = [dict(event_family=family, scored_tokens=count, cross_entropy=nll / count,
                            perplexity=math.exp(nll / count))
                       for family, (count, nll) in families.items() if count]
        write_csv(output_dir / f'{split}_by_event.csv', family_rows)
        loss = float(piece_nll.sum() / expected)
        report['splits'][split] = dict(cross_entropy=loss, perplexity=math.exp(loss), scored_tokens=expected,
                                      pieces=len(pieces), windows=len(windows))
        print(split, json.dumps(report['splits'][split]), flush=True)
    report['evaluation_seconds'] = time.monotonic() - started
    temporary = output_dir / 'evaluation.json.tmp'
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(output_dir / 'evaluation.json')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', default=16, type=int)
    options = parser.parse_args()
    evaluate_checkpoint(options.checkpoint, options.data_dir, options.output_dir, options.device, options.batch_size)
