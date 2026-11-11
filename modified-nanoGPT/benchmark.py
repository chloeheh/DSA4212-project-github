"""Measure A10 microbatch throughput at a fixed 32-sequence effective batch."""
import argparse
import gc
import json
import statistics
import time

import numpy as np
import torch

from data_loader import MaestroData
from model import GPT, GPTConfig


def benchmark(data_dir):
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    data = MaestroData(data_dir, 1024)
    report = {'gpu': torch.cuda.get_device_name(), 'effective_batch_size': 32,
              'block_size': 1024, 'trials': {}, 'selected': {}}
    total_memory = torch.cuda.get_device_properties(0).total_memory
    for size, layers, heads, width in [('small', 6, 4, 256), ('medium', 8, 8, 512), ('large', 12, 12, 768)]:
        trials = []
        for microbatch in (4, 8, 16, 32):
            model = optimizer = loss = None
            try:
                torch.manual_seed(1337)
                rng = np.random.default_rng(1337)
                model = GPT(GPTConfig(n_layer=layers, n_head=heads, n_embd=width,
                                      vocab_size=data.vocab_size, block_size=1024,
                                      dropout=0.1, bias=False)).cuda().train()
                optimizer = model.configure_optimizers(0.1, 3e-4, (0.9, 0.95), 'cuda')
                accumulation = 32 // microbatch
                torch.cuda.reset_peak_memory_stats()
                durations = []
                for iteration in range(9):  # three warm-up steps, six measured steps
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    total_loss = torch.zeros((), device='cuda')
                    for _ in range(accumulation):
                        x, y = data.batch('train', microbatch, rng)
                        x = torch.from_numpy(x).pin_memory().cuda(non_blocking=True)
                        y = torch.from_numpy(y).pin_memory().cuda(non_blocking=True)
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            _, loss = model(x, y)
                        total_loss += loss.detach()
                        (loss / accumulation).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if not torch.isfinite(total_loss).item():
                        raise FloatingPointError('Non-finite loss in benchmark')
                    torch.cuda.synchronize()
                    if iteration >= 3:
                        durations.append(time.perf_counter() - started)
                peak = torch.cuda.max_memory_allocated()
                trial = dict(batch_size=microbatch, gradient_accumulation_steps=accumulation,
                             tokens_per_second=32768 / statistics.median(durations),
                             peak_gpu_memory_gb=peak / 1e9, safe_memory=peak < 0.85 * total_memory)
                trials.append(trial)
                print(json.dumps({'size': size, **trial}), flush=True)
            except torch.cuda.OutOfMemoryError:
                trials.append(dict(batch_size=microbatch, error='out_of_memory'))
                print(f'{size}, batch {microbatch}: out of memory', flush=True)
            finally:
                model = optimizer = loss = x = y = total_loss = _ = None
                gc.collect()
                torch.cuda.empty_cache()
        eligible = [trial for trial in trials if trial.get('safe_memory')]
        if not eligible:
            raise RuntimeError(f'No safe batch size for {size}')
        best = max(eligible, key=lambda trial: trial['tokens_per_second'])
        baseline = next(trial for trial in trials if trial['batch_size'] == 4)
        report['trials'][size] = trials
        report['selected'][size] = {**best, 'speedup_over_batch4': best['tokens_per_second'] / baseline['tokens_per_second']}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from pathlib import Path
    result = benchmark(args.data_dir)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result['selected'], indent=2), flush=True)
