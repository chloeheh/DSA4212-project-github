"""Single-GPU MAESTRO training: python modified-nanoGPT/train.py config/maestro_small.py."""
import csv
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
import time

import numpy as np
import torch

from data_loader import MaestroData
from model import GPT, GPTConfig

# All sizes share these settings; size configs change only the architecture and output path.
data_dir = str(Path(__file__).resolve().parent.parent / 'data' / 'maestro')
out_dir = 'out/maestro-small'
init_from = 'scratch'  # scratch or resume
seed = 1337
batch_size = 4
gradient_accumulation_steps = 8
block_size = 1024
n_layer = 6
n_head = 4
n_embd = 256
dropout = 0.1
bias = False
learning_rate = 3e-4
min_lr = 3e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
max_iters = 1_000_000_000
max_time_seconds = 7200.0
warmup_fraction = 0.02
eval_interval_seconds = 600.0
eval_iters = 20
eval_batch_size = 4  # fixed across model sizes, independently of tuned training microbatch
log_interval = 10
eval_only = False
device = 'cuda'
dtype = 'bfloat16'
compile = False  # avoid compilation overhead in a short compute-budget experiment

config_keys = [k for k, v in globals().items()
               if not k.startswith('_') and type(v) in (int, float, bool, str)]


def main(config):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This experiment uses one GPU per model; do not use torchrun.')
    if max_time_seconds <= 0 or not 0 < warmup_fraction < 1:
        raise ValueError('Use a positive time budget and warmup_fraction between 0 and 1.')
    if min(batch_size, gradient_accumulation_steps, eval_iters, eval_batch_size, max_iters, log_interval) < 1:
        raise ValueError('Batch sizes, iteration counts, and log_interval must be positive.')
    if eval_interval_seconds <= 0 or dtype not in ('float32', 'float16', 'bfloat16'):
        raise ValueError('Invalid evaluation interval or dtype.')
    if init_from not in ('scratch', 'resume'):
        raise ValueError('Music models support init_from=scratch or resume.')
    is_cuda = device.startswith('cuda')
    if is_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use Modal, or --device=cpu --dtype=float32 for a smoke test.')
    if is_cuda:
        torch.cuda.set_device(torch.device(device) if ':' in device else 0)
        if dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('This GPU does not support bfloat16; use --dtype=float16.')
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train_rng = np.random.default_rng(seed)
    data = MaestroData(data_dir, block_size)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if init_from == 'scratch' and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty; choose a new output directory or resume.')
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                      block_size=block_size, vocab_size=data.vocab_size, dropout=dropout, bias=bias)
    model = GPT(GPTConfig(**model_args)).to(device)
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), 'cuda' if is_cuda else 'cpu')
    scaler = torch.amp.GradScaler('cuda', enabled=is_cuda and dtype == 'float16')
    step, best_val_loss, elapsed_before = 0, float('inf'), 0.0
    tokens_seen = 0
    if init_from == 'resume':
        # weights_only=False is needed for NumPy RNG metadata; only resume our trusted checkpoints.
        checkpoint = torch.load(output / 'ckpt.pt', map_location='cpu', weights_only=False)
        if checkpoint['model_args'] != model_args:
            raise ValueError('Resume config does not match checkpoint architecture/vocabulary.')
        for key in ('seed', 'batch_size', 'gradient_accumulation_steps', 'max_time_seconds',
                    'learning_rate', 'min_lr', 'warmup_fraction', 'weight_decay', 'beta1', 'beta2'):
            if checkpoint['config'][key] != config[key]:
                raise ValueError(f'Resume must preserve {key}.')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        step = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        elapsed_before = checkpoint['elapsed_seconds']
        tokens_seen = checkpoint['tokens_seen']
        train_rng.bit_generator.state = checkpoint['numpy_rng']
        torch.set_rng_state(checkpoint['torch_rng'])
        if is_cuda:
            torch.cuda.set_rng_state(checkpoint['cuda_rng'])
        del checkpoint
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    num_params = model.get_num_params(non_embedding=False)
    raw_model = model
    if compile:
        model = torch.compile(model)
    amp_dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}[dtype]

    def context():
        return torch.amp.autocast('cuda', dtype=amp_dtype) if is_cuda and dtype != 'float32' else nullcontext()

    def batch(split, rng, size=batch_size):
        x, y = data.batch(split, size, rng)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if is_cuda:
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = {}
        # Recreate generators so each size/step uses exactly the same evaluation windows.
        for split, offset in [('train', 1), ('val', 2)]:
            rng = np.random.default_rng(seed + offset)
            values = []
            for _ in range(eval_iters):
                x, y = batch(split, rng, eval_batch_size)
                with context():
                    _, loss = model(x, y)
                values.append(loss.item())
            losses[split] = float(np.mean(values))
        model.train()
        return losses

    started = time.monotonic()

    def elapsed():
        return elapsed_before + time.monotonic() - started

    def save_checkpoint(name):
        state = dict(model=raw_model.state_dict(), optimizer=optimizer.state_dict(),
                     scaler=scaler.state_dict(), model_args=model_args, config=config,
                     iter_num=step, best_val_loss=best_val_loss, elapsed_seconds=elapsed(),
                     tokens_seen=tokens_seen, numpy_rng=train_rng.bit_generator.state,
                     torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state() if is_cuda else None)
        temporary = output / f'{name}.tmp'
        torch.save(state, temporary)
        temporary.replace(output / name)

    metrics_path = output / 'metrics.csv'
    fields = ['step', 'elapsed_seconds', 'tokens_seen', 'train_loss', 'val_loss', 'val_perplexity', 'learning_rate']
    last_losses = None

    def record_evaluation(lr):
        nonlocal best_val_loss, last_losses
        last_losses = evaluate()
        if not all(math.isfinite(v) for v in last_losses.values()):
            raise FloatingPointError(f'Non-finite evaluation loss: {last_losses}')
        exists = metrics_path.exists()
        with metrics_path.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(dict(step=step, elapsed_seconds=elapsed(), tokens_seen=tokens_seen,
                                 train_loss=last_losses['train'], val_loss=last_losses['val'],
                                 val_perplexity=math.exp(last_losses['val']), learning_rate=lr))
        print(f"step {step}: train {last_losses['train']:.4f}, val {last_losses['val']:.4f}, elapsed {elapsed():.1f}s", flush=True)
        if last_losses['val'] < best_val_loss:
            best_val_loss = last_losses['val']
            save_checkpoint('best.pt')
        save_checkpoint('ckpt.pt')

    # Evaluation and periodic checkpointing count towards the training-loop budget.
    # Initialization and the final evaluation/save are outside this budget.
    lr = 0.0
    record_evaluation(lr)
    next_eval = elapsed() + eval_interval_seconds
    tokens_per_step = batch_size * gradient_accumulation_steps * block_size
    model.train()
    while not eval_only and step < max_iters and elapsed() < max_time_seconds:
        step_started = time.monotonic()
        fraction = min(elapsed() / max_time_seconds, 1.0)
        if fraction < warmup_fraction:
            lr = learning_rate * max(fraction / warmup_fraction, 1e-3)
        else:
            ratio = (fraction - warmup_fraction) / (1 - warmup_fraction)
            lr = min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (learning_rate - min_lr)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        for _ in range(gradient_accumulation_steps):
            x, y = batch('train', train_rng)
            with context():
                _, loss = model(x, y)
            loss_sum += loss.detach()
            scaler.scale(loss / gradient_accumulation_steps).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        # Synchronize once per optimizer step, so elapsed time includes actual GPU work.
        loss_value = loss_sum.item() / gradient_accumulation_steps
        if not math.isfinite(loss_value):
            raise FloatingPointError(f'Non-finite training loss at step {step}')
        step += 1
        tokens_seen += tokens_per_step
        if step % log_interval == 0:
            dt = time.monotonic() - step_started
            print(f'step {step}: loss {loss_value:.4f}, {tokens_per_step / dt:.0f} tokens/s, {elapsed():.1f}s', flush=True)
        if elapsed() >= next_eval and elapsed() < max_time_seconds:
            record_evaluation(lr)
            next_eval = elapsed() + eval_interval_seconds

    budget_elapsed = elapsed()
    if not eval_only:
        record_evaluation(lr)
    summary = dict(parameters=num_params, iterations=step, tokens_seen=tokens_seen,
                   budget_seconds=max_time_seconds, training_loop_seconds=budget_elapsed,
                   total_elapsed_seconds=elapsed(), best_val_loss=best_val_loss,
                   final_train_loss=last_losses['train'], final_val_loss=last_losses['val'],
                   final_val_perplexity=math.exp(last_losses['val']),
                   peak_gpu_memory_gb=torch.cuda.max_memory_allocated() / 1e9 if is_cuda else None,
                   gpu=torch.cuda.get_device_name() if is_cuda else device, model_args=model_args)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    exec((Path(__file__).resolve().parent / 'configurator.py').read_text())
    main({key: globals()[key] for key in config_keys})
