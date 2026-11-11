"""Run with: modal run --detach modified-nanoGPT/modal_train.py --run-id maestro-v1"""
from pathlib import Path
import re

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VOLUME_NAME = 'maestro-nanogpt-results'
app = modal.App('maestro-nanogpt')
results = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
# Pin training dependencies so all sizes run in the same environment.
image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.6.0', 'numpy==2.2.3')
    .env({'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4'})
    .add_local_dir(HERE, remote_path='/workspace/modified-nanoGPT',
                   ignore=['__pycache__', '*.pyc', 'out', '.DS_Store'])
)
# Upload tokenized inputs only; raw MIDI and the held-out test set are unnecessary for training.
for filename in ('train.bin', 'val.bin', 'meta.pkl', 'piece_offsets.json', 'summary.json'):
    image = image.add_local_file(ROOT / 'data' / 'maestro' / filename,
                                 remote_path=f'/workspace/data/maestro/{filename}')


def validate_options(run_id, seconds):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        raise ValueError('run_id must be 1–80 letters, digits, underscores, or hyphens.')
    if not 0 < seconds <= 7200:
        raise ValueError('seconds must be in (0, 7200].')


@app.function(image=image, gpu='A10', cpu=4, memory=16384,
              timeout=3 * 60 * 60, retries=0, volumes={'/results': results})
def train_model(size: str, run_id: str, seconds: float = 7200.0,
                smoke: bool = False, resume: bool = False):
    import json
    import subprocess
    import sys
    import torch

    validate_options(run_id, seconds)
    if size not in ('small', 'medium', 'large'):
        raise ValueError('Unknown model size')
    print(f'GPU: {torch.cuda.get_device_name(0)}; size: {size}; budget: {seconds}s', flush=True)
    results.reload()
    output = Path('/results') / run_id / size
    if resume and (output / 'summary.json').is_file():
        previous = json.loads((output / 'summary.json').read_text())
        if previous['budget_seconds'] != seconds:
            raise ValueError('Resume must use the original time budget.')
        if previous['training_loop_seconds'] >= seconds or (smoke and previous['iterations'] >= 2):
            print(f'{size} already completed; skipping.', flush=True)
            previous['size'] = size
            return previous
    resume_checkpoint = resume and (output / 'ckpt.pt').is_file()
    command = [sys.executable, '-u', '/workspace/modified-nanoGPT/train.py',
               f'/workspace/modified-nanoGPT/config/maestro_{size}.py',
               '--data_dir=/workspace/data/maestro', f'--out_dir={output}',
               f'--max_time_seconds={float(seconds)}']
    tuning_path = Path('/results') / run_id / 'tuning.json'
    if tuning_path.is_file():
        tuning = json.loads(tuning_path.read_text())['selected'][size]
        command.extend([f"--batch_size={tuning['batch_size']}",
                        f"--gradient_accumulation_steps={tuning['gradient_accumulation_steps']}"])
    if resume_checkpoint:
        command.append('--init_from=resume')
    if smoke:
        # Use the real architecture, sequence length, and batch size to check A10 memory.
        command.extend(['--max_iters=2', '--eval_iters=2', '--log_interval=1'])
    output.mkdir(parents=True, exist_ok=True)
    if not resume_checkpoint and any(output.iterdir()):
        raise FileExistsError(f'{output} already contains a run; use a new run-id or --resume.')
    # Periodically publish completed, atomically-renamed checkpoints to persistent storage.
    # stdout/stderr are inherited so training progress appears in Modal logs.
    process = subprocess.Popen(command, cwd='/workspace')
    try:
        while process.poll() is None:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                results.commit()
        if process.returncode:
            raise RuntimeError(f'{size} training exited with code {process.returncode}')
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        results.commit()
    summary = json.loads((output / 'summary.json').read_text())
    summary['size'] = size
    return summary


@app.function(image=image, gpu='A10', cpu=4, memory=16384,
              timeout=1800, retries=0, volumes={'/results': results})
def tune_batches(run_id: str):
    import json
    import subprocess
    import sys

    validate_options(run_id, 7200.0)
    results.reload()
    folder = Path('/results') / run_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'tuning.json'
    if path.is_file():
        return json.loads(path.read_text())
    subprocess.run([sys.executable, '-u', '/workspace/modified-nanoGPT/benchmark.py',
                    '--data-dir=/workspace/data/maestro', f'--output={path}'],
                   cwd='/workspace', check=True)
    results.commit()
    return json.loads(path.read_text())


# A CPU coordinator keeps the sequence alive when the local terminal disconnects.
@app.function(image=image, timeout=9 * 60 * 60, retries=0,
              volumes={'/results': results})
def run_experiment(run_id: str, seconds: float = 7200.0, smoke: bool = False,
                   resume: bool = False):
    import csv
    import json

    validate_options(run_id, seconds)
    results.reload()
    tuning_path = Path('/results') / run_id / 'tuning.json'
    if not smoke:
        tuning = (json.loads(tuning_path.read_text()) if tuning_path.is_file()
                  else tune_batches.spawn(run_id).get())
        print('Selected A10 microbatches:', tuning['selected'], flush=True)
    summaries = []
    for size in ('small', 'medium', 'large'):
        # Reconcile completed runs on the CPU without allocating another GPU.
        results.reload()
        summary_path = Path('/results') / run_id / size / 'summary.json'
        summary = None
        if resume and summary_path.is_file():
            previous = json.loads(summary_path.read_text())
            if previous['budget_seconds'] != seconds:
                raise ValueError('Resume must use the original time budget.')
            if previous['training_loop_seconds'] >= seconds or (smoke and previous['iterations'] >= 2):
                summary = {**previous, 'size': size}
                print(f'{size} already completed; using saved results.', flush=True)
        if summary is None:
            summary = train_model.spawn(size, run_id, seconds, smoke, resume).get()
        summaries.append(summary)
        results.reload()
        folder = Path('/results') / run_id
        (folder / 'comparison.json').write_text(json.dumps(summaries, indent=2))
        fields = ['size', 'parameters', 'iterations', 'tokens_seen', 'training_loop_seconds',
                  'best_val_loss', 'final_val_loss', 'final_val_perplexity', 'peak_gpu_memory_gb']
        with (folder / 'comparison.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(summaries)
        results.commit()
    print(json.dumps(summaries, indent=2), flush=True)
    return summaries


@app.local_entrypoint()
def main(run_id: str = '', seconds: float = 7200.0, smoke: bool = False,
         resume: bool = False):
    from datetime import datetime, timezone
    if not run_id:
        run_id = datetime.now(timezone.utc).strftime('maestro-%Y%m%d-%H%M%S')
    validate_options(run_id, seconds)
    print(f'Run: {run_id}; sequential A10 jobs; {seconds}s/model; smoke={smoke}')
    print(f'Results: Modal Volume {VOLUME_NAME}, /{run_id}/')
    call = run_experiment.spawn(run_id, seconds, smoke, resume)
    print(f'Coordinator call: {call.object_id}', flush=True)
    call.get()
