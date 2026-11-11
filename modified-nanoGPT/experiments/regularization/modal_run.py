"""Launch three controlled dropout ablations, two hours per size on one A10 at a time.

From the repository root:
modal run --detach modified-nanoGPT/experiments/regularization/modal_run.py
"""
import json
from pathlib import Path
import runpy
import sys

import modal

# Modal imports this launcher as /root/modal_run.py; training code is mounted separately.
TRAINING_ROOT = (Path('/workspace/modified-nanoGPT') if Path('/workspace/modified-nanoGPT').is_dir()
                 else Path(__file__).resolve().parents[2])
sys.path.insert(0, str(TRAINING_ROOT))
from modal_train import image, results, validate_options, VOLUME_NAME

app = modal.App('maestro-regularization')
BASELINE_RUN = 'maestro-20261104-optimized-v1'
DEFAULT_RUN = 'maestro-dropout02-20261106-v1'
SIZES = ('small', 'medium', 'large')
CONFIG_DIR = Path('/workspace/modified-nanoGPT/experiments/regularization')


def validate_ablation(baseline, candidate):
    """Reject accidental changes to the controlled baseline before training."""
    differences = {key for key, value in baseline.items()
                   if candidate.get(key) != value and key != 'out_dir'}
    if differences != {'dropout'} or baseline['dropout'] != 0.1 or candidate['dropout'] != 0.2:
        raise ValueError(f'Only dropout 0.1 -> 0.2 may change; found {sorted(differences)}')


@app.function(image=image, gpu='A10', cpu=4, memory=16384,
              timeout=10800, retries=0, scaledown_window=2,
              volumes={'/results': results})
def train_regularized(size: str, run_id: str, resume: bool = False):
    import subprocess

    validate_options(run_id, 7200.0)
    if size not in SIZES or run_id == BASELINE_RUN:
        raise ValueError('Choose a valid model size and a separate experiment run ID.')
    results.reload()
    baseline_dir = Path('/results') / BASELINE_RUN / size
    baseline_config = json.loads((baseline_dir / 'config.json').read_text())
    baseline_summary = json.loads((baseline_dir / 'summary.json').read_text())
    config_path = CONFIG_DIR / f'{size}_dropout02.py'
    settings = runpy.run_path(str(config_path))
    validate_ablation(baseline_config, settings)
    if baseline_summary['training_loop_seconds'] < 7200:
        raise ValueError('The baseline must be a completed two-hour run.')
    folder = Path('/results') / run_id
    output = folder / size
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / 'summary.json'
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if summary['training_loop_seconds'] >= 7200 and summary['model_args']['dropout'] == 0.2:
            print('This experiment is already complete; no additional training.', flush=True)
            return summary
    if not resume and any(output.iterdir()):
        raise FileExistsError('Run already has outputs. Use --resume to continue its original budget.')
    resume_checkpoint = resume and (output / 'ckpt.pt').is_file()
    if resume and any(output.iterdir()) and not resume_checkpoint:
        raise FileNotFoundError('Existing outputs have no checkpoint to resume.')
    manifest = {
        'experiment': 'model_size_dropout_ablation',
        'baseline_run': BASELINE_RUN,
        'model': size,
        'change': {'dropout': {'baseline': 0.1, 'experiment': 0.2}},
        'budget_seconds': 7200,
        'selection_metric': 'best validation next-token cross-entropy (lower is better)',
        'test_split_used': False,
        'note': 'One seed and a fixed time budget; token exposure can differ with throughput.',
    }
    (folder / f'{size}_experiment.json').write_text(json.dumps(manifest, indent=2))
    results.commit()
    command = [sys.executable, '-u', '/workspace/modified-nanoGPT/train.py',
               str(config_path), f'--out_dir={output}']
    if resume_checkpoint:
        command.append('--init_from=resume')
    print(f'A10 run: {size}, dropout=0.2, 7200-second budget, no batching sweep.', flush=True)
    process = subprocess.Popen(command, cwd='/workspace')
    try:
        while process.poll() is None:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                results.commit()
        if process.returncode:
            raise RuntimeError(f'Training exited with code {process.returncode}; inspect saved checkpoint/logs.')
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        results.commit()
    summary = json.loads(summary_path.read_text())
    comparison = {
        'baseline': baseline_summary,
        'dropout_02': summary,
        'best_val_loss_change': summary['best_val_loss'] - baseline_summary['best_val_loss'],
        'final_val_loss_change': summary['final_val_loss'] - baseline_summary['final_val_loss'],
        'interpretation': 'Negative loss change favors dropout=0.2; use best validation loss as primary metric.',
    }
    (folder / f'{size}_baseline_comparison.json').write_text(json.dumps(comparison, indent=2))
    results.commit()
    print(json.dumps(comparison, indent=2), flush=True)
    return summary


@app.function(image=image, timeout=9 * 3600, retries=0, volumes={'/results': results})
def run_study(run_id: str, resume: bool = False):
    """CPU coordinator preserves the three-run sequence after local disconnection."""
    import csv

    validate_options(run_id, 7200.0)
    if run_id == BASELINE_RUN:
        raise ValueError('The dropout experiment must not overwrite the baseline.')
    results.reload()
    folder = Path('/results') / run_id
    # Verify all three baseline/config pairs before spending GPU time on any size.
    for size in SIZES:
        baseline = json.loads((Path('/results') / BASELINE_RUN / size / 'config.json').read_text())
        validate_ablation(baseline, runpy.run_path(str(CONFIG_DIR / f'{size}_dropout02.py')))
    rows = []
    for size in SIZES:
        results.reload()
        summary_path = folder / size / 'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
        if summary is not None:
            if summary['model_args']['dropout'] != 0.2 or summary['budget_seconds'] != 7200:
                raise ValueError(f'{size}: existing outputs are from another experiment.')
            if summary['training_loop_seconds'] < 7200:
                summary = None
        if summary is None:
            summary = train_regularized.spawn(size, run_id, resume).get()
        else:
            print(f'{size}: complete; reusing saved results without GPU training.', flush=True)
        results.reload()
        baseline = json.loads((Path('/results') / BASELINE_RUN / size / 'summary.json').read_text())
        rows.append({
            'model': size,
            'baseline_best_val_loss': baseline['best_val_loss'],
            'dropout02_best_val_loss': summary['best_val_loss'],
            'best_val_loss_change': summary['best_val_loss'] - baseline['best_val_loss'],
            'baseline_final_val_loss': baseline['final_val_loss'],
            'dropout02_final_val_loss': summary['final_val_loss'],
            'tokens_seen': summary['tokens_seen'],
            'training_loop_seconds': summary['training_loop_seconds'],
        })
        (folder / 'comparison.json').write_text(json.dumps(rows, indent=2))
        with (folder / 'comparison.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        results.commit()
    print(json.dumps(rows, indent=2), flush=True)
    return rows


@app.local_entrypoint()
def main(run_id: str = DEFAULT_RUN, resume: bool = False):
    validate_options(run_id, 7200.0)
    if run_id == BASELINE_RUN:
        raise ValueError('Choose a separate experiment run ID.')
    for size in SIZES:
        local_baseline = TRAINING_ROOT.parent / 'results' / BASELINE_RUN / size / 'config.json'
        candidate = runpy.run_path(str(Path(__file__).with_name(f'{size}_dropout02.py')))
        validate_ablation(json.loads(local_baseline.read_text()), candidate)
    print(f'Three sequential A10 runs, 7200 seconds each; {VOLUME_NAME}/{run_id}/', flush=True)
    call = run_study.spawn(run_id, resume)
    print(f'Coordinator call: {call.object_id}', flush=True)
    call.get()
