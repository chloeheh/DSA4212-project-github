"""Check saved evidence and report tables without datasets, checkpoints or a GPU.

--data additionally fingerprints excluded raw/prepared data against the audited
local copy. --checkpoints DIR verifies selected best.pt files under DIR/<run>/<size>.
Does not train, download, write files, or claim independent experimental replication.
"""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'maestro-20261104-optimized-v1'
FOLLOWUP = 'maestro-dropout02-20261106-v1'
EVALUATION = 'maestro-dropout02-best-test-20261107-v1'
SIZES = ('small', 'medium', 'large')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def close(actual, expected, context):
    require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8),
            f'{context}: {actual} != {expected}')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def report_row(section, size, values):
    text = (ROOT / 'report/sections' / section).read_text()
    text = re.sub(r'\\textbf\{([^{}]+)\}', r'\1', text)
    expected = size.capitalize() + ' & ' + ' & '.join(values)
    require(expected in text, f'Report table disagrees with evidence: {expected}')


def verify(checkpoints=None):
    source = ROOT / 'results'
    require(load(ROOT / 'modified-nanoGPT/experiments/evaluation/protocol.json') ==
            load(source / EVALUATION / 'protocol.json'), 'Code and saved evaluation protocols differ')
    data_text = (ROOT / 'report/sections/01_data.tex').read_text()
    for number in ('962', '137', '177', '33,293,600', '3,790,187', '4,334,269'):
        require(number in data_text, f'Report data counts missing/changed: {number}')
    summaries = {}
    for run in (BASELINE, FOLLOWUP):
        summaries[run] = {}
        for size in SIZES:
            folder = source / run / size
            summary = load(folder / 'summary.json')
            config = load(folder / 'config.json')
            metrics = rows(folder / 'metrics.csv')
            # Check the recorded minimum and last evaluation, not just copied summaries.
            close(min(float(row['val_loss']) for row in metrics), summary['best_val_loss'], f'{run}/{size} minimum')
            close(float(metrics[-1]['val_loss']), summary['final_val_loss'], f'{run}/{size} final validation')
            close(float(metrics[-1]['train_loss']), summary['final_train_loss'], f'{run}/{size} final train')
            require(summary['tokens_seen'] == summary['iterations'] * 32768, f'{run}/{size}: exposure')
            require(summary['training_loop_seconds'] >= 7200, f'{run}/{size}: unfinished run')
            require(summary['model_args']['dropout'] == config['dropout'], f'{run}/{size}: dropout')
            summaries[run][size] = summary
            if run == FOLLOWUP:
                baseline = load(source / BASELINE / size / 'config.json')
                differences = {key for key in baseline.keys() | config.keys()
                               if baseline.get(key) != config.get(key)}
                require(differences <= {'dropout', 'out_dir'} and config['dropout'] == 0.2,
                        f'{size}: uncontrolled follow-up configuration changes: {differences}')
    tuning = load(source / BASELINE / 'tuning.json')
    for size in SIZES:
        selected = tuning['selected'][size]
        trials = tuning['trials'][size]
        best_trial = max((trial for trial in trials if trial['safe_memory']),
                         key=lambda trial: trial['tokens_per_second'])
        require(selected['batch_size'] == best_trial['batch_size'] == 32, f'{size}: benchmark choice')
        batch4 = next(trial for trial in trials if trial['batch_size'] == 4)
        close(selected['speedup_over_batch4'], selected['tokens_per_second'] / batch4['tokens_per_second'],
              f'{size}: benchmark speedup')
        report_row('04_compute_efficiency.tex', size, [f"{batch4['tokens_per_second']:,.0f}",
                   f"{selected['tokens_per_second']:,.0f}",
                   f"${selected['speedup_over_batch4']:.2f}\\times$", f"{selected['peak_gpu_memory_gb']:.2f}"])
        baseline = summaries[BASELINE][size]
        report_row('04_compute_efficiency.tex', size, [f"{baseline['iterations']:,}",
                   f"{baseline['tokens_seen']/1e9:.3f}", f"{baseline['tokens_seen']/33293600:.1f}"])
        args = baseline['model_args']
        report_row('02_question_models_metrics.tex', size, [str(args['n_layer']), str(args['n_head']),
                   str(args['n_embd']), f"{baseline['parameters']:,}"])
    comparison = rows(source / EVALUATION / 'comparison.csv')
    require([row['model'] for row in comparison] == list(SIZES), 'Evaluation model identities/order')
    reference = {}
    for row in comparison:
        size = row['model']
        evaluation = load(source / EVALUATION / size / 'evaluation.json')
        run = BASELINE if size == 'small' else FOLLOWUP
        close(evaluation['selection_val_loss'], summaries[run][size]['best_val_loss'], f'{size}: selection')
        require(row['source_run'] == run and run in evaluation['checkpoint'], f'{size}: checkpoint source')
        require(evaluation['selected_step'] == int(row['selected_step']), f'{size}: selected step')
        if checkpoints:
            checkpoint = checkpoints / run / size / 'best.pt'
            require(sha256(checkpoint) == evaluation['checkpoint_sha256'], f'{checkpoint}: checksum mismatch')
        for split, count, targets, key in [('validation', 137, 3790050, 'full_validation_loss'),
                                            ('test', 177, 4334092, 'test_loss')]:
            folder = source / EVALUATION / size
            pieces = rows(folder / f'{split}_per_piece.csv')
            identities = {p['midi_filename']: int(p['scored_tokens']) for p in pieces}
            require(len(identities) == len(pieces) == count, f'{size}/{split}: piece identities')
            if size == 'small':
                reference[split] = identities
            require(identities == reference[split], f'{size}/{split}: mismatched pieces/counts')
            for suffix in ('per_piece', 'by_event'):
                entries = rows(folder / f'{split}_{suffix}.csv')
                total = sum(int(p['scored_tokens']) for p in entries)
                require(total == targets, f'{size}/{split}/{suffix}: coverage')
                loss = sum(int(p['scored_tokens']) * float(p['cross_entropy']) for p in entries) / total
                close(loss, float(row[key]), f'{size}/{split}/{suffix}: aggregate')
                close(loss, evaluation['splits'][split]['cross_entropy'], f'{size}/{split}: JSON aggregate')
                close(math.exp(loss), evaluation['splits'][split]['perplexity'], f'{size}/{split}: perplexity')
        baseline = summaries[BASELINE][size]
        followup = summaries[FOLLOWUP][size]
        metric_rows = rows(source / BASELINE / size / 'metrics.csv')
        best = min(metric_rows, key=lambda x: float(x['val_loss']))
        report_row('05_baseline_results.tex', size, [f"{baseline['best_val_loss']:.4f}",
                   f"{float(best['elapsed_seconds']) / 60:.1f}", f"{baseline['final_train_loss']:.4f}",
                   f"{baseline['final_val_loss']:.4f}"])
        report_row('08_regularization_results.tex', size, [f"{baseline['best_val_loss']:.4f}",
                   f"{followup['best_val_loss']:.4f}", f"${followup['best_val_loss']-baseline['best_val_loss']:+.4f}$",
                   f"{baseline['final_val_loss']:.4f}", f"{followup['final_val_loss']:.4f}"])
        report_row('09_final_evaluation.tex', size, [row['dropout'], f"{float(row['full_validation_loss']):.4f}",
                   f"{float(row['test_loss']):.4f}", f"{float(row['test_perplexity']):.4f}"])
    print('PASS: training summaries, controlled dropout changes, validation/test coverage, losses, benchmark/exposure, and report result tables')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', action='store_true')
    parser.add_argument('--checkpoints', type=Path)
    args = parser.parse_args()
    try:
        verify(args.checkpoints)
        if args.data:
            for name, expected in load(ROOT / 'docs/data_manifest.json')['files'].items():
                path = ROOT / name
                require(path.stat().st_size == expected['bytes'] and sha256(path) == expected['sha256'],
                        f'{name}: data fingerprint mismatch')
            print('PASS: all raw MIDI/metadata and prepared binary/offset/summary fingerprints')
        else:
            print('SKIP: excluded data (use --data after preparation)')
        if not args.checkpoints:
            print('SKIP: original model weights (use --checkpoints DIR after retrieval)')
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'FAIL: {error}\n')


if __name__ == '__main__':
    main()
