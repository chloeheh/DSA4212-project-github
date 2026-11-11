"""Evaluate validation-selected checkpoints; explicitly replace only the old small entry."""
import json
from pathlib import Path
import sys

import modal

TRAINING_ROOT = (Path('/workspace/modified-nanoGPT') if Path('/workspace/modified-nanoGPT').is_dir()
                 else Path(__file__).resolve().parents[2])
sys.path.insert(0, str(TRAINING_ROOT))
from modal_train import image as training_image, results, validate_options

app = modal.App('maestro-final-evaluation')
image = training_image.add_local_file(TRAINING_ROOT.parent / 'data/maestro/test.bin',
                                      remote_path='/workspace/data/maestro/test.bin')
# Preserve the existing evaluation ID; its historical name does not specify current selections.
DEFAULT_RUN = 'maestro-dropout02-best-test-20261107-v1'


@app.function(image=image, gpu='A10', cpu=4, memory=16384, timeout=3600,
              retries=0, scaledown_window=2, volumes={'/results': results})
def run_evaluation(run_id: str, replace_small: bool = False):
    import csv
    import shutil
    import subprocess

    validate_options(run_id, 7200.0)
    results.reload()
    protocol = json.loads((TRAINING_ROOT / 'experiments/evaluation/protocol.json').read_text())
    sources = protocol['sources']
    if run_id in {item['source_run'] for item in sources.values()}:
        raise ValueError('Evaluation cannot overwrite a training run')
    for size, selection in sources.items():
        source = Path('/results') / selection['source_run'] / size
        config = json.loads((source / 'config.json').read_text())
        if config['dropout'] != selection['dropout'] or not (source / 'best.pt').is_file():
            raise ValueError(f'{size}: missing or incorrect selected checkpoint')
    output = Path('/results') / run_id
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / 'protocol.json'
    if protocol_path.is_file() and json.loads(protocol_path.read_text()) != protocol:
        old = json.loads(protocol_path.read_text())
        if not replace_small or old.get('dropout') != 0.2 or old.get('source_run') != sources['medium']['source_run']:
            raise ValueError('Protocol differs. Use --replace-small only to replace the original all-0.2 small entry.')
        archive = output / 'history' / 'original_all_dropout02'
        archive.mkdir(parents=True, exist_ok=True)
        for name in ('protocol.json', 'comparison.json', 'comparison.csv'):
            path = output / name
            if path.is_file():
                shutil.copy2(path, archive / name)
        if (output / 'small').is_dir():
            if (archive / 'small').exists():
                raise FileExistsError('Small-result archive exists; inspect before replacing')
            (output / 'small').rename(archive / 'small')
        print('Archived original small dropout 0.2 result; medium/large will be reused.', flush=True)
    protocol_path.write_text(json.dumps(protocol, indent=2))
    results.commit()
    comparison = []
    for size in protocol['models']:
        selection = sources[size]
        checkpoint = Path('/results') / selection['source_run'] / size / 'best.pt'
        destination = output / size
        report_path = destination / 'evaluation.json'
        if not report_path.is_file():
            print(f'Evaluating {size}: dropout {selection["dropout"]}, best.pt; no training.', flush=True)
            subprocess.run([sys.executable, '-u', str(TRAINING_ROOT / 'evaluate.py'),
                            f'--checkpoint={checkpoint}', '--data-dir=/workspace/data/maestro',
                            f'--output-dir={destination}'], cwd='/workspace', check=True)
        else:
            print(f'{size}: reusing existing inference results.', flush=True)
        report = json.loads(report_path.read_text())
        if report['model_args']['dropout'] != selection['dropout'] or report['checkpoint'] != str(checkpoint):
            raise ValueError('Saved evaluation belongs to another checkpoint')
        val, test = report['splits']['validation'], report['splits']['test']
        comparison.append(dict(model=size, dropout=selection['dropout'], source_run=selection['source_run'],
                               checkpoint='best.pt', selected_step=report['selected_step'],
                               selected_training_minutes=report['selected_elapsed_seconds'] / 60,
                               selection_sampled_val_loss=report['selection_val_loss'],
                               full_validation_loss=val['cross_entropy'], validation_perplexity=val['perplexity'],
                               test_loss=test['cross_entropy'], test_perplexity=test['perplexity'],
                               validation_scored_tokens=val['scored_tokens'], test_scored_tokens=test['scored_tokens']))
        (output / 'comparison.json').write_text(json.dumps(comparison, indent=2))
        with (output / 'comparison.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)
        results.commit()
    print(json.dumps(comparison, indent=2), flush=True)
    return comparison


@app.local_entrypoint()
def main(run_id: str = DEFAULT_RUN, replace_small: bool = False):
    validate_options(run_id, 7200.0)
    print('Inference only: small dropout 0.1; medium/large dropout 0.2; validation-selected best.pt.', flush=True)
    call = run_evaluation.spawn(run_id, replace_small)
    print(f'Call: {call.object_id}; updating maestro-nanogpt-results/{run_id}/', flush=True)
    call.get()
