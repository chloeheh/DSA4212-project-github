"""Execute the canonical tokenizer notebook without opening Jupyter.

Uses a fresh kernel; notebook source and saved outputs are left unchanged.
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite', action='store_true', help='replace existing prepared data')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not (root / 'maestro-data/maestro-v3.0.0.csv').is_file():
        parser.error('Extract MAESTRO v3.0.0 MIDI-only into maestro-data/ first; see docs/reproduction.md')
    if any((root / 'data/maestro' / name).exists() for name in
           ('train.bin', 'val.bin', 'test.bin', 'meta.pkl', 'piece_offsets.json', 'summary.json')) and not args.overwrite:
        parser.error('Prepared data already exists. Use --overwrite to regenerate it.')
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    notebook = nbformat.read(root / 'notebooks/01_tokenization.ipynb', as_version=4)
    manager = KernelManager(kernel_name='python3')
    # Use this interpreter even if an unrelated global python3 kernel is registered.
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    print('Tokenizing 1,276 MIDI files; this can take several minutes.', flush=True)
    NotebookClient(notebook, timeout=3600, km=manager,
                   resources={'metadata': {'path': str(root)}}).execute(cleanup_kc=True)
    print('Prepared data/maestro/. Now run: python scripts/verify_reproducibility.py --data')


if __name__ == '__main__':
    main()
