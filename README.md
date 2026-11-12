# MAESTRO music-event modeling

Which model size gives the best next-event predictions within two hours on one NVIDIA A10? We train small, medium, and large GPTs from random weights. The baseline's rising validation losses motivate a dropout follow-up. Small remains best in the recorded runs; stronger dropout helps the larger models but does not reverse the ranking.

**[Read the report](report/main.pdf)** · [Reproduce the experiments](docs/reproduction.md) · [Sources and interpretation limits](docs/source_notes.md)

## Project flow

```text
maestro-data/ → notebooks/01_tokenization.ipynb → data/maestro/
                                                      ↓
                                             modified-nanoGPT/
                                                      ↓
                                          results/ → notebooks/
                                                      ↓
                                           plots/ → report/
```

| Folder | What belongs here |
| --- | --- |
| `notebooks/` | Four notebooks in workflow order: tokenize, inspect baseline, compare dropout, inspect test results. |
| `modified-nanoGPT/` | Adapted model, loader, benchmark, training/evaluation runners, configurations, and tests. The upstream name is retained so existing launch paths remain valid. |
| `results/` | Saved configurations, metrics, evaluation outputs, and the original test-evaluation history. |
| `plots/` | Notebook exports and derived comparison CSVs, grouped by run ID. |
| `scripts/` | Reproducible analysis of saved evaluation files. |
| `report/` | Submitted PDF and its complete LaTeX, bibliography, tables, and included figures. |
| `docs/` | One reproduction guide and a source-to-claim reading map. |

Raw MIDI in `maestro-data/`, generated data in `data/maestro/`, and model checkpoints are excluded from Git. Download/prepare the data or retrieve original checkpoints as described in the reproduction guide. Saved metrics can be analyzed without GPU training.

## Environment and data preparation

Use Python 3.11. The remote training image pins PyTorch 2.6.0 and NumPy 2.2.3. For notebooks and the Modal client:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.2.3 pandas matplotlib pretty_midi jupyterlab ipykernel modal==1.6.3
jupyter lab
```

Download the **MAESTRO v3.0.0 MIDI-only archive** from the [official dataset page](https://magenta.tensorflow.org/datasets/maestro). Extract its contents under `maestro-data/`, preserving year folders and `maestro-v3.0.0.csv`/`.json`. Retain the dataset's README and license. Run [01_tokenization.ipynb](notebooks/01_tokenization.ipynb) in order. It preserves the supplied splits and writes token files plus metadata under `data/maestro/`; rerunning export cells overwrites those generated files. All notebooks find the project root from either the root or `notebooks/`.

Expected piece counts are 962 training, 137 validation, and 177 test; token counts are 33,293,600 / 3,790,187 / 4,334,269. The 376-token vocabulary uses note starts/endings, time shifts, velocity bins, pedal changes, and piece boundaries. It loses some original MIDI detail.

## Reproduce results

The [reproduction guide](docs/reproduction.md) gives launch/resume/download commands, frozen run IDs, output descriptions, evaluation history, local checks, and checkpoint requirements. GPU runs require Modal authentication and billing. The six completed training runs represent approximately 12 training-loop GPU-hours, plus benchmarking and overhead.

For saved-result analysis, run the notebooks in `notebooks/` in numerical order, skipping tokenization if data preparation is not needed. Notebook analysis exports to `plots/<run-id>/`. For the report's paired-piece and event-contribution checks:

```bash
python3 scripts/analyze_saved_evaluation.py
```

This needs only Python's standard library and saved CSVs. It performs no model training or inference. Small's lower loss on all 177 test pieces is conditional on these saved checkpoints; it does not measure variation across training seeds.

## Check and build

```bash
python3 -m unittest discover -s modified-nanoGPT/tests -v
bash report/build_report.sh
```

Tests require NumPy. Building requires Tectonic or latexmk. Submit **report/main.pdf**; all LaTeX sources and included figures are in `report/`. The limit is **ten pages total, including references and appendix**. See the [report guide](report/README.md) for editing and the map from report claims to experimental evidence.
