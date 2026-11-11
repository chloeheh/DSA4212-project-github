# MAESTRO music-event modeling

This project tokenizes MAESTRO piano MIDI performances and trains three decoder-only GPT models from random initialization to predict the next music-event token. The research objective is the best performance obtainable **within** a fixed compute budget per training run: primary evaluation uses `best.pt`, selected by the lowest recorded validation cross-entropy, rather than requiring the final two-hour weights. The experiment compares small, medium, and large models under the same two-hour training budget on a Modal NVIDIA A10 GPU.

The workflow is:

```text
maestro-data/           tokenization.ipynb          data/maestro/
Raw MIDI + metadata  → Music-event tokenization  → Token binaries + metadata
                                                        ↓
                                              modified-nanoGPT/
                                              Benchmark and train
                                                        ↓
                                              Modal results Volume
                                              Checkpoints + metrics
```

## Repository layout

| Path | Purpose |
| --- | --- |
| [main.tex](main.tex) / [main.pdf](main.pdf) | Project report capped at ten pages including cover, excluding references and appendix, with modular sources and evidence notes under [report/](report/README.md). |
| [tokenization.ipynb](tokenization.ipynb) | Validates MIDI paths and dataset splits, defines the event vocabulary, encodes performances, creates an example reconstructed MIDI, and writes training inputs. |
| [maestro-data/](maestro-data/) | Source dataset: MIDI files grouped by year, original CSV/JSON metadata, and dataset README/license. |
| [data/maestro/](data/maestro/) | Generated tokenized dataset consumed by the training code. |
| [modified-nanoGPT/](modified-nanoGPT/) | GPT architecture, piece-aware data loader, training loop, A10 batching benchmark, Modal runner, size configs, and tests. |
| [modified-nanoGPT/experiments/regularization/](modified-nanoGPT/experiments/regularization/) | Controlled follow-up: small, medium, and large dropout ablations (0.1 to 0.2), each trained for two hours; frozen configs and a dedicated Modal launcher. |
| [modified-nanoGPT/README.md](modified-nanoGPT/README.md) | Detailed training configuration, checkpoint/resume behavior, Modal commands, and output descriptions. |
| [training_analysis.ipynb](training_analysis.ipynb) | Displays training/validation learning curves and checkpoint summaries inline using NumPy, pandas, and Matplotlib. |
| [regularization_analysis.ipynb](regularization_analysis.ipynb) | Dropout 0.2 learning curves and per-size comparisons against the original dropout 0.1 baselines; exports figures under `plots/maestro-dropout02-20261106-v1/`. |
| [test_evaluation.ipynb](test_evaluation.ipynb) | Final validation/test metrics and per-piece/event-family diagnostics for the validation-selected small dropout 0.1 and medium/large dropout 0.2 best checkpoints. |
| [modified-nanoGPT/experiments/evaluation/](modified-nanoGPT/experiments/evaluation/) | Frozen final-evaluation protocol and inference-only Modal runner for the validation-selected small dropout 0.1 and medium/large dropout 0.2 best checkpoints. |
| [plots/](plots/) | Exported PNG/PDF figures and checkpoint summary tables, grouped by experiment run ID. |

## 1. Prepare the source dataset

Place the MAESTRO v3 MIDI dataset in `maestro-data/`, preserving the relative paths used by its metadata:

```text
maestro-data/
├── 2004/
│   └── ... .midi
├── 2006/
├── ...
├── 2018/
├── maestro-v3.0.0.csv
├── maestro-v3.0.0.json
├── README
└── LICENSE
```

The notebook reads `maestro-v3.0.0.csv` to locate MIDI files and preserve the supplied train/validation/test assignments. Its input paths are relative to the repository root. Audio recordings are not inputs to this pipeline.

The source dataset and its license are separate from the project code. See the supplied [dataset README](maestro-data/README) and [LICENSE](maestro-data/LICENSE) for its source and terms.

## 2. Tokenize the MIDI performances

From the repository root, create a local environment and install the notebook dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas pretty_midi jupyterlab ipykernel
python -m ipykernel install --user --name maestro-project --display-name "MAESTRO project"
jupyter lab
```

Open [tokenization.ipynb](tokenization.ipynb), select the environment's kernel, and run its cells in order. Keep the notebook working directory at the repository root. The notebook checks that all referenced MIDI files exist before processing them.

The tokenizer represents performances using 376 event tokens:

| Token family | Count | Meaning |
| --- | ---: | --- |
| `NOTE_ON_0`–`NOTE_ON_127` | 128 | Start a note of the specified pitch. |
| `NOTE_OFF_0`–`NOTE_OFF_127` | 128 | End a note of the specified pitch. |
| `TIME_SHIFT_1`–`TIME_SHIFT_100` | 100 | Advance time by 0.01–1.00 seconds; longer gaps use multiple tokens. |
| `VELOCITY_0`–`VELOCITY_15` | 16 | Set the velocity category for subsequent note-on events. |
| `PEDAL_ON`, `PEDAL_OFF` | 2 | Change the sustain pedal state from MIDI controller CC64. |
| `BOS`, `EOS` | 2 | Mark the beginning and end of each piece. |

`midi_to_tokens()` converts a MIDI file into token IDs. Events are sorted by quantized time; note-offs precede pedal events and note-ons at the same tick. Timing uses 10 ms steps and velocity uses 16 bins, so the representation is lossy. Other MIDI information, including original tempo/program metadata and controllers other than CC64, is not retained in the event stream.

`tokens_to_midi()` reconstructs an approximate piano performance for inspection. The notebook writes `roundtrip_example.mid` from one encoded source piece; it is a tokenizer check, not model-generated music.

### Generated files

The notebook creates these files in `data/maestro/`. Re-running the export cells overwrites them.

| File | Contents |
| --- | --- |
| `train.bin` | Training token IDs, stored as `uint16`. |
| `val.bin` | Validation token IDs; the metadata split `validation` maps to this filename. |
| `test.bin` | Held-out test token IDs. |
| `meta.pkl` | Vocabulary size, token-to-ID and ID-to-token mappings, tokenizer settings, and split counts. |
| `piece_offsets.json` | Each piece's source filename, token start offset, token length, and duration, grouped by split. |
| `summary.json` | Piece counts, token counts, and mean tokens per piece for each split. |
| `roundtrip_example.mid` | Approximate reconstruction used to inspect tokenization. |

The currently generated dataset has the following counts, recorded in [summary.json](data/maestro/summary.json):

| Split | Pieces | Tokens |
| --- | ---: | ---: |
| Training | 962 | 33,293,600 |
| Validation | 137 | 3,790,187 |
| Test | 177 | 4,334,269 |

## 3. Train the model-size experiment

The model learns to predict the next event ID from the available context. For example, each sampled window supplies aligned sequences:

```text
Input:   token₀, token₁, token₂, …, token₁₀₂₃
Target:  token₁, token₂, token₃, …, token₁₀₂₄
```

The loader uses `piece_offsets.json` to keep inputs and targets within one piece. `train.py` reads the vocabulary size from `meta.pkl`, overriding the generic `GPTConfig` defaults in `model.py`.

| Model | Layers | Heads | Embedding width | Total parameters |
| --- | ---: | ---: | ---: | ---: |
| Small | 6 | 4 | 256 | 5,080,320 |
| Medium | 8 | 8 | 512 | 25,891,328 |
| Large | 12 | 12 | 768 | 86,029,056 |

All sizes use a 1,024-token context and an effective batch of 32 sequences. The Modal runner benchmarks microbatch sizes for each architecture while preserving that effective batch. Fixed validation samples remain the same across sizes. Models train from scratch using AdamW and BF16 mixed precision on the A10.

Install and authenticate Modal locally; PyTorch is installed in the remote training image:

```bash
python -m pip install 'modal==1.6.3'
modal setup
```

Smoke-test the architectures with two optimizer steps per size:

```bash
modal run --detach modified-nanoGPT/modal_train.py --run-id maestro-smoke-v1 --smoke --seconds 120
```

Launch a full experiment with a fresh run ID:

```bash
modal run --detach modified-nanoGPT/modal_train.py --run-id maestro-v1
```

This starts the batching benchmark, followed by **small → medium → large**, each with a 7,200-second training-loop budget. Allow roughly six GPU hours plus benchmarking, startup, and finalization. The budget includes initial/periodic evaluation and checkpoint writes; initialization and final evaluation/save add overhead. Modal GPU access requires enabled billing.

For local GPU training, resume instructions, CPU checks, and detailed settings, see the [training README](modified-nanoGPT/README.md).

## 4. Inspect results

Remote outputs persist in the Modal Volume `maestro-nanogpt-results`, under the chosen run ID. Download a run from the repository root:

```bash
mkdir -p results
modal volume get maestro-nanogpt-results /maestro-v1 ./results/maestro-v1
```

The run directory contains `tuning.json` with measured batching choices and `comparison.csv`/`comparison.json` with model results. Each size has its own configuration, metrics, best/latest checkpoints, and summary. See the [output descriptions](modified-nanoGPT/README.md) for details.

Lower validation loss indicates better next-event prediction on the sampled validation windows. An equal-time comparison measures which size performs best within the budget; it does not hold the number of processed tokens constant. Token perplexity also does not directly measure musical quality.

After downloading the metrics, open [training_analysis.ipynb](training_analysis.ipynb) in Jupyter or VS Code and run its cells in order. It uses NumPy, pandas, and Matplotlib:

```bash
python -m pip install numpy pandas matplotlib ipykernel
```

Set `RUN_NAME` in the notebook to the experiment to analyze (the default is `maestro-20261104-optimized-v1`). Figures and checkpoint tables display inline. Exports are saved separately from the input results, in `plots/<RUN_NAME>/`:

- `learning_curves.png` and `.pdf`: full trajectories and zoomed views after initialization.
- `validation_comparison.png` and `.pdf`: validation curves overlaid across sizes.
- `checkpoint_summary.csv`: best checkpoints and final losses.

Points represent actual recorded evaluations, about ten minutes apart, and stars mark the lowest observed validation loss. Connecting lines do not represent additional measurements. The earlier exports under `results/maestro-20261104-optimized-v1/plots/` are retained; new notebook exports go into the root `plots/` folder.

Final evaluation is specified in the [evaluation protocol and revision history](modified-nanoGPT/experiments/evaluation/README.md): small dropout 0.1 and medium/large dropout 0.2 use their validation-selected `best.pt` checkpoints. The inference-only runner evaluates full validation and test splits under the same piece-aware coverage rule, scores every eligible target once, and reports token-weighted cross-entropy and perplexity. The original sampled validation loss is retained as the checkpoint-selection metric and is distinct from full-coverage evaluation. The original all-0.2 evaluation was revised in place to replace small with the setting already preferred by validation; this occurred after initial test access and is disclosed in the report. Original results are archived for audit. `model.py` provides token generation and the tokenization notebook provides MIDI decoding; a standalone generation workflow is not yet implemented.

## Checks

Run the data-loader tests from the repository root:

```bash
python -m unittest discover -s modified-nanoGPT/tests -v
```

These tests use temporary synthetic data to check valid piece boundaries, input/target alignment, independent sampling generators, and rejection of invalid inputs. They require NumPy and do not train a model or require a GPU. The tokenization notebook also checks split lengths, token ranges, and beginning/end markers after export.
