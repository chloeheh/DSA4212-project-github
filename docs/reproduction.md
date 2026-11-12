# Reproducing the MAESTRO experiments

Start with the [root guide](../README.md) for environment setup and data preparation. Run shell commands below from the repository root. The adapted nanoGPT code remains in `modified-nanoGPT/`; notebooks are in `notebooks/`. Future augmentation and relative-attention experiments described in the report have **not** been implemented or run.

## 1. Environment and inputs

The training image pins Python 3.11, PyTorch 2.6.0, and NumPy 2.2.3. Install Modal 1.6.3 locally, then authenticate:

```bash
modal setup
```

The tokenizer produces these files in `data/maestro/`:

| File | Purpose |
| --- | --- |
| `train.bin`, `val.bin`, `test.bin` | `uint16` event IDs for the original dataset splits. |
| `meta.pkl` | Vocabulary size/mappings, tokenizer settings, and counts. |
| `piece_offsets.json` | Source filename, starting offset, length, and duration per piece. |
| `summary.json` | Piece/token counts for checking preparation. |
| `roundtrip_example.mid` | Decoded source example to inspect tokenization, not generated music. |

The training runner uploads only training/validation binaries and metadata. The evaluation runner additionally uploads the test binary. Preserve the original dataset's README/license. The event representation rounds timing to 10 ms, bins velocity, retains sustain CC64, and omits other original MIDI metadata.

## 2. Baseline model-size comparison

| Size | Layers | Heads | Width | Parameters |
| --- | ---: | ---: | ---: | ---: |
| Small | 6 | 4 | 256 | 5,080,320 |
| Medium | 8 | 8 | 512 | 25,891,328 |
| Large | 12 | 12 | 768 | 86,029,056 |

All models start from random weights: vocabulary 376, context 1,024, effective batch 32, dropout 0.1, AdamW weight decay 0.1, seed 1337. The learning rate rises during the first 2% of the time budget, then follows cosine decay from 0.0003 towards 0.00003. Configurations are in `modified-nanoGPT/config/`.

The A10 benchmark tests microbatches 4/8/16/32 with gradient accumulation keeping the effective batch fixed. It measures six full updates after three warm-up steps and chooses highest measured throughput below 85% allocated GPU memory. The saved benchmark selects batch 32 for each size. This is not a search over learning rates or all speed settings; compilation stays disabled.

Smoke-test real architectures (two optimizer steps per size), then launch a fresh run:

```bash
modal run --detach modified-nanoGPT/modal_train.py --run-id maestro-smoke-v1 --smoke --seconds 120
modal run --detach modified-nanoGPT/modal_train.py --run-id maestro-v1
```

Small → medium → large run sequentially, each with a 7,200-second loop budget. Budget includes initial/periodic validation and checkpoint saving; setup/final evaluation add overhead, and an ongoing update may finish past the boundary. A three-hour function timeout is a safety limit, not the training duration. Allow six GPU-hours plus benchmarking and overhead. Detached coordination survives local disconnection; no automatic retries repeat paid work.

Use a fresh run ID for a new experiment. Resume an interrupted run within its original budget:

```bash
modal run --detach modified-nanoGPT/modal_train.py --run-id maestro-v1 --resume
```

Completed sizes are skipped. Latest checkpoints preserve optimizer/scaler, random states, tokens, and elapsed time. Checkpoints are written atomically around ten-minute intervals and at completion; Volume publication occurs every minute. Work since the latest available checkpoint may be repeated after interruption.

## 3. Dropout follow-up

The completed baseline showed medium/large training losses falling while validation losses rose. This motivated increasing dropout from 0.1 to 0.2, rather than an initial assumption that dropout needed to be stronger. Small is included as a comparison with less obvious overfitting.

The three frozen `*_dropout02.py` configurations in `modified-nanoGPT/experiments/regularization/` change only dropout and the output directory relative to each saved baseline. Architecture, seed, optimizer, weight decay, schedule, context, effective batch, evaluation windows, and time budget stay fixed. The launcher validates the pairs before GPU training. It reuses baseline batching, performs no new sweep, and retrains all models from scratch.

```bash
modal run --detach modified-nanoGPT/experiments/regularization/modal_run.py
modal run --detach modified-nanoGPT/experiments/regularization/modal_run.py --resume
```

The default baseline reference is `maestro-20261104-optimized-v1`; default follow-up ID is `maestro-dropout02-20261106-v1`. To reproduce this exact follow-up procedure, the saved baseline configs/summaries must exist in the Modal results Volume under that baseline ID. A fresh unrelated baseline requires deliberately updating this reference and the frozen configurations together; do not silently compare mismatched runs. Resume uses the original budgets and skips completed sizes. Allow another six GPU-hours plus overhead.

The primary criterion is lower **best validation loss within each size**, not merely a smaller training/validation gap. Baseline best losses are 1.765233 / 1.801809 / 1.812228. `best_val_loss_change` is follow-up minus baseline, so negative favors stronger dropout. Each setting has only one seed; this does not establish statistical significance or an optimum. No weight-decay trial was run.

## 4. Checkpoint selection and final evaluation

Training-time checks use 20 batches of four 1,024-token windows per split. Evaluation seeds reset so checks reuse the same windows without advancing the training sampler. Dropout is disabled. `best.pt` has the lowest recorded validation loss; `ckpt.pt` is the latest/resumable state. Test loss does not select checkpoints.

The selected settings are small dropout 0.1 from the baseline and medium/large dropout 0.2 from the follow-up. Candidate identities and exact scoring are frozen in [protocol.json](../modified-nanoGPT/experiments/evaluation/protocol.json).

```bash
modal run --detach modified-nanoGPT/experiments/evaluation/modal_evaluate.py
```

Default ID: `maestro-dropout02-best-test-20261107-v1`. This is inference only, with BF16 on the A10, float32 token cross-entropy, and float64 accumulation. The runner checks the protocol and recorded checkpoint path/dropout before reusing completed results. It records checkpoint SHA-256 during inference, but does not rehash the current checkpoint when reusing an existing result. Keep source checkpoints immutable or use a new evaluation ID after changing them. Its one-hour timeout is a safety cap; there are no retries.

Every token after each piece's first is scored exactly once, including EOS, short pieces, and partial tails. Consecutive blocks have at most 1,024 targets; context resets per block and never crosses pieces. Padding targets are excluded. Overall loss is summed token loss divided by scored tokens; perplexity is its exponential. Validation and test use the same procedure: 3,790,050 and 4,334,092 targets. This differs from full sliding-context scoring and from the sampled validation estimator used to select checkpoints.

### Revision history to preserve

The original final evaluation used dropout 0.2 for all sizes. After viewing those test outputs, small was replaced with dropout 0.1, already preferred by pre-test validation. Only small inference was rerun; medium/large outputs were reused. The `--replace-small` option supported that explicit migration and is not needed to reuse the current completed evaluation. Original protocol/results remain under `history/original_all_dropout02/`. This is a disclosed post-test revision, not a new training experiment or a wholly untouched final protocol. Do not delete its archive or select future settings from these test outputs.

## 5. Saved outputs and download

All remote outputs use Modal Volume `maestro-nanogpt-results`:

```bash
mkdir -p results
modal volume get maestro-nanogpt-results /maestro-v1 ./results/maestro-v1
```

Substitute the actual run ID for both paths. If downloading over existing files, inspect conflicts before using `--force`.

| Output | Contents |
| --- | --- |
| Baseline `tuning.json` | Batch trials, timing/memory, and selected microbatches. |
| Training `comparison.csv` / `.json` | Summary across sizes. |
| Per-size `config.json`, `metrics.csv`, `summary.json` | Resolved settings, recorded losses/timing/tokens, and final/best summaries. |
| Per-size `best.pt`, `ckpt.pt` | Selected and latest states, excluded from Git. Load only trusted checkpoints/pickle metadata. |
| Follow-up comparison/experiment JSONs | Changes, per-size comparisons, and provenance. |
| Evaluation `protocol.json`, `comparison.csv` / `.json` | Selections, procedure, history, and aggregate scores. |
| Evaluation per-size `evaluation.json` | Checkpoint SHA-256, step, selection loss, configuration, exposure, and scores. |
| `validation_per_piece.csv`, `test_per_piece.csv` | Per-piece target counts, loss, and perplexity. |
| `validation_by_event.csv`, `test_by_event.csv` | Target counts and mean losses by event family. |

Results in Git retain the three completed runs' compact metrics/configuration/evaluation files. Checkpoints and datasets are not committed. To independently rerun inference, retrieve original checkpoints from the owner's Modal Volume, or retrain. Retraining can give different exact scores even with the same seed/hardware. A fresh clone can still check reported metrics and regenerate analysis from saved CSVs.

## 6. Analysis and local checks

Use [02_baseline_analysis.ipynb](../notebooks/02_baseline_analysis.ipynb), [03_dropout_followup.ipynb](../notebooks/03_dropout_followup.ipynb), and [04_test_evaluation.ipynb](../notebooks/04_test_evaluation.ipynb). Their default run IDs match the saved experiments. They require NumPy, pandas, and Matplotlib, but no GPU. Exports go to `plots/<run-id>/`; `results/` holds the source evidence. Report figures are a deliberately included subset copied to `report/figures/` for standalone compilation.

```bash
python3 scripts/analyze_saved_evaluation.py
python3 -m unittest discover -s modified-nanoGPT/tests -v
```

The first command needs only the standard library. It checks matched identities/counts, reproduces aggregate loss from per-piece rows, calculates equal-piece means and win counts, decomposes loss differences by event share, and regenerates both diagnostic report tables. These are exploratory checks of already inspected test outputs, not new training or independent replications. Tests use synthetic data to check boundaries, alignment, deterministic sampling, target coverage, padding, and invalid inputs; they require NumPy.

For local training, additionally install `torch==2.6.0`, then:

```bash
python3 modified-nanoGPT/train.py config/maestro_small.py
```

Configurations resolve relative to the current directory or the code folder. A reduced CPU integration check is:

```bash
python3 modified-nanoGPT/train.py config/maestro_small.py --device=cpu --dtype=float32 --n_layer=1 --n_head=2 --n_embd=32 --block_size=16 --batch_size=2 --gradient_accumulation_steps=1 --eval_iters=2 --max_iters=2 --out_dir=/tmp/maestro-cpu-check
```

Use a fresh output directory if repeating that check. CPU checks verify execution, not A10 speed or final experimental quality. The implementation uses one GPU per model; no distributed training or pretrained text weights are used.
