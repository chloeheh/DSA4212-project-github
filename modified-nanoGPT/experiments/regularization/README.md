# Dropout ablation across model sizes

This controlled follow-up trains **small, medium, and large** GPTs with dropout increased from **0.1 to 0.2**, keeping weight decay at **0.1**. Each model starts from scratch and receives the same **two-hour A10 training-loop budget** as its own baseline. Runs proceed sequentially: small → medium → large. Allow roughly six GPU hours plus startup and finalization. The original batching choices are reused; there is no additional benchmark or hyperparameter sweep.

## Rationale

The medium and large baselines show training loss continuing to fall while validation loss rises after its minimum. This is consistent with overfitting. Small largely plateaus late in training, so additional dropout could help less or even hurt it; including all three sizes tests that possibility directly.

Dropout masks activations during training to discourage reliance on specific feature combinations. The [original dropout study](https://jmlr.org/papers/v15/srivastava14a.html) supports this mechanism in its studied settings, not an improvement guarantee for this particular MAESTRO experiment. Here, the existing setting controls embedding dropout, attention-probability dropout, attention-output dropout, and MLP-output dropout. Evaluation disables these masks.

The baseline already uses [AdamW's decoupled weight decay](https://arxiv.org/abs/1711.05101) of 0.1 on matrix parameters. Increasing dropout tests a complementary regularization mechanism while retaining the original optimizer settings. The loss curves do not establish that weight shrinkage is inadequate, nor that dropout must outperform stronger weight decay. The moderate 0.2 setting is an experimental hypothesis, not a known optimum. Because we do not run a weight-decay intervention, this study cannot rank dropout against increased weight decay.

## Controlled comparison

| Setting | Original runs | Follow-up runs |
| --- | --- | --- |
| Architectures | Small: 6 layers / 4 heads / width 256; medium: 8 / 8 / 512; large: 12 / 12 / 768 | Same per size |
| Dropout | 0.1 | **0.2** |
| AdamW weight decay | 0.1 | Same |
| Peak/minimum learning rate | 0.0003 / 0.00003 | Same |
| Schedule | 2% warmup, then cosine decay over two hours | Same |
| Context length | 1,024 tokens | Same |
| Training microbatch / accumulation | 32 / 1 | Same; reuse baseline batching |
| Seed | 1337 | Same |
| Training-loop budget | 7,200 seconds per model | Same |
| Evaluation | 20 batches of 4 windows per split, approximately every 600 seconds | Same fixed windows/settings |
| Precision/device | BF16 / A10 | Same |

The baseline experiment is `maestro-20261104-optimized-v1`. Each configuration is frozen from the corresponding baseline's `config.json`, changing only dropout and the output destination. The launcher validates all three config pairs before GPU training begins. Each model uses its original architecture and initialization seed; none is fine-tuned from a baseline checkpoint. Equal time does not guarantee identical token exposure or optimization trajectories.

## Files

- `small_dropout02.py`, `medium_dropout02.py`, `large_dropout02.py`: frozen configurations for the three interventions.
- `modal_run.py`: dedicated launcher with a CPU coordinator, sequential GPU training, persistent checkpoints, and baseline comparisons.
- Existing `train.py`, `model.py`, and `data_loader.py` perform training; their defaults are unchanged.

## Launch and resume

From the repository root:

```bash
modal run --detach modified-nanoGPT/experiments/regularization/modal_run.py
```

The default run ID is `maestro-dropout02-20261106-v1`. A remote CPU coordinator preserves the sequence if the local client disconnects. There are no automatic retries. Initialization and final evaluation/save add overhead beyond the training-loop budget.

To resume interrupted work within each model's original budget:

```bash
modal run --detach modified-nanoGPT/experiments/regularization/modal_run.py --resume
```

Completed models are skipped on the CPU. A partially trained model resumes its checkpoint, and unstarted sizes start from scratch. Use the same run ID for resuming; a new run ID starts a separate experiment. The launcher rejects attempts to use the baseline run ID.

Outputs persist in Modal Volume `maestro-nanogpt-results`:

```text
maestro-dropout02-20261106-v1/
├── comparison.csv
├── comparison.json
├── small_experiment.json
├── small_baseline_comparison.json
├── medium_experiment.json
├── medium_baseline_comparison.json
├── large_experiment.json
├── large_baseline_comparison.json
├── small/
├── medium/
└── large/
    ├── config.json
    ├── metrics.csv
    ├── summary.json
    ├── best.pt
    └── ckpt.pt
```

All three size directories contain the listed training outputs. Comparisons are written after each successful model; a failure preserves earlier results. Download a completed run with:

```bash
modal volume get maestro-nanogpt-results /maestro-dropout02-20261106-v1 ./results/maestro-dropout02-20261106-v1
```

## Evaluation criteria

Compare each size against **its own baseline**, rather than comparing different architectures and attributing differences to dropout:

| Model | Baseline best validation cross-entropy |
| --- | ---: |
| Small | 1.765233 |
| Medium | 1.801809 |
| Large | 1.812228 |

Lower best validation cross-entropy is the primary improvement criterion. Also compare final validation loss, the learning curves, token counts, memory, and generated music. A smaller training/validation gap alone is insufficient if validation performance worsens. `best_val_loss_change` in the comparison files is follow-up minus baseline, so negative values favor dropout 0.2.

These are single-seed results on sampled validation windows with approximately ten-minute checkpoint resolution. They cannot establish statistical significance. The test split remains unused during adaptation selection. The existing root `training_analysis.ipynb` can visualize this experiment by changing `RUN_NAME` to `maestro-dropout02-20261106-v1`; its interpretation markdown refers to the baseline and must be revisited for a new run.

## Completed-run plots

Open [regularization_analysis.ipynb](../../../../regularization_analysis.ipynb) from the repository root and run its cells in order. The notebook reads the downloaded dropout metrics and original baseline metrics, displays both learning curves and baseline overlays inline, and writes PNG/PDF figures plus checkpoint/comparison CSV tables to `plots/maestro-dropout02-20261106-v1/`. No GPU is needed for analysis.
