# MAESTRO model-size experiment

This document covers the training implementation and execution details. For the complete repository workflow, source MAESTRO dataset, tokenization notebook, and generated data files, start with the [project README](../README.md).

These are decoder-only GPTs trained from scratch on your existing 376-token music-event vocabulary, not pretrained text GPT-2 weights.

| Size | Layers | Heads | Width | Total parameters |
| --- | ---: | ---: | ---: | ---: |
| Small | 6 | 4 | 256 | 5,080,320 |
| Medium | 8 | 8 | 512 | 25,891,328 |
| Large | 12 | 12 | 768 | 86,029,056 |

All models use context length 1,024, an effective batch of 32 sequences (32,768 target tokens per optimizer step), dropout 0.1, AdamW, seed 1337, and the same learning-rate schedule. Before a full Modal run, `benchmark.py` measures microbatch sizes 4, 8, 16, and 32 on the A10 for each architecture. It selects the highest measured throughput using less than 85% of GPU memory, adjusting gradient accumulation to keep the effective batch unchanged. The measured trials and selections are saved in `tuning.json`. Validation always uses microbatch size 4 and 20 batches per split, regardless of the training microbatch. Direct local runs retain the default microbatch 4 with eight accumulation steps. A piece-aware loader samples uniformly over valid windows; neither inputs nor next-token targets cross a piece boundary. Vocabulary and boundaries come from `data/maestro/meta.pkl` and `piece_offsets.json`. Only training and validation splits are used. Keep `test.bin` for final evaluation after choosing the model using validation results.

Each run has a **7,200-second training-loop budget**, including initial/periodic validation and checkpoint writes. Initialization and final validation/checkpointing add overhead. The loop stops after the current optimizer step, so it can exceed its budget by one step. The Modal hard timeout is three hours to allow startup/finalization; it does not determine training duration. Learning-rate warmup uses the first 2% of the time budget, followed by cosine decay. Resume continues the original budget rather than granting another two hours.

This answers “which size performs best under the same time budget?” Larger models can process fewer tokens. It does not isolate size at equal tokens or prove convergence. Compare validation loss (lower is better), token counts, throughput, and memory; use more seeds for stronger conclusions. Fixed evaluation windows are identical across sizes and evaluations, and evaluation does not consume the training data RNG. Perplexity is over music-event tokens, not musical quality.

## Run on Modal

Run commands from the project root. Install/authenticate the Modal CLI if needed:

```bash
python3 -m pip install 'modal==1.6.3'
modal setup
```

First smoke-test all three real architectures on an A10 (two optimizer steps each, short validation):

```bash
modal run --detach modified-nanoGPT/modal_train.py --run-id maestro-smoke-v1 --smoke --seconds 120
```

Then launch the experiment:

```bash
modal run --detach modified-nanoGPT/modal_train.py --run-id maestro-v1
```

The runner uploads token files and code, builds a pinned Python 3.11/PyTorch 2.6.0/NumPy 2.2.3 image, and runs **small → medium → large sequentially**, each on one A10. Allow roughly six GPU hours plus a short A10 batching benchmark and startup/finalization overhead. The remote CPU coordinator keeps the sequence running if the local client disconnects. Monitor the dashboard URL printed by Modal. Each size gets its own persistent output directory in the `maestro-nanogpt-results` Volume. No automatic retries are enabled, to avoid silently repeating paid runs.

Use a fresh run ID for a new experiment. To resume an interrupted experiment with saved checkpoints:

```bash
modal run --detach modified-nanoGPT/modal_train.py --run-id maestro-v1 --resume
```

Completed models are checked before resuming, so they do not receive an extra training budget. An interrupted step or work since the last published checkpoint may be repeated. Checkpoints save model, optimizer, scaler, random states, token count, and elapsed budget. Publishing happens every minute; new checkpoints are written about every ten minutes and at completion.

Download results:

```bash
modal volume get maestro-nanogpt-results /maestro-v1 ./results/maestro-v1
```

The run directory contains `comparison.csv`/`comparison.json`. Each size directory contains:

- `config.json`: resolved configuration.
- `metrics.csv`: step, elapsed seconds, tokens seen, train/validation loss, validation perplexity, and learning rate.
- `ckpt.pt`: latest state, for resuming.
- `best.pt`: state with lowest validation loss.
- `summary.json`: parameter count, final/best loss, steps, tokens, elapsed time, and peak GPU memory.

If a job fails, earlier sizes and their comparison rows remain saved. Checkpoints use atomic renames so a partially written checkpoint does not replace the latest complete one. Only load pickle metadata/checkpoints you trust.

## Local or other single-GPU host

```bash
python3 -m pip install torch==2.6.0 numpy==2.2.3
python3 modified-nanoGPT/train.py config/maestro_small.py
```

Configuration files resolve from the current directory or `modified-nanoGPT/`. `--key=value` arguments override config files. Small CPU integration check:

```bash
python3 modified-nanoGPT/train.py config/maestro_small.py --device=cpu --dtype=float32 --n_layer=1 --n_head=2 --n_embd=32 --block_size=16 --batch_size=2 --gradient_accumulation_steps=1 --eval_iters=2 --max_iters=2 --out_dir=/tmp/maestro-cpu-check
```

The training script is intentionally for one GPU per model. Distributed training, text pretrained weights, and the original OpenWebText defaults have been replaced by the MAESTRO experiment setup. The underlying GPT architecture in `model.py` is unchanged.

Modal requires a payment method for GPU functions. If a launch reports “Please add a payment method to use A10G GPU functions,” enable billing in your Modal workspace and rerun the smoke test before the full experiment. This error occurs before GPU training starts.

Implementation references: [Modal GPU selection](https://modal.com/docs/guide/gpu), [persistent Volumes](https://modal.com/docs/guide/volumes), [function timeouts](https://modal.com/docs/guide/timeouts), and [detached long training](https://modal.com/docs/examples/long-training).

Data-loader regression checks (NumPy only):

```bash
python3 -m unittest discover -s modified-nanoGPT/tests -v
```

The batching benchmark uses the real MAESTRO loader and full forward/backward/optimizer steps with BF16. It measures six steps after three warm-up steps per candidate. Tuning weights are discarded; each training run starts from scratch. This is a measured batching optimization, not a guarantee of globally optimal speed or model quality. `torch.compile` remains disabled; compiler modes and learning-rate hyperparameters are not searched. Relevant practices are described in the [PyTorch performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide).

## Follow-up regularization experiment

The [dropout ablation across model sizes](experiments/regularization/README.md) trains small, medium, and large sequentially for two hours each on an A10, with dropout 0.2 instead of 0.1. It reuses each baseline’s batching and other training settings, and saves outputs in a separate experiment directory. See that experiment directory for the evidence, limitations, and launch/resume commands.

## Final checkpoint policy and test evaluation

The project targets best performance **within** the fixed training budget. Primary evaluation uses validation-selected `best.pt` checkpoints, not necessarily final `ckpt.pt` states. The [final evaluation protocol](experiments/evaluation/README.md) documents the validation-selected small dropout 0.1 and medium/large dropout 0.2 checkpoints, including the disclosed revision after the first all-0.2 evaluation and scores full validation and test splits under the same piece-aware coverage rule. `evaluate.py` reports token-weighted cross-entropy, perplexity, and piece/event diagnostics without training. See the root `test_evaluation.ipynb` for analysis once outputs are downloaded.
