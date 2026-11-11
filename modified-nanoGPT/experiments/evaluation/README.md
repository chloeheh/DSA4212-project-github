# Final evaluation: validation-selected configurations

The project objective is **best performance obtained within a fixed compute budget per training run**, rather than performance restricted to the final two-hour weights. This decision should guide the final LaTeX report. The primary checkpoint rule is the minimum recorded validation cross-entropy (`best.pt`) within each run's two-hour budget.

The current primary comparison replaces the previous small entry in the **same** evaluation dataset:

- Small, dropout **0.1**, `maestro-20261104-optimized-v1/small/best.pt`.
- Medium, dropout **0.2**, `maestro-dropout02-20261106-v1/medium/best.pt`.
- Large, dropout **0.2**, `maestro-dropout02-20261106-v1/large/best.pt`.

These settings had the lowest sampled validation losses within each size among the two tested dropout values. They are not globally optimal models. Since dropout varies across sizes, this is a comparison of validation-selected configurations, not an architecture-only comparison.

### Evaluation history

The original evaluation used dropout 0.2 for all sizes. After those test results were viewed, the user requested replacing small with dropout 0.1, which had already been preferred by validation before test access. Only small inference was repeated; medium/large results were reused unchanged. The old small result, protocol, and comparison are preserved under `history/original_all_dropout02/`. This is a revision of the existing evaluation, not a new training experiment; it is disclosed rather than claimed to be wholly preregistered before all test access.

## Protocol

`protocol.json` records the revised candidate identities, objective, scoring rule, and evaluation history. No training or optimizer updates occur. Inference uses `model.eval()`, `torch.inference_mode()`, and BF16 autocasting on the A10. Cross-entropy is accumulated from float32 token losses in float64 sums.

`evaluate.py` scores both the full validation split and the full test split using identical deterministic piece-aware windowing:

1. For each piece, predict every token after its first token exactly once, including EOS.
2. Use consecutive blocks of up to 1,024 targets. Context resets at each block boundary; this is not sliding-window evaluation with a full 1,024-token history for every target.
3. Retain short pieces and final partial blocks. Right-padding has target -1 and is excluded from all metrics.
4. Never carry context across a piece boundary.
5. Divide summed negative log likelihood by the total number of scored tokens. Perplexity is `exp(cross_entropy)`.

With the current dataset this scores **3,790,050 validation targets** across 137 pieces and **4,334,092 test targets** across 177 pieces. These are raw split token counts minus one first token per piece.

The original training curves used 80 randomly sampled validation windows per evaluation. Full-coverage results can therefore differ from that checkpoint-selection loss. Compare full validation and test using the new matched protocol; retain the original selection loss as provenance rather than presenting the metrics as interchangeable. Per-piece averages are diagnostics, not the aggregation used for the overall token-weighted result.

## Run

From the repository root:

```bash
modal run --detach modified-nanoGPT/experiments/evaluation/modal_evaluate.py
```

Default evaluation run ID: `maestro-dropout02-best-test-20261107-v1`. The function evaluates missing selected checkpoints and reuses completed matching results. The small-entry replacement used `--replace-small`; repeated ordinary invocations now reuse all completed results. It has a one-hour safety timeout and no retries; actual inference should be much shorter than training. It reads the existing checkpoints from the persistent Volume, uploads the test binary, and writes to a separate evaluation directory. Completed per-model evaluations are reused when the same run ID is invoked again.

Outputs in Modal Volume `maestro-nanogpt-results`:

```text
maestro-dropout02-best-test-20261107-v1/
├── protocol.json
├── comparison.csv
├── comparison.json
├── small/
├── medium/
└── large/
    ├── evaluation.json
    ├── validation_per_piece.csv
    ├── validation_by_event.csv
    ├── test_per_piece.csv
    └── test_by_event.csv
```

Each model directory has the same five files. `evaluation.json` records the checkpoint SHA-256, selected step, selection loss, training exposure, model configuration, scored-token counts, and aggregate metrics. The other files support piece-level and event-family analysis. No checkpoint files are copied into the evaluation directory.

Download the small result bundle:

```bash
mkdir -p results
modal volume get --force maestro-nanogpt-results /maestro-dropout02-best-test-20261107-v1/ ./results/
```

## Validation checks

Window tests verify exact target coverage, piece boundaries, and final partial blocks. A local reduced-model integration check with uniform logits verifies cross-entropy `ln(vocab_size)` and independence from evaluation batch grouping. The revised selection is based on the earlier validation ranking, not an attempt to optimize test loss. The timing of the revision after initial test access is explicitly retained in the report. Single-seed comparisons should not be presented as statistical significance or direct measurements of musical quality.
