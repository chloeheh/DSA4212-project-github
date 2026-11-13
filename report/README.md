# Building and checking the report

**Submit [main.pdf](main.pdf).** The limit is ten physical PDF pages total, including figures, references, and appendix. The restored NUS cover identifies Teo Chloe (A0286502E) and the repository URL. This version places figures and diagnostic tables in their relevant sections, with no separate appendix. Add other actual group members if applicable before submitting one group PDF.

## Source layout

| Path | Edit here for |
| --- | --- |
| `main.tex` | Original cover layout, author details, formatting, and section order. |
| `sections/01_data.tex` through `10_conclusions.tex` | Main argument, from setup to analysis and future experiments. |
| `references.bib` | Citations. |
| `figures/` | Three result figures copied from notebook exports, plus the supplied NUS cover logo. |
| `tables/event_family_losses.tex` | Event-family test results. |
| `tables/paired_test_summary.tex` | Generated equal-piece and paired-piece checks. |
| `build/` | Ignored compiler intermediates; do not edit. |

Install Tectonic or a TeX distribution with latexmk. From the repository root:

```bash
python3 scripts/analyze_saved_evaluation.py
bash report/build_report.sh
```

The first command regenerates both diagnostic tables from saved CSVs. The second compiles `report/main.tex` and updates `report/main.pdf`. A fallback compiler path supports the original machine; other machines need an installed compiler. Upload this folder's contents to Overleaf and select `main.tex` to compile independently. No root-level LaTeX or bibliography copies are needed.

## Evidence map

| Report sections | Source evidence |
| --- | --- |
| 1 | `notebooks/01_tokenization.ipynb`, dataset metadata and generated `data/maestro/summary.json`. |
| 2–3 | `modified-nanoGPT/config/`, model, loader, training loop, and Modal launcher. |
| 4 | `results/maestro-20261104-optimized-v1/tuning.json` and per-size summaries. |
| 5–6 | Baseline metrics and `notebooks/02_baseline_analysis.ipynb`. |
| 7–8 | Frozen regularization configs, follow-up metrics, and `notebooks/03_dropout_followup.ipynb`. |
| 9 | Final-evaluation protocol, saved per-piece/event files, `notebooks/04_test_evaluation.ipynb`, and `scripts/analyze_saved_evaluation.py`. |
| 10 | Findings above and cited papers in `references.bib`. Proposed augmentation/relative-attention experiments are untested. |

Paths in this map are relative to the repository root. Full setup, dependencies, launch commands, result retrieval, and checkpoint limitations are in [the reproduction guide](../docs/reproduction.md). Raw data and checkpoints are excluded from Git; saved CSVs support analysis without retraining.

## Reporting checks

Keep the initial hypothesis about model size separate from the dropout question developed after observing the baseline. Preserve the unsuccessful small-model follow-up and the disclosed revision after initial test access. A ranking across pieces is not a replication across random seeds, and token loss is not listening quality. The report distinguishes published empirical evidence, official documentation, practitioner advice, and untested interpretations.

After editing, rebuild and inspect the complete PDF for page count, readable figures, unresolved citations, and agreement with saved metrics. Confirm that the argument reflects the author's actual work and understanding. The repository does not contain extra seeds, stronger weight-decay runs, new architectures, or listening studies, so these must remain future work.
