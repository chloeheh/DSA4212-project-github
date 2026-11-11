# Report source and evidence map

`../main.tex` is the report entry point. The document is split into ten files in `sections/` to keep edits manageable. `../main.pdf` is the compiled deliverable. The cover and main report together are limited to at most ten pages. References and the figure appendix are separate and excluded from that limit.

| Section | Section file | Main evidence |
| --- | --- | --- |
| 1 | `01_data.tex` | Tokenization notebook, dataset metadata, `data/maestro/summary.json` |
| 2 | `02_question_models_metrics.tex` | Architecture configs, objective, checkpoint rule, metric definition |
| 3 | `03_implementation.tex` | `model.py`, `data_loader.py`, `train.py`, Modal launcher |
| 4 | `04_compute_efficiency.tex` | Baseline `tuning.json` and per-model summaries |
| 5 | `05_baseline_results.tex` | Baseline curves and checkpoint summary |
| 6 | `06_baseline_analysis.tex` | Baseline validation progress and train/validation gaps |
| 7 | `07_regularization_design.tex` | Frozen regularization configs and dropout/AdamW references |
| 8 | `08_regularization_results.tex` | Dropout comparison notebook, CSV tables, and figure |
| 9 | `09_final_evaluation.tex` | Full-coverage evaluation summaries, per-piece and event-family diagnostics |
| 10 | `10_conclusions.tex` | Synthesis, limitations, future experiments, qualified MoE proposal |

## Reporting decisions preserved

- Objective: best performance **within** a two-hour A10 training-loop budget per run.
- Primary checkpoint: minimum recorded sampled validation loss (`best.pt`), not automatically the final weights.
- Baseline and regularization analyses distinguish best from final validation losses.
- Final testing uses **small dropout 0.1, medium dropout 0.2, large dropout 0.2**, all at their validation-selected best checkpoints. The small entry replaces the original dropout 0.2 entry in the same evaluation; history is archived and disclosed in the text.
- Sampled selection loss and deterministic full-coverage evaluation are not interchangeable.
- All actual results are single-seed; the report does not claim significance, globally optimal hyperparameters, or evaluated musical quality.
- MoE is future work, not an implemented experiment or an almost-certain improvement.
- High EOS loss and the possible effect of within-piece fixed-window sampling are presented as a diagnostic and an untested mechanism, respectively.

## Build

After saving edits to `main.tex`, any file in `report/sections/`, or `references.bib`, run from the repository root:

```bash
bash report/build_report.sh
```

The script compiles the complete report and synchronizes **both** `main.pdf` and `report/build/main.pdf`. It uses installed Tectonic or latexmk, with a fallback to the temporary compiler used on this machine. If that temporary compiler is removed, install Tectonic/a TeX distribution or use Overleaf.

Edit the cover in `main.tex`, the report body in `report/sections/*.tex`, and citations in `references.bib`. Always compile `main.tex`, not an individual subsection file. Files such as `.aux`, `.out`, `.log`, `.blg`, and `.bbl` in `report/build/` are generated build artifacts, not report source, and should not be edited by hand. Reopen/refresh the PDF preview after a successful build.

Tectonic was used to check the current PDF. The compiler binary and PDF inspection tools used during drafting are temporary tools, not project dependencies. For Overleaf, upload `main.tex`, `references.bib`, `nus_logo.jpg`, `report/sections/`, `report/appendix.tex`, `report/generated_event_table.tex`, and the referenced PDF figures with their existing relative paths. Recompile from `main.tex`.

The ten-page cap conservatively includes the cover and main text, but excludes references and appendix. Body sections flow continuously to avoid wasting pages. Large figures are in `report/appendix.tex` and referenced from the main sections. Changes can affect pagination; recompile and recheck the physical PDF pages before References.

## Cover details

The cover in `main.tex` preserves the name, student ID, and email supplied in the user-edited template. Its repository URL is the configured Git origin. Edit those fields directly in `main.tex` as needed, then rebuild.
