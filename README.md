# pcl-detection-nlp

Coursework repository for the SemEval 2022 PCL binary classification task.

The current structure is aligned with the coursework spec in `70016_1_spec.pdf` and the baseline direction in `paper/2020.coling-main.518.pdf`:

- task focus: SemEval 2022 Task 4 Subtask 1 (`PCL` vs `No PCL`)
- required outputs: `dev.txt` and `test.txt`
- baseline-ready modules: data loading, binary label conversion, EDA, evaluation, submission export
- model families: tuned sparse baselines, sparse ensembles, and an optional `roberta` training path

## Structure (Spec-Focused)

```text
report.tex
report_assets/
dev.txt
test.txt
coursework_runner.ipynb
BestModel/
  checkpoints/
  code/
  bundle_manifest.json
  coursework_runner.ipynb
  dev.txt
  test.txt
archive/
  basic_tests.ps1
  configs/
  artifacts/
configs/
  baseline.json
  best_artifact_ensemble.json
  best_roberta_finetune.json
  best_roberta_probe.json
  best_hybrid_ensemble.json
  best_tfidf_ensemble.json
  best_tfidf_svm.json
  roberta_download_template.json
data/
report_assets/
src/
  pcl_detection/
    cli.py
    config.py
    data_pipeline.py
    experiment_utils.py
    models.py
    notebook_support.py
    training_pipeline.py
tests/
  test_data.py
  test_experiments.py
  test_export.py
  test_metrics.py
```

## Project organization

- `report.tex`: the final report source, compiled to the PDF required by the coursework submission.
- `report_assets/`: the notebook-generated figures used directly by `report.tex`.
- `dev.txt` and `test.txt`: the submission-ready prediction files required by the leaderboard.
- `coursework_runner.ipynb`: the main interactive notebook for step-by-step coursework execution, including tests, EDA, training, export, and figures.
- `BestModel/`: the final selected local model bundle, including the chosen config, direct notebook/code snapshots, bundled checkpoints, and mirrored submission files.
- `archive/`: historical exploratory configs, older helper scripts, and selected artifacts retained for traceability but removed from the main working surface.
- `configs/`: active experiment configurations. `best_artifact_ensemble.json` stores the current recommended local model, `best_roberta_finetune.json` keeps the strongest single fine-tuned RoBERTa checkpoint, `best_roberta_probe.json` keeps the strongest linear-probe variant, `best_hybrid_ensemble.json` keeps the strongest non-transformer local model, `best_tfidf_ensemble.json` keeps the strongest pure sparse-score ensemble, `best_tfidf_svm.json` keeps the best single-model sparse baseline, while `baseline.json` and `roberta_download_template.json` remain as lightweight starting points.
- `data/`: the local dataset files used by the loader and notebook.
- `src/pcl_detection/data_pipeline.py`: data loading, label conversion, preprocessing, and EDA logic.
- `src/pcl_detection/models.py`: model definitions, including sparse baselines, sparse ensembles, and the optional RoBERTa path.
- `src/pcl_detection/training_pipeline.py`: training, evaluation, error analysis, and submission export logic.
- `src/pcl_detection/experiment_utils.py`: experiment ranking helpers for comparing saved runs under `artifacts/`.
- The tests now import directly from the concrete pipeline modules, so there is no separate workflow compatibility layer to maintain.
- `tests/`: regression checks for data loading, metric correctness, and submission-format validation.

## Execution surfaces

- Notebook-first exploration: `coursework_runner.ipynb` is the primary and intended convenience layer for guided execution, report-oriented visuals, and end-to-end coursework runs.
- CLI-based execution: the package CLI remains available as a lower-level interface for repeatable experiments and internal batch operations.
- Config-driven experiments: switching files in `configs/` lets you compare model variants without modifying source code.
- Local artifact review: the local `artifacts/` directory remains useful while experimenting, but the committed summary surface is `BestModel/` plus the figures in `report_assets/`.
- Historical runs are intentionally moved under `archive/` so the root `configs/` and `artifacts/` directories stay focused on the final deliverable path.
- The legacy PowerShell smoke-test helper is also archived so the root surface stays fully notebook-first.
- Built-in experiment ranking: the CLI now includes a `compare` subcommand to summarize and rank saved experiment metrics.

## Current baseline status

- The current task target is SemEval 2022 Task 4 Subtask 1 (`PCL` vs `No PCL`).
- The required submission files are `dev.txt` and `test.txt`.
- The current recommended local model is the calibrated artifact ensemble in `configs/best_artifact_ensemble.json`.
- The strongest single fine-tuned checkpoint remains `configs/best_roberta_finetune.json`.
- The strongest linear-probe variant remains `configs/best_roberta_probe.json`.
- The strongest non-transformer local model remains `configs/best_hybrid_ensemble.json`.
- The strongest pure sparse-score ensemble remains `configs/best_tfidf_ensemble.json`.
- The strongest single-model sparse baseline remains `configs/best_tfidf_svm.json`.
- The repository also includes an optional RoBERTa pipeline, but it depends on local pretrained weights or allowed Hugging Face downloads.

## Submission expectations

- The repository is organized to support the coursework flow from dataset inspection to final export.
- The spec-facing deliverables are `report.tex` (and its compiled PDF), `BestModel/`, `dev.txt`, and `test.txt`.
- The final deliverables are expected to include model artifacts, local evaluation outputs, and submission-ready prediction files.
- Submission files must remain spec-compliant: one prediction per line, with binary labels only (`0` or `1`).
- The canonical submission files remain at the repository root as `dev.txt` and `test.txt`, and mirrored copies are included in `BestModel/` to keep the final bundle self-contained.
- `BestModel/checkpoints/roberta_download/model.safetensors` is a large checkpoint file. If you push it to GitHub, use Git LFS or another large-file workflow instead of a normal Git push.
- The notebook is useful for report preparation because it groups tests, data analysis, training outputs, and plots in one place.
- The repository no longer relies on a separate root-level package shim for convenience; notebook usability is handled directly inside `coursework_runner.ipynb`.

## Notes

- The loader reads `data/dontpatronizeme_pcl.tsv`, `data/train_semeval_parids-labels.csv`, `data/dev_semeval_parids-labels.csv`, and `data/task4_test.tsv`.
- Binary labels follow the paper setup: original labels `0/1 -> 0`, `2/3/4 -> 1`.
- `tfidf_svm` prefers `scikit-learn` for a real TF-IDF + linear SVM, and falls back to a lightweight token-scoring baseline if `scikit-learn` is not installed.
- `roberta` now supports local-cache loading, optional Hugging Face downloads, and a fallback offline tiny model, but its usefulness still depends on available pretrained weights.
- If you want the strongest practical local result, use the calibrated artifact ensemble. If you want the strongest single trainable checkpoint, use the fine-tuned RoBERTa path.
