This directory contains the final selected local model setup for the coursework submission.

Contents:

- `best_artifact_ensemble.json`: the selected configuration for the best current local calibrated score ensemble.
- `best_roberta_finetune.json`: the selected configuration for the best current fine-tuned RoBERTa model.
- `best_roberta_probe.json`: the selected configuration for the best current local RoBERTa linear-probe model.
- `best_hybrid_ensemble.json`: the selected configuration for the best local hybrid sparse ensemble model.
- `best_tfidf_ensemble.json`: the strongest pure sparse-score ensemble kept for reference.
- `best_tfidf_svm.json`: the strongest single-model sparse baseline kept for reference.
- `metrics.json`: the latest local development-set metrics for the selected configuration.

Notes:

- The implementation code lives in `src/pcl_detection/`.
- The notebook runner lives in `coursework_runner.ipynb`.
- The selected ensemble artifact is stored under `artifacts/best_artifact_ensemble/`.
- The strongest single fine-tuned RoBERTa checkpoint remains stored under `artifacts/roberta_download/`.
- The corresponding submission-ready prediction files are generated at the repository root as `dev.txt` and `test.txt`.
- Older exploratory runs are kept separately under `archive/` so this directory remains focused on the final selection.
