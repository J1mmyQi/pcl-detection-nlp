This directory contains the final selected local model bundle for the coursework submission.

Contents:

- `best_artifact_ensemble.json`: the selected self-contained configuration for the best current local calibrated score ensemble.
- `best_roberta_finetune.json`: the selected configuration for the best current fine-tuned RoBERTa model.
- `best_roberta_probe.json`: the selected configuration for the best current local RoBERTa linear-probe model.
- `best_hybrid_ensemble.json`: the selected configuration for the best local hybrid sparse ensemble model.
- `best_tfidf_ensemble.json`: the strongest pure sparse-score ensemble kept for reference.
- `best_tfidf_svm.json`: the strongest single-model sparse baseline kept for reference.
- `metrics.json`: the latest local development-set metrics for the selected configuration.
- `bundle_manifest.json`: a compact manifest describing the bundled checkpoints, code snapshot, and submission files.
- `coursework_runner.ipynb`: a direct notebook copy for the final selected bundle.
- `code/pcl_detection/`: a direct code snapshot of the core implementation modules used by the final bundle.
- `checkpoints/`: the local checkpoint files required to reload the selected final model without depending on the gitignored `artifacts/` directory.
- `dev.txt` and `test.txt`: submission-ready copies of the final predictions.

Notes:

- The implementation code also remains available in `src/pcl_detection/`, but the `code/` snapshot in this directory is included so the final bundle is self-contained.
- The notebook runner also remains available at the repository root, but this directory includes a direct copy for the same reason.
- The selected ensemble can be reloaded directly from `BestModel/best_artifact_ensemble.json`, which points to the bundled `checkpoints/` paths instead of the gitignored `artifacts/` directory.
- The repository root keeps the canonical submission files in `dev.txt` and `test.txt`; this directory includes mirrored copies so the final bundle contains the same validated outputs.
- Older exploratory runs are kept separately under `archive/` so this directory remains focused on the final selection.
- The bundled RoBERTa checkpoint includes a large `model.safetensors` file; if you push this directory to GitHub, use Git LFS or another large-file workflow.
