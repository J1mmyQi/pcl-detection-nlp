# pcl-detection-nlp

Project scaffold for the SemEval 2022 PCL binary classification coursework.

The current structure is aligned with the coursework spec in `70016_1_spec.pdf` and the baseline direction in `paper/2020.coling-main.518.pdf`:

- task focus: SemEval 2022 Task 4 Subtask 1 (`PCL` vs `No PCL`)
- required outputs: `dev.txt` and `test.txt`
- baseline-ready modules: data loading, binary label conversion, EDA, evaluation, submission export
- model slots: a working `tfidf_svm` baseline and a `roberta` training stub for the next iteration

## Structure

```text
coursework_runner.ipynb
configs/
  baseline.json
  best_tfidf_svm.json
  exp_*.json
basic_tests.ps1
src/
  pcl_detection/
    cli.py
    config.py
    data_pipeline.py
    models.py
    training_pipeline.py
    workflow.py
tests/
  test_data.py
  test_export.py
  test_metrics.py
```

## Project organization

- `coursework_runner.ipynb`: the main interactive notebook for step-by-step coursework execution, including tests, EDA, training, export, and figures.
- `configs/`: experiment configurations. `best_tfidf_svm.json` stores the current recommended tuned baseline, while the `exp_*.json` files capture comparison runs.
- `basic_tests.ps1`: a small helper script for the most common test groups.
- `src/pcl_detection/data_pipeline.py`: data loading, label conversion, preprocessing, and EDA logic.
- `src/pcl_detection/models.py`: model definitions, including the tuned TF-IDF + SVM baseline and the RoBERTa scaffold.
- `src/pcl_detection/training_pipeline.py`: training, evaluation, error analysis, and submission export logic.
- `src/pcl_detection/workflow.py`: compatibility layer that re-exports the older workflow API used by the current tests.
- `tests/`: regression checks for data loading, metric correctness, and submission-format validation.

## Execution surfaces

- Notebook-first exploration: `coursework_runner.ipynb` is the primary and intended convenience layer for guided execution, report-oriented visuals, and end-to-end coursework runs.
- CLI-based execution: the package CLI remains available as a lower-level interface for repeatable experiments and internal batch operations.
- Config-driven experiments: switching files in `configs/` lets you compare model variants without modifying source code.
- Artifact review: the `artifacts/` directory is the main place to compare metrics, saved models, EDA summaries, and dev-set error examples.

## Current baseline status

- The current task target is SemEval 2022 Task 4 Subtask 1 (`PCL` vs `No PCL`).
- The required submission files are `dev.txt` and `test.txt`.
- The current recommended baseline is the tuned TF-IDF + SVM configuration in `configs/best_tfidf_svm.json`.
- The repository also includes a RoBERTa training scaffold, but it is still a placeholder rather than a completed fine-tuning pipeline.

## Submission expectations

- The repository is organized to support the coursework flow from dataset inspection to final export.
- The final deliverables are expected to include model artifacts, local evaluation outputs, and submission-ready prediction files.
- Submission files must remain spec-compliant: one prediction per line, with binary labels only (`0` or `1`).
- The notebook is useful for report preparation because it groups tests, data analysis, training outputs, and plots in one place.
- The repository no longer relies on a separate root-level package shim for convenience; notebook usability is handled directly inside `coursework_runner.ipynb`.

## Notes

- The loader reads `data/dontpatronizeme_pcl.tsv`, `data/train_semeval_parids-labels.csv`, `data/dev_semeval_parids-labels.csv`, and `data/task4_test.tsv`.
- Binary labels follow the paper setup: original labels `0/1 -> 0`, `2/3/4 -> 1`.
- `tfidf_svm` prefers `scikit-learn` for a real TF-IDF + linear SVM, and falls back to a lightweight token-scoring baseline if `scikit-learn` is not installed.
- `roberta` is intentionally a scaffold: the interface and config path are in place, but actual Hugging Face fine-tuning still needs to be filled in next.
- If you later implement the `roberta` path, add `transformers` and `torch` at that stage instead of putting them in the base install now.
