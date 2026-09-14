# bitrase-1.algo-2 — aborted run log

Status: **aborted; no usable result**. This experiment must not be treated as a model comparison or submission candidate.

## Attempt 1

- Started with the frozen M1 candidate pool and a LightGBM binary classifier from scratch.
- Training feature construction was too slow and was interrupted before fitting completed.
- No fitted model, dev report, or checkpoint was produced.

## Attempt 2

- The run was restarted with a deterministic cap of 12,000 training documents and at most 30 sampled negatives per document to bound compute.
- During implementation review, the initial dev path was corrected from `LGBMClassifier.predict()` (hard labels) to the underlying booster probability output. The corrected path was present in the restarted source, but the run was interrupted during feature construction before a model was fitted.
- Negative sampling had no inclusion-probability or class-prior correction. Even if it had completed, its probabilities would require explicit calibration or an honest sampling-bias qualification before decoder comparison.
- Only a partial `config.json` and the source-level experiment scaffold may remain. There is no valid `model.pkl`, `report.json`, or accepted dev score from this experiment.

The next implementation should use a more efficient feature-building path, freeze the chosen train cap before execution, and account for the negative-sampling design in the training objective or calibration. No Kaggle submission was generated.
