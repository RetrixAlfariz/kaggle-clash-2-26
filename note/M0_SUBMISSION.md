# M0 test submission

`src/submit_m0.py` generates the test submission from the frozen M0 dictionary in `output/m0/v1/dictionary.parquet`. It loads only `test.parquet` and `test_slots.parquet` through `PreparedData`; it does not fit, tune, or read holdout data.

Run:

```text
uv run src/submit_m0.py
```

The output is `output/submissions/m0/submission.csv` with the exact columns `row_id,Predicted`, original sample-submission row order, and predictions formatted as `LABEL:start:end`. The script refuses to invent spans if the decoder cannot satisfy a document's exposed slot count, and writes the blocked-document count to `report.json` before stopping. `report.json` also records dictionary and prepared-manifest hashes plus the final CSV hash.

This is an M0 development baseline submission. It uses the exposed test entity count and should not be interpreted as blind unknown-count NER performance.
