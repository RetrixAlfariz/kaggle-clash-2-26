# P0 technicalities

Files:

- `src/build_span_cache.py`: baseline source/checkpoint verification, external pre-classifier hook, native hidden cache, exact logits replay, and full dev baseline prediction replay.
- `src/span_probe.py`: complete span universe, gold protection, deduplicated sampling with inclusion weights, trainable head, grouped interval DP.
- `src/run_span_probe.py`: mmap cache verification, train-only sampling plans, three-epoch head fitting, evaluation/checkpoints, paired document bootstrap.
- `src/predict_span_probe.py`: hash-verified discovery inference, full raw-dev replay, test prediction and manual CSV.
- `tests/test_span_probe.py`: complete candidate indexing, protected/positive priority, HT sampling sanity, exhaustive decoder equivalence, prefix means, finite gradients, and chunked weighted gradient equivalence.

Use the existing `.venv-neural` runtime: PyTorch2.8.0+cu128,NumPy2.5.3,PyArrow25.0.1 on RTX5060Ti16GB. Baseline module hashes are preserved. There is no dependency migration or downloaded pretrained weight.

```powershell
.venv-neural/Scripts/python.exe -B -m unittest discover -s tests -p test_span_probe.py
.venv-neural/Scripts/python.exe -B src/build_span_cache.py --split dev
.venv-neural/Scripts/python.exe -B src/build_span_cache.py --split train
.venv-neural/Scripts/python.exe -B src/run_span_probe.py
```

Cache output: `output/bitrase-2/p0/cache/{train,dev}`. Native FP16 states are stored in flat hidden.npy with boundaries.npy and offsets.npy. Documents retain prepared row order; encoder forwards follow the original length-sorted32-document batches. Train also saves epoch4 exact-K predictions. Each completed split has an artifact-hash manifest; the head loader checks every cache file and parent provenance before fitting.

Dev cache completed with zero logit replay difference and all6,943 known-K predictions identical to the saved baseline. Native states require about7.03GiB total train+dev, plus0.11GiB token boundaries and metadata. No lossy downcast was introduced. The first cache attempt failed before model inference completed because the replay invoked the extraction hook twice; fixed by detaching the hook before replay. Its partial cache is preserved as `dev_failed_hook`; it has no complete manifest and is never loaded for training.

The inference decoder groups candidates by ending token, yielding the same objective as interval-prefix DP at O(T*W*K). Original character offsets are taken directly from cached token bounds. Infeasible counts or nonfinite scores fail explicitly; no BIO fallback or fabricated predictions.

Every seed output is separate. Checkpoint and prediction hashes identify evaluated artifacts. Bootstrap draws10,000 paired document samples; it does not correct checkpoint selection bias or dependencies between templates. No Kaggle upload is performed by these scripts.

The initial discovery attempt stopped on the first batch before an optimizer update because a hard gradient check prevented GradScaler's normal overflow recovery. Its config/sampling artifacts and exact failed runner are preserved in `seed2026_failed_amp`. A replay of the same real32-document train batch confirmed scale65536 overflowed, scale32768 succeeded, and exactly one optimizer update occurred after retry. The corrected runner retries the same batch/dropout realization with reduced scale, logs retries separately, and starts at1024. This numerical repair did not use dev results or modify sampling/objective.

Completed follow-up commands:

```powershell
.venv-neural/Scripts/python.exe -B src/run_span_probe.py --seed 3407 --epochs 2 --replication
.venv-neural/Scripts/python.exe -B src/run_span_probe.py --seed 1337 --epochs 2 --replication
.venv-neural/Scripts/python.exe -B src/predict_span_probe.py
.venv-neural/Scripts/python.exe -B src/verify_submission.py --submission output/bitrase-2/p0/seed2026/submission.csv --report output/bitrase-2/p0/seed2026/submission_validation.json
```

The predictor compares replay by document ID, since prepared data and length-sorted evaluation Parquet have different row orders. CSV includes only row_id/Predicted in original slot order. It verifies source hashes before write and refuses overwrite. Native encoder replay and all selected span predictions were identical; no numerical tolerance was used to excuse decision changes.
