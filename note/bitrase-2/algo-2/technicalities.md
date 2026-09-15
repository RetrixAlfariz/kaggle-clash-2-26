# Posterior audit technicalities

New diagnostic code only:
- `src/posterior_math.py`: float64 forward/backward exact-K labeled interval DP;
  zero-energy count DP; slot marginals; H=logZ−expected scaled energy.
- `src/run_posterior_audit.py`: four CPU workers read mmap P1 logits/pairs and
  saved predictions; dev gold is loaded explicitly for evaluation. No decoding
  or parameter optimization. Aggregates reliability and paired document bootstrap.
- `src/verify_posterior_audit.py`: independent historical hash, coverage, count,
  prediction order/nonoverlap, correctness and transition verification.
- `tests/test_posterior_math.py`: brute-force labeled-world enumeration, original
  decoder agreement, uniform scores, empty structure, infeasibility and tau checks.

Commands from repository root, using the existing neural environment:

```powershell
.venv-neural/Scripts/python.exe -B -m unittest tests.test_posterior_math -v
.venv-neural/Scripts/python.exe -B src/run_posterior_audit.py
.venv-neural/Scripts/python.exe -B src/verify_posterior_audit.py
```

The main runner refuses an existing `output/bitrase-2/algo-2` directory. Preserve
the completed run; a future rerun needs a separately reviewed output destination.
Do not delete or overwrite evidence to rerun it. Source/protocol/input hashes are
in `input_manifest.json`; the frozen P1 validator checks its original data and
source hashes before and after. Original P0/P1 files remain unchanged.

Artifacts in `output/bitrase-2/algo-2`:
- `documents.parquet`: 20,829 rows, one per document/temperature; entropy,
  log count, normalized measures, utility/correctness differences and churn.
- `slots.parquet`: 512,016 rows, document/temperature/rule/slot, confidence,
  correctness, selected/gold label and fixed stratification fields.
- `summary.json`: aggregate bootstrap, reliability bins and strata, paired
  correctness strata, unchanged frozen P1 mechanism summaries.
- `validation.json`: original inputs unchanged and numerical invariants passed;
  output hashes for the main run.
- `independent_validation.json`: separate historical P1 hash and row/count checks.

Units: entropy/count use nats; confidence, accuracy, ECE, Brier and JSON gains use
fractions. Report gains multiply JSON values by 100 to express percentage points.
Means of per-document H/K and H/log count differ from ratios of pooled totals.
Mean MAP probability is not exp(mean MAP log probability).

All candidate marginals are temporary worker arrays; only selected-slot marginals
are persisted. The complete support is exactly the frozen width<=16 universe,
with seven labels. Count includes label assignments and treats chronological
nonoverlapping interval collections as unique structures, without counting skip
operations as extra structures. Zero K has one empty structure in tests; all
audited documents have positive K.

The report is descriptive on a repeatedly used dev set. No bin or temperature
optimization, statistical claims per label, new score fitting, or unseen-template
generalization claims are made. No web access or related-paper model is needed.
