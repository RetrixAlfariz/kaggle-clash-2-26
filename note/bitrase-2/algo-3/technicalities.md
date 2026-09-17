# Train-only scale/ranking diagnostic — technicalities

This experiment uses a newly trained replica of the P0 architecture. It does not
recalibrate the original P0 checkpoint and does not produce a new Kaggle result.
The split is a single three-way TRAIN partition, not cross-validated OOF.

## Execution

From the repository root, using the existing CUDA environment:

```powershell
.venv-neural/Scripts/python.exe -B src/train_scale_ranking.py
.venv-neural/Scripts/python.exe -B src/verify_scale_training.py
.venv-neural/Scripts/python.exe -B -m unittest tests.test_scale_ranking_audit -v
.venv-neural/Scripts/python.exe -B src/scale_ranking_audit.py --cal output/bitrase-2/algo-3/CAL --eval output/bitrase-2/algo-3/EVAL --out output/bitrase-2/algo-3/audit
.venv-neural/Scripts/python.exe -B src/verify_scale_audit.py
```

Both training and audit refuse existing output directories. Preserve completed
evidence; a future rerun requires its own output location and protocol.

## Isolation and lineage

`train_scale_ranking.py` loads prepared TRAIN only. Grouping uses document text
and IDs; original train indices and detected family IDs are saved. The vocabulary
is learned only from FIT. The encoder is newly initialized, trained with BIO
supervision for four epochs and frozen. A freshly initialized P0 span head is
trained for two epochs using FIT hidden states, gold and fresh FIT BIO predictions
for forced negative sampling. The existing inverse-probability sampling and
optimizer routine are reused without invoking their original experiment runners.

CAL/EVAL token features have dummy targets. Frozen inference scores and metadata
are generated after head training, before the diagnostic starts; EVAL gold is
packaged as metadata but is not used to generate its scores or fit any weights.
The audit opens CAL first, writes the fitted scalar, then opens EVAL for analysis.
The procedural boundary is between inference/data preparation and diagnostic
evaluation, not a claim that EVAL text was never read by the inference pipeline.

Training source hashes, the original TRAIN hash, split assignment hash, vocabulary
and checkpoint hashes are saved. Each CAL/EVAL cache includes a manifest over its
files. The independent verifier reconstructs the family-level assignment and
checks coverage, disjointness and producer hashes. Hashes document lineage;
they cannot by themselves establish absence of leakage, which also requires the
source review recorded in the report.

## Statistical meaning

The scalar fit minimizes complete-structure NLL conditional on known K and the
frozen candidate universe. It uses only documents whose entire gold structure
is representable. For unsupported gold, true structured NLL is infinite;
conditional finite NLL statistics must always disclose their denominator.
Exact-slot decisions retain all EVAL documents.

The CAL optimizer computes log partition and expected raw energy together using
a float64 expectation-semiring DP. EVAL uses a bounded four-worker CPU pool.
The strongest wrong structure comes from an exact top-two DP retaining two
distinct paths and the top two labels per interval; alternative label assignments
are not collapsed away. Tests enumerate complete tiny worlds including ties and
check an analytic interior temperature optimum and boundary optima. Independent
validation recomputes ordered-slot counts, gold and MAP energies, per-label
metrics, and pooled NLL, and checks 20 supported documents' NLL against the
separate forward/backward posterior implementation.

Gold-vs-strongest-wrong structure ranking includes alternative labels as well as
alternative intervals. Local span ranking holds all other gold entities fixed;
its feasible replacements lie between the neighboring gold intervals and may
change label at the same boundary. This is an optimistic, gold-conditioned
diagnostic, not a deployable decoder or an estimate of leaderboard headroom.

Positive scalar scaling preserves complete-structure energy ordering and MAP
decisions; MBR can change because posterior mass across structures changes.
Consequently NLL, ranking and exact-slot MBR effects are reported separately.
Decision bootstrap resamples documents and pools correct-count differences over
their K totals; relative NLL bootstrap pools document NLL sums. Neither bootstrap
captures refitting the temperature, retraining, split variation or repeated prior
research decisions. Per-label diagnostics are descriptive.

The fixed architecture, epochs and training settings inherit earlier research
choices. This train-only evaluation does not erase that history or make an
external generalization claim. No original dev, holdout or test artifact is used.
