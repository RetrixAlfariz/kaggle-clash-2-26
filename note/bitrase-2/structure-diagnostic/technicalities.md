# Train-only structure diagnostic — technicalities

This diagnostic is separate from bitrase-2.algo-2, calibration, Slot-MBR, joint training, and production model changes. See the [frozen protocol](protocol.md), [frozen scalar schema](feature_schema.json), and [results](report.md).

## Frozen model and data path

Only `PreparedData.load_train()` is called. The selected P0 epoch2 head is hash-checked against `520646a7295797707e11fa24a8f06a5670c9d3f3201e81cf911ff11ec15b3e06`. Train hidden states, boundaries, offsets, and old BIO predictions are verified against the P0 train-cache manifest; that manifest is bound to P0's original training configuration. Provenance checks deliberately skip the cache's dev-file hash rather than opening dev. No P0 weights are modified.

Required span intervals are scored using frozen P0 features and CUDA FP16 autocast, with native logits promoted to float32 then float64 for probes. The diagnostic scores only the union of required sampled intervals, not all train intervals. Full-universe train span logits were not previously stored for byte-level replay; the selected head/cache and its feature computation are preserved, while batch matrix shapes differ from full-universe inference.

All eight logits are retained as control columns. Candidate-minus-NONE and candidate-minus-best-other margins and fixed hinge terms are derived before the basic controls. Basic controls and structural values come from `DocumentLayout` and raw offsets, with no annotation-dependent section labels.

## Implementations

- `src/document_structure.py`:21 shape/newline/adjacency controls and19 structural scalars. Token and character lengths use log1p; positions/distances are normalized. Paragraphs separate at blank physical lines. Colon features use boundary lines, and absence is represented explicitly by the preceding-colon flag and sentinel distance. Feature schema is frozen before extraction.
- `src/structure_families.py`: train-only normalized duplicates plus digit-masked word5-shingle MinHash retrieval/Jaccard verification, complete-component sampling, and group-preserving folds. No preexisting cross-split audit edges are reused.
- `src/structure_probe.py`: float64 ridge-logistic Newton solver with deterministic backtracking;0.001 regularization on standardized coefficients, no intercept penalty, and maximum100 iterations. Feature means/scales use fitting folds only and are saved alongside coefficients.
- `src/run_structure_diagnostic.py`: staged freeze/extract/fit/evaluate commands; deduplicated labeled candidates and fixed hard pairs; out-of-fold prediction storage; exact weighted AUC and paired family bootstrap. Existing environment/dependencies are used without installs.

The control has34 columns (13 score-derived plus21 shape/newline/adjacency). The augmented probe has53 columns (same34 plus19 structure). Seven separate label-specific probes are fitted for each of three folds and two feature conditions:42 fits total. This permits label-specific structural effects without using the gold entity class as an inferred region feature.

## Sample and target population

Grouping examines55,582 train documents. The frozen retrieval produced341,543 candidate comparisons and no verified Jaccard>=0.8 near-template edges. One duplicate/numeric-equivalent component had size2. The deterministic sample selected6,000 singleton components, assigned2,000 per fold. Broad template-family isolation is therefore unresolved; the evaluation is effectively document-disjoint probe evaluation. No threshold was relaxed after inspection.

The sample contains397,157 unique labeled candidates and273,347 gold-versus-hard-negative pairs. Candidates are an enriched diagnostic mixture, not a probability sample of the entire labeled-span universe. Each included candidate and pair receives weight1; AUC/log-loss must be interpreted for this mixture. Source bitmasks overlap after deduplication, and a true gold triple remains positive regardless of how it entered the pool.

At most four hard negatives per document/label are selected by frozen P0 energy, with candidate row order breaking ties; each eligible gold candidate is paired with each selected negative. Hard negatives come from boundary perturbations, old-model candidates, or alternative-label candidates. Random-only candidates never enter hard pairs. Both probes are evaluated on exactly the same fixed pairs.

## Validation and uncertainty

16 synthetic tests passed before extraction: feature/newline parity, line/paragraph/colon boundaries including CRLF, metric ties/weights, train-only scaling, probe fitting, family grouping and fold isolation, and bootstrap metric direction. The exact vectorized weighted AUC used for bootstrap was checked against the independently implemented rank/tie function over100 random fixtures.

Bootstrap uses1,000 whole-component multiplicity resamples with seed20260916. Since selected components are singletons, this equals a paired document bootstrap for this sample. Candidate and pair contributions remain together. Probes are not refitted for each resample, and the intervals do not account for P0's exposure to train or unknown broader template dependence. Per-label intervals are descriptive; the advance decision uses the predeclared pooled and stability gates.

## Commands and artifacts

```powershell
.venv-neural/Scripts/python.exe -B src/run_structure_diagnostic.py freeze
.venv-neural/Scripts/python.exe -B src/run_structure_diagnostic.py extract
.venv-neural/Scripts/python.exe -B src/run_structure_diagnostic.py fit
.venv-neural/Scripts/python.exe -B src/run_structure_diagnostic.py evaluate
```

Artifact root: `output/bitrase-2/structure-diagnostic/`. `frozen.json` binds protocol, schema, sources, and passing tests. `grouping.json` records discovered component sizes and sampled memberships. `data/` contains features, targets, labels, candidate offsets/source tags, fixed pairs, documents/folds, and a manifest. `probes/` contains OOF probabilities, fitted coefficients/scaling, and a manifest. `results.json` records pooled/label/fold metrics, paired intervals, source counts, and every gate outcome. Stages refuse to overwrite their completed artifacts.

Train-cache verification, grouping, feature extraction, and scoring took59.63 seconds locally. Timings are descriptive, not cross-system benchmarks. Future work must preserve the frozen evidence rather than silently editing feature definitions in place.
