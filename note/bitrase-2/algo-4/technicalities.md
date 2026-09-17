# Algo-4 execution and reproduction

Python: existing `.venv-neural/Scripts/python.exe`, no dependency installation. CUDA RTX 5060 Ti. Native Windows memory accounting uses GetProcessMemoryInfo; the sum of parent peak RSS and four times maximum worker peak RSS is a conservative estimate, not a simultaneous process-tree peak.

## Entry points

Run from repository root, in order:

```powershell
.venv-neural/Scripts/python.exe -B src/train_global_ranking.py prepare
.venv-neural/Scripts/python.exe -B src/train_global_ranking.py benchmark
.venv-neural/Scripts/python.exe -B src/train_global_ranking.py train
.venv-neural/Scripts/python.exe -B src/train_global_ranking.py score
.venv-neural/Scripts/python.exe -B src/evaluate_global_ranking.py --control output/bitrase-2/algo-4/run/EVAL/control --challenger output/bitrase-2/algo-4/run/EVAL/challenger --out output/bitrase-2/algo-4/run/EVAL/evaluation
.venv-neural/Scripts/python.exe -B src/verify_global_ranking.py
```

These commands refuse completed outputs; they are not instructions to overwrite the recorded experiment. Exact preparation/configuration, training inputs, checkpoint hashes and CE replay digests live in `output/bitrase-2/algo-4/run/`. No training command loads original dev/holdout/test.

The fresh encoder reuses the verified implementation in `train_scale_ranking.py` with a new output directory; it does not load algo-3 parameters. FIT BIO predictions used for forced-negative CE sampling come from the fresh FIT2 encoder. Encoder activations are cached at the inherited float16 storage precision, then converted to float32 for both heads. Head computation and optimization are FP32 with TF32 disabled.

## Exact mining

`global_ranking_dp.py` stores top-two distinct paths at each token-prefix/entity-count state. A skip path cannot coincide with a path whose final span ends at the current token. Distinct last spans, labels and predecessor ranks likewise yield distinct structures. Thus retaining two predecessor paths and the two strongest labels per interval is sufficient for the global top two. This is exact top-two computation over the complete label universe, not heuristic candidate pruning.

Candidate-energy construction is vectorized across entity counts; tie order is documented in the module. Enumerated tiny cases test top-two scores, feasibility and path uniqueness, including ties and alternative labels. Final MAP/MBR decoding retains the pre-existing decoder convention.

`benchmark_before_vectorization.json` records an intermediate exact implementation. `benchmark.json` records the final optimized implementation on the same deterministic 256 FIT2 documents. Neither benchmark uses entity labels for performance selection. Real token lengths and expected K determine computational shape; synthetic hidden states and a random head avoid requiring training before the runtime gate.

## CE and ranking

The CE digest includes document order, candidate intervals, document indexes, labels, inverse-probability weights and per-batch RNG seeds. The control and challenger must have identical per-epoch digests. A synthetic test independently checks identical dropout-enabled CE outputs after an intervening dropout-disabled ranking forward.

Ranking forwards score only the selected gold and mined competing structures, with dropout disabled and gradients enabled. Shared spans cancel in the energy difference. A gradient test compares the implementation against a direct energy-difference expression, including shared-span cancellation and division by K. Unsupported gold documents contribute only CE.

`lambda.json` records the initial batch norms and the ratio-of-medians estimate. Monitoring uses the same 512 documents, fixed CE samples and each epoch's frozen competitors. No lambda fitting uses evaluation labels. Saved epoch-1 heads are provenance artifacts; evaluation uses epoch 2 only.

## Evaluation

The evaluator validates complete paired candidate/boundary/metadata coverage and frozen array hashes. It independently re-mines strongest wrong structures from final scores using the prior audit's exact helper, then reports MAP and Slot-MBR tau=1 on all 3,000 EVAL2 documents. Unsupported complete gold structures remain in exact-slot metrics and are identified separately in ranking diagnostics.

`eval_documents.json` retains gold slots and all four chronological prediction sequences for independent reconciliation. Confidence intervals resample whole discovered families and pool correct slots over all sampled slots. Each arm/decoder's absolute score and the paired improvement have different meanings; crossing 82% internally does not establish a Kaggle score.

The independent verifier checks split/family eligibility, input/checkpoint hashes, CE digests, lambda arithmetic, every stored competitor's feasibility and incorrectness, finite final parameters, and exact-slot/entity counts reconstructed from saved predictions.
