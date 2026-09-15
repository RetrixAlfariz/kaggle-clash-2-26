# bitrase-2.algo-1 implementation and verification

The experiment is specified by the [frozen protocol](protocol.md). The protocol's status describes the pre-run freeze; completed outcomes are recorded separately in the [report](report.md).

## Implementation

- `src/slot_mbr.py`: grouped-label float64 forward/backward partition, labeled slot marginals, exact slot-MBR DP, and independent MAP utility comparison. Partition uses label logsumexp; MAP uses maximum label energy. Strict greater-than comparisons preserve skip ties, followed by smaller start and lowest label index. Full label marginals are vectorized; default diagnostics are JSON serializable.
- `src/run_slot_mbr.py`: verification, immutable dev logits/candidate generation, full MAP replay, fixed-temperature decoding, independent structural/metric evaluation, paired bootstrap, and positional analysis. CPU decoding uses four processes; no gold is passed to workers. No training, test, holdout, or submission loader is called.
- `src/slot_diagnostics.py`: fixed-K gap accounting, common-anchor displacement episodes, and a disjoint nine-cell gold-entity transition table. Fixed MAP episode references partition into repaired, retained-displaced, or lost.

Original P0 source files and checkpoints are reused without modification. Frozen logits are generated from the verified native hidden-state cache using the same32-document length-sorted batching,32768-span chunks, CUDA FP16 autocast, and selected span head as P0. Eight raw logits are losslessly promoted to float32 and retained. MAP replay uses original P0 float32 score subtraction; posterior computation uses float64 direct logit differences.

## Verification and replay

14 tests passed before score extraction, including1,000 deterministic random exhaustive decoder fixtures and200 randomized positional-accounting cases. Exhaustive enumeration, independently indexed labeled-candidate prefix/suffix DP, and grouped token DP agree on partitions and full slot marginals. Tests also check MAP and MBR optima, label collapse, concentrated/uniform scores, extreme logits, adjacency, overlap, impossible K, K=0/K=T, trailing token gaps, deterministic ties, and replay canonicalization.

Central review corrected development issues before any real-data evaluation: a MAP diagnostic initially used label logsumexp instead of label maximum; tests were strengthened to check MAP predictions/scores and brute-force marginals explicitly. Episode segmentation was corrected to use common anchors rather than treating missing gold entities as zero-displacement boundaries. No dev scores informed these fixes.

All6,943 saved P0 dev predictions replayed exactly, including K, ordered character triples, and complete coverage. Normalizing only document serialization order produced the same canonical SHA256 for both artifacts:

```text
0f79816fa564ae4db43cd94d16e50611ae85062e0cca9aa76dac94a217e64b51
```

The Parquet bytes differ because serialization/document order differ. Canonicalization first rejects duplicate documents, duplicate or unordered entities, invalid labels/bounds, and overlaps; it never sorts an invalid entity sequence to conceal a mismatch.

## Frozen artifact

Location: `output/bitrase-2/algo-1/frozen/`. Contains25,347,544 intervals for6,943 dev documents: `logits.npy`, `pairs.npy`, token and span offsets, character boundaries, document metadata, and a manifest. Every consumer checks all file hashes and verified source hashes. Files have OS read-only flags, and extraction refuses to overwrite the artifact.

Observed dev maximums:449 tokens and30 entities per document. The implementation uses input dimensions, not the proposal's433-token training estimate. Extraction took11.64 seconds; MAP replay took14.35 seconds, excluding stage-start artifact validation.

## Commands

Executed from the repository root using the existing neural environment:

```powershell
.venv-neural/Scripts/python.exe -B src/run_slot_mbr.py verify
.venv-neural/Scripts/python.exe -B src/run_slot_mbr.py freeze
.venv-neural/Scripts/python.exe -B src/run_slot_mbr.py replay
.venv-neural/Scripts/python.exe -B src/run_slot_mbr.py run --tau 1 --workers 4
.venv-neural/Scripts/python.exe -B src/run_slot_mbr.py run --tau 0.5 --workers 4
.venv-neural/Scripts/python.exe -B src/run_slot_mbr.py run --tau 2 --workers 4
```

Stages refuse to overwrite existing artifacts. Sensitivity execution requires primary performance and mechanism artifacts first. For independent reproduction use a separate checkout/output directory; preserve the original evidence rather than deleting it.

Bootstrap uses10,000 paired document resamples, seed20260915, with pooled correct-slot counts divided by pooled K. Intervals are percentile95% intervals and do not correct repeated dev use. No temperature is fitted or selected.

## Diagnostic definitions and evidence files

Simple episodes are maximal nonzero common-anchor runs bounded by zero-displacement anchors or document sentinels, with all displacements either+1 or all-1. Other excursions are complex. Missing exact gold entities do not themselves reset displacement.

An MBR episode is reported as new/changed when its reference triple set or sign differs from every MAP simple episode. This includes shortened or modified episodes and is not automatically newly introduced harm. Actual harm is measured separately by retained correct-to-displaced and correct-to-lost transitions.

Per-temperature artifacts under `output/bitrase-2/algo-1/tau{1,0.5,2}/` retain predictions, per-document scores, performance/bootstrap, decoder numerical residuals, per-document positional diagnostics, aggregate counts, and episode-length histograms. Performance contains measured wall time and process-memory data; decoder workspace bytes cover explicitly counted NumPy arrays, not total process memory. Worker peak working sets include mmap pages and imports and must not be added as unique memory because shared pages may be counted repeatedly.
