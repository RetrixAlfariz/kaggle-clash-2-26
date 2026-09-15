# Train-only document-structure diagnostic — results

**Decision: do not advance this fixed feature set to a dedicated structure-aware scorer or GCN.** Structural features add a small, statistically detectable candidate-level signal, but do not deliver the predeclared useful effect or stable hard-pair ordering gains. No dev, holdout, or test evaluation was used. This is a train-only probe result, not a Kaggle or exact-slot model result.

## Frozen comparison

The control contains34 columns: all eight P0 logits, candidate margins and fixed margin hinges, basic shape, and existing P0 newline/adjacency flags. The augmented probe contains the same34 plus19 raw-text structural scalars from the six requested families. Three document/group-preserving folds and seven label-specific logistic probes give42 fits across both conditions. Scaling is fitted only on fitting folds; the fixed ridge coefficient is0.001. All42 fits converged within10 iterations (maximum final gradient norm5.48e-9). No feature or solver setting was selected from outcomes.

P0 already trained on these documents. Out-of-fold probe predictions prevent probe-fit reuse, but do not make the underlying P0 scores out-of-sample. Interpret the results as conditional residual information in this train sample.

## Sample and grouping limitation

The deterministic sample contains **6,000 documents**, **397,157 unique labeled candidates** (74,305 correct; 322,852 incorrect), and **273,347 gold-versus-hard-negative pairs**. Thirty-two unaligned and two over-width gold spans were outside the candidate support. Each included candidate/pair has weight1. This enriched mixture is not the natural all-span population; its class prevalence, log-loss, and discrimination must not be treated as full-universe performance.

Family discovery used only train text: normalized duplicate grouping plus digit-masked word5-shingle MinHash retrieval and Jaccard>=0.8 verification. Across55,582 train documents it found55,581 components, one of size2, with341,543 retrieved comparisons but no verified near-template edges. The selected6,000 components were all singletons; each fold therefore contains2,000 documents and2,000 proxy families.

**Broader template-family isolation was not established.** The frozen algorithm was executed without relaxing its threshold. Group-preservation is mechanically correct, but this evaluation is effectively document-disjoint, not demonstrated unseen-template evaluation. The bootstrap likewise reduces to a paired document bootstrap in this sample.

## Pooled metrics

| Metric | Control | + Structure | Improvement | Paired95% interval |
|---|---:|---:|---:|---:|
| Candidate AUC |0.99754789|0.99756518|+0.00001729|[+0.00000663,+0.00002770]|
| Candidate log-loss (lower better) |0.04566273|0.04551194|+0.00015079 reduction|[+0.00009459,+0.00020368]|
| Relative log-loss reduction |—|—|0.3302%|[0.2078%,0.4458%]|
| Hard-pair ordering accuracy |97.591340%|97.590974%|-0.000366pp|[-0.012064,+0.012079]pp|

The pairwise difference is equivalent to one fewer correctly ordered pair in273,347 pairs. Both probes use identical gold/hard-negative pairs; ties receive half credit. Lower log-loss does not translate into better hard-candidate ordering here.

## Results by candidate label

Each entry shows control -> structure. These are candidate-label diagnostics, not inferred section annotations.

| Label | Candidates | Hard pairs | AUC | Log-loss | Pair accuracy (%) |
|---|---:|---:|---:|---:|---:|
|NAME|93,630|90,316|0.993364 -> 0.993371|0.086358 -> 0.086262|94.4052 -> 94.3930|
|DATE|57,947|56,261|0.997016 -> 0.997070|0.050995 -> 0.050657|99.0953 -> 99.1148|
|EMAIL|39,017|27,633|0.999415 -> 0.999415|0.015582 -> 0.015583|99.8299 -> 99.8227|
|PHONE|33,062|27,357|0.999129 -> 0.999112|0.021329 -> 0.021276|99.6272 -> 99.6198|
|ADDRESS|30,403|21,328|0.996670 -> 0.996684|0.052540 -> 0.052264|97.7869 -> 97.8198|
|USERNAME|101,351|26,820|0.998486 -> 0.998515|0.021790 -> 0.021744|99.1797 -> 99.1611|
|JOB_TITLE|41,747|23,632|0.996191 -> 0.996259|0.047324 -> 0.046928|99.2341 -> 99.2383|

| Label | AUC delta95% interval | Relative log-loss reduction95% interval (%) | Pair accuracy delta95% interval (pp) |
|---|---:|---:|---:|
|NAME|[-0.0000077,+0.0000203]|[+0.0319,+0.1910]|[-0.0421,+0.0155]|
|DATE|[+0.0000211,+0.0000877]|[+0.2911,+1.0011]|[-0.0036,+0.0486]|
|EMAIL|[-0.0000244,+0.0000268]|[-0.7888,+0.7710]|[-0.0217,+0.0071]|
|PHONE|[-0.0000481,+0.0000087]|[-0.3654,+0.8263]|[-0.0221,+0.0072]|
|ADDRESS|[-0.0000368,+0.0000621]|[+0.1357,+0.9225]|[-0.0095,+0.0799]|
|USERNAME|[-0.0000145,+0.0000782]|[-0.2738,+0.6564]|[-0.0488,+0.0074]|
|JOB_TITLE|[+0.0000055,+0.0001343]|[+0.3251,+1.3628]|[-0.0335,+0.0388]|

Only DATE, ADDRESS, and JOB_TITLE have positive pairwise point estimates. All seven label-level pairwise intervals include zero. These intervals are descriptive, without multiple-comparison correction, and no label-specific rescue experiment was selected.

## Fold stability

| Fold | AUC delta | Relative log-loss reduction (%) | Pair accuracy delta (pp) |
|---|---:|---:|---:|
|0|+0.00002174|+0.4185|+0.00665|
|1|+0.00001131|+0.3179|-0.01426|
|2|+0.00001798|+0.2546|+0.00653|

Log-loss improves in all three folds, but ordering worsens in fold1. Thus the conditional likelihood signal is small and relatively stable; the useful ordering signal is absent.

## Predeclared gate and its limitation

The frozen gate required at least+0.005 AUC,1% relative log-loss reduction, and+0.01 pairwise accuracy, positive lower95% bounds, positive log-loss/pair gains in all folds, and positive pair estimates in at least five supported labels. None of the three minimum-effect checks passed; fold/label stability and the joint-positive-interval check also failed.

**AUC ceiling caveat:** the control AUC of0.99755 leaves only0.00245 headroom, so the predeclared+0.005 AUC threshold was unattainable in this realized sample. We do not revise it post hoc, and we do not use that gate failure alone as evidence against structure. Independently, the upper95% bound for relative log-loss gain is0.4458%, below the1% minimum, and the upper bound for ordering gain is0.0121 percentage points, far below the1-point minimum. Those outcomes support stopping this branch under the specified diagnostic.

## Do gains survive the existing newline/score controls?

Yes, a tiny AUC/log-loss signal survives the specified controls; both probes contain the same raw logits, margins, score hinges, shape, and newline features. It would be inaccurate to claim structure contains no information. What does not survive is a practically useful, stable improvement in ordering gold spans against hard negatives.

All19 structural columns vary in the extracted sample. For example,8.94% of candidates cross a line and20.26% have a preceding colon on their start line. The null ordering result is not explained by an entirely absent/constant structural feature set. No semantic header/signature detector or text-keyword feature was included.

## Uncertainty and validation

Intervals use1,000 paired whole-component resamples of the fixed OOF predictions, seed20260916. Candidate and pair contributions are resampled together. Probes are not refitted. Intervals are conditional on the chosen sample, approximation to families, and P0 train exposure; they do not establish unseen-template or test performance.

Sixteen synthetic tests passed before extraction. Independent validation checked all candidate exact targets, document/cache alignment, family fold isolation, feature dimensions/finiteness,100 sampled newline/adjacency rows, and the complete hard-pair multiset. A separate rank-sum AUC calculation and direct log-loss/pair-win calculations reproduced both probes' pooled metrics to1e-12. Evidence is retained as `output/bitrase-2/structure-diagnostic/independent_validation.json`.

## Decision and artifacts

**Stop pursuing this predefined structural-feature scorer branch.** Do not add line embeddings, a signature-role module, or a GCN on the basis of these results. Do not mix these probes into bitrase-2.algo-2, posterior calibration, or the production decoder. No production scoring change was made.

This is a scoped negative decision, not proof that all document structure is useless. It applies to this feature set, low-capacity probe, enriched train mixture, and already-trained P0 representation. No extra feature sweep or stronger model is justified by the present evidence.

Files: [frozen protocol](protocol.md), [feature schema](feature_schema.json), [technicalities and commands](technicalities.md). Numerical results, per-label intervals, fit coefficients/scaling, OOF predictions, family assignments, source counts, and hash manifests are preserved in `output/bitrase-2/structure-diagnostic/`.

Local measured times: extraction/grouping59.63s,42 fits5.47s, metrics/bootstrap31.72s. No dev/holdout/test inputs, new pretrained models, Kaggle upload, or change to the existing P0/P1 artifacts occurred.
