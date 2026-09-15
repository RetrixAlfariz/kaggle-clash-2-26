# bitrase-2.algo-1: Frozen Span Slot-MBR

Status: frozen research protocol; implementation-ready, not executed. Experiment alias: P1. This document consolidates Aestem's proposal and the amendments accepted by Vian and Opheline.

## Question and frozen reference

Does exact slot-aware minimum Bayes-risk decoding improve exact-slot accuracy over interval MAP when both use identical P0 scores and candidates?

Reference: P0 discovery seed2026, span-head epoch2, with its frozen epoch4 encoder. Saved dev results: 80.9354% exact-slot, 93.1740% exact entity F1; 69,067 correct slots and 79,511 exact unordered entities among 85,336 slots in 6,943 documents. P0's user-reported Kaggle score of0.805 remains the external reference. See [P0 report](../p0/report.md).

Freeze encoder, head, tokenizer, token/character boundaries, all token intervals of width1–16, eight raw logits per interval, seven entity labels, document-provided K, and exact-K/non-overlap constraints. Preserve all dev gold in evaluation, including gold outside candidate support. No gold enters decoding.

No retraining, candidate changes or pruning, score reweighting, temperature selection, calibration fitting, reserved holdout access, or Kaggle submission belongs inside P1. Sensitivity runs are diagnostics, not alternate promotion candidates.

## Structured model and exact decision

For token span s=[a,b), entity label l, and positive temperature tau:

```text
e[s,l] = (z[s,l] - z[s,NONE]) / tau
A[s] = logsumexp_l e[s,l]
P(Y | x,K) = exp(sum_(s,l in Y) e[s,l]) / Z
```

Y ranges over chronological, exactly-K, non-overlapping labeled-span sequences. MAP maximizes summed energies. Slot-MBR maximizes the sum of the selected labeled spans' slot-specific posterior marginals. This is optimal for expected correct-slot count under this assumed distribution, not necessarily the unknown true distribution.

Use log-space forward and backward partitions, with impossible states at negative infinity:

```text
F[0,0] = 0; B[T,0] = 0
F[t,k] = logsumexp(F[t-1,k], {F[a,k-1] + A[a,t]})
B[t,k] = logsumexp(B[t+1,k], {A[t,b] + B[b,k-1]})
logZ = F[T,K] = B[0,K]
log mu[s,l,j] = F[a,j-1] + e[s,l] + B[b,K-j] - logZ
```

Take branches apply only for k>0 and valid spans of width1–16. Skip propagation gives F[t,0]=B[t,0]=0. Each feasible slot's labeled marginals must sum to1.

Labels must be summed within the partition. Because labels do not affect compatibility, final rewards may use r[s,j]=max_l mu[s,l,j], retaining the corresponding label. The reward DP is:

```text
R[t,0] = 0
R[t,k] = max(R[t-1,k], {R[a,k-1] + r[[a,t),k]})
```

Backtrack R[T,K] for the valid MBR decision. Maximize sums of probabilities, not sums of log marginals. Infeasible K is a hard failure; no placeholders or fallback predictions. Specify deterministic tie-breaking before real-data evaluation; MAP must preserve P0's tie-breaking.

The labeled-candidate reference implementation consumes direct energies, with no sigmoid, probability clipping, or Bernoulli log-odds conversion. Grouped-label logsumexp is the equivalent production formulation. Measure runtime and memory using actual input dimensions; do not hard-code a training-set maximum as an inference limit.

## Execution order and verification gates

1. Exhaustive tiny-case verification. Compare brute-force enumeration, direct-energy labeled-candidate DP, and grouped-label DP on at least1,000 deterministic randomized fixtures with T<=6, W<=3, K<=3, L<=3. Verify logZ, forward/backward agreement, labeled slot marginals, per-slot normalization, MAP optimum, and MBR expected-utility optimum. Compare decisions modulo exact ties. Use float64 verification. Include adjacency, overlap, multiple labels per boundary, equal-score ties, K=0, K=T, impossible K, extreme logits, concentrated and nearly uniform posteriors, and explicit MAP/MBR and label-collapse counterexamples.
2. Freeze immutable P0 dev logits and candidates. Record document IDs, K, token boundaries, every candidate interval and its eight raw logits, label order, numeric precision, and source/checkpoint/artifact hashes. All decoders consume this same artifact. No lossy score conversion is permitted to silently change the reference.
3. Exact prediction-level MAP replay. Compare against the saved P0 epoch2 predictions for every dev document: identical entity counts, identical (start,end,label) triples, identical chronological ordering, and complete document coverage with no duplicates or extras. Require byte-identical content/hash where serialization permits; otherwise compare independently canonicalized records row-for-row and record canonical hashes. Canonicalization may normalize document serialization order, but must not conceal incorrect entity ordering, duplicate documents, or missing records. Matching only80.9354% is insufficient. Any document mismatch stops P1 before MBR evaluation; investigate candidates, scores, tie-breaking, or implementation.
4. Primary Slot-MBR at tau=1, with no fitted calibration.
5. Exact-slot performance comparison against replayed P0 MAP.
6. Repaired / retained-displaced / lost / newly introduced positional-error analysis.
7. Fixed tau=0.5 and tau=2 sensitivity diagnostics. Report both; never select the best temperature on dev.

### Complete label-collapse regression fixture

T=3, K=2, labels A/B:

| Span | Energy A | Energy B |
|---|---:|---:|
| [0,1) | 0.5 | 1.0 |
| [1,2) | 1.5 | -1.5 |
| [2,3) | -0.5 | 0.5 |
| [0,2) | -1.0 | 0.0 |
| [1,3) | -0.5 | -1.0 |

Full-posterior MBR selects ([0,1),B),([1,2),A), with utility0.8509544260. Collapsing labels before the partition selects ([0,1),B),([2,3),B); its utility under the full posterior is0.8021515788. Independently enumerated during protocol review.

## Performance evidence

Primary metric: exact-slot accuracy over the full dev denominator. Secondary metrics: exact entity F1, boundary F1, exact-document rate, per-label slot accuracy, and P0-compatible error decomposition. Preserve metric definitions and report raw counts.

Research-positive requires both a gain of at least0.20 percentage points over P0 MAP and a strictly positive lower bound of a95% paired document-bootstrap interval. Use10,000 paired resamples, calculating total correct slots divided by total slots in each resample, not an unweighted average of document accuracies. Freeze and record the bootstrap RNG seed before evaluation. Report observed delta, bootstrap median, and2.5th/97.5th percentiles.

This performance gate is separate from the mechanism assessment. Entity F1 need not increase. Repeated use of dev remains a limitation; bootstrap does not restore blind validation. Record every real-data evaluation.

## Positional mechanism evidence

For exact triples common to prediction and gold, define displacement as predicted rank minus gold rank. Consecutive common entities, with document sentinels, define gaps. If a gap contains P non-common predictions and G non-common gold entities, its imbalance is P-G. Cumulative imbalance before a common entity must equal that entity's rank displacement.

Identify simple shift episodes as departures from zero to+1 or-1 displacement, a run of displaced common entities, and return to zero without excursions to other displacement values. Report complex displacements separately. Fix these definitions before dev comparison.

For each MAP episode, keep its exact triples as a fixed reference set. Classify every reference entity under MBR as:

- REPAIRED: the exact triple remains and now occupies the correct slot.
- RETAINED-DISPLACED: the exact triple remains at an incorrect slot.
- LOST: the exact triple is no longer selected.

These mutually exclusive counts must sum to the original reference-set size. Complete repair requires every reference entity to be REPAIRED. Partial repair rates always accompany all three counts; losing an entity never counts as repairing it. Report new displacements, episode lengths, absolute-displacement-one counts, and documents affected. Define matching of episodes and attribution of new episodes deterministically before evaluation; report entity-level transitions as the unambiguous base accounting.

Report performance and mechanism separately. Accuracy can improve without strong support for the simple-shift explanation. Fewer wrong-rank entities alone is insufficient if exact entities were lost.

## Interpretation and future research boundaries

A positive result means Slot-MBR improved decisions under these frozen P0 scores on this repeatedly used dev set. It does not establish posterior calibration or guaranteed Kaggle improvement.

A negative result means only that this posterior and decision rule did not provide sufficient improvement. It does not isolate calibration, representation, head quality, missing interactions, or candidate support as the cause.

Temperature sign reversals indicate sensitivity to posterior concentration; they do not authorize selecting a diagnostic temperature. Future calibration is a separate experiment: calibration documents must be excluded from all supervised encoder/head training. Structured NLL requires gold sequences in candidate support; any restriction to fully representable documents must be reported, including exclusions and its effect on scope.

## Target-scale illustration: 88% and 92%

On85,336 dev slots, P0 has69,067 correct slots and10,444 exact-but-misranked entities. Reaching at least88% requires75,096 correct slots (+6,029);92% requires78,510 (+9,443). These gains correspond to approximately57.7% and90.4% of the currently exact-but-misranked entities.

This illustration assumes no regressions elsewhere and hypothetical gains coming only from currently exact-but-misranked entities. It does not assume other error categories are incapable of improving: boundary errors, missing entities, label errors, and different composition decisions could also contribute. It illustrates the scale required; it is neither an oracle nor a reachable upper bound. Unordered entity F1 does not establish attainable slot accuracy.88% remains a research target and92% a stretch target, not predicted outcomes of P1.

## Required outputs

Preserve protocol/configuration, immutable-artifact manifests, exhaustive-verification results, prediction-level replay evidence, all primary and sensitivity predictions, per-document metrics, paired uncertainty, positional transition counts, and measured runtime/memory. The final report must answer separately: did Slot-MBR improve exact-slot, and what did it repair, retain, lose, or newly break?
