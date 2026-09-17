# bitrase-2.algo-3 — Train-only Scale and Ranking Diagnostic

Completed 2026-09-15. Prepared for Vian and Aestem.

**Decision: do not promote the NLL-fitted global temperature for decoding.**
It improves out-of-fit structured probability fit, but significantly worsens
exact-slot MBR on the same evaluation documents. Both scale mismatch and incorrect
relative energies are present; the experiment does not identify one exclusive
cause or establish joint training as the solution.

## What was tested

One deterministic three-way split of prepared TRAIN: **51,582 FIT / 1,000 CAL /
3,000 EVAL**. This is a single split, not OOF cross-validation. Neither CAL nor EVAL
contributed supervised examples to the new encoder or span head. Vocabulary was
built on FIT only. Both models were initialized from scratch: existing ByteCNN
BiLSTM BIO encoder, four fixed epochs, followed by a frozen encoder and the P0
direct-span head, two fixed epochs. No checkpoint selection or early stopping.

FIT forced negatives used predictions from this fresh FIT-trained BIO model.
Original P0/P1 checkpoints, scores and caches were not used. The candidate universe
remained all <=16-token spans with seven positive labels. No structure features,
GCN, external data, pretrained weights, dev/holdout/test access or submission.

Detected families remained whole. The existing text-only duplicate/template
search found 55,581 families among 55,582 documents: one two-document family,
which landed in FIT, and otherwise singletons. CAL and EVAL therefore have 1,000
and 3,000 singleton groups. Approximate retrieval checked 341,543 candidate edges
and verified no additional >=0.8 Jaccard links. **This does not prove broad
unseen-template isolation.**

The frozen [protocol](protocol.md) set two independent useful-effect gates:
- Probability fit: >=1% relative structured NLL reduction and positive lower 95%
  paired-bootstrap limit.
- Decision utility: >=0.2 percentage-point exact-slot MBR gain and positive lower
  95% paired-bootstrap limit.

The one scalar beta was fitted on CAL by minimizing complete-structure NLL,
using bounds [0.25,4] and 18 fixed bisection steps. The fitted value was saved
before EVAL was opened for diagnostic analysis. Inference and packaging of EVAL
metadata happened beforehand, after neural training; EVAL labels were not used
to generate scores or fit parameters.

## Results: the two endpoints disagree

The CAL optimum was interior: **beta=0.8407225608825684**, equivalently
**tau=1.1894530330555497**. CAL NLL per slot fell from 0.29117259 to 0.28268320,
a 2.9156% relative improvement. No further scalar was tried on EVAL.

| EVAL endpoint | Baseline tau=1 | CAL-fitted tau | Change / 95% CI | Gate |
|---|---:|---:|---|---|
| Structured NLL per supported slot | 0.28614863 | 0.28077396 | 1.8783% relative reduction [1.0799%, 2.6273%] | PASS |
| Exact-slot MBR, all slots | 80.2589% | 79.7921% | **−0.4669 pp [−0.7336, −0.1993]** | FAIL; observed harm |
| Exact-slot MAP, all slots | 79.8382% | 79.8382% | Predictions identical | Invariant |

NLL used **2,992 fully supported EVAL documents / 36,741 slots**. Eight EVAL
documents containing 101 gold slots lacked complete-gold support and were excluded
only from finite structured NLL and complete-gold ranking statistics. Their true
complete-gold NLL under this support is infinite. All **3,000 EVAL documents /
36,842 slots** remained in the decision endpoint. CAL fitting likewise used 998
fully supported documents / 12,292 slots, with two unsupported documents reported
separately rather than assigned finite loss.

The bootstrap used 2,000 paired document draws, seed 20260917. NLL ratios pool
document NLL sums; decision ratios pool correct counts over K totals. Intervals
condition on this split, neural model and fitted scalar; they do not include
temperature refitting, neural retraining or split uncertainty. Probability fit
improvement is evidence from a proper scoring rule, not proof that the entire
posterior is calibrated.

The replica's scores must not be compared directly with original P0/P1 dev scores
as a model regression or gain: these are different evaluation documents and a
different FIT-trained checkpoint. The within-replica comparison above is the
controlled experiment.

## How decoding became worse

Baseline MBR has 29,569 correct slots; fitted-temperature MBR has 29,397:
**172 fewer correct slots**. Of the 698 ordered slots that change, 204 are repaired
and 376 previously correct slots become incorrect. The remaining 118 changes
remain incorrect. At document level, 79 improve, 111 worsen and 2,810 retain the
same number of correct slots. Equal correct count does not imply equal predictions.

Expected and realized gains below each compare MBR with the fixed MAP decision,
using that row's own posterior for expected utility:

| Posterior | Expected MBR−MAP gain | Realized MBR−MAP gain |
|---|---:|---:|
| tau=1 | +0.6313 pp | +0.4207 pp |
| CAL-fitted tau | +0.8052 pp | −0.0461 pp |

These expected values are not a cross-temperature prediction of fitted MBR's
advantage over baseline MBR. Their probability distributions differ. The primary
decision comparison remains the paired **−0.4669 pp** result above.

Thus a statistically supported NLL improvement does not imply improved slot
utility. This repeats the earlier audit's warning under a fresh out-of-fit
experiment, without selecting the temperature on EVAL.

## What the raw ranking reveals

For every fully supported EVAL document, an exact top-two labeled-structure DP
identified the strongest incorrect feasible complete structure. Alternative labels
at unchanged boundaries are included. Gold relative to that rival:

| Result | Documents |
|---|---:|
| Gold strictly wins | 1,525 |
| Energy tie | 2 |
| Gold loses | **1,465 (48.9639%)** |
| Total fully supported | 2,992 |

Positive scalar scaling cannot reverse any of these strict energy orderings.
Exact prediction-level MAP replay passed on all 3,000 documents, including those
with support failures. Tie handling remains a separate issue from strict losses.

This does not mean 48.96% of all spans are poorly ranked. A single wrong decision
can make a complete structure incorrect. The supporting local diagnostic holds
all other gold slots fixed and compares each gold span/label with the strongest
distinct feasible replacement between its gold neighbors. It permits both boundary
and label changes. Among 36,741 such comparisons, **34,662 strictly win (94.3415%),
2 tie, and 2,077 lose (5.6531%)**.

The high local win rate uses gold context and cannot be treated as a free-decoding
accuracy, reachable oracle, or upper bound for the 88%/92% targets. Multiple local
errors and globally coordinated alternatives make complete-structure ranking a
different question.

## Results by label

Decision columns include all EVAL gold slots; local-ranking columns include only
fully supported documents. These are descriptive, without label-wise significance
claims or label selection.

| Gold label | All slots | Baseline MBR | Fitted MBR | Gain pp | Local strict losses / comparisons |
|---|---:|---:|---:|---:|---:|
| NAME | 11,244 | 77.5436% | 77.0277% | −0.5158 | 1,198 / 11,219 |
| DATE | 7,103 | 81.7683% | 81.2474% | −0.5209 | 247 / 7,086 |
| EMAIL | 4,144 | 84.6042% | 84.0251% | −0.5792 | 24 / 4,134 |
| PHONE | 4,019 | 83.7522% | 83.2545% | −0.4976 | 41 / 4,009 |
| ADDRESS | 3,464 | 77.9734% | 77.5404% | −0.4330 | 156 / 3,453 |
| USERNAME | 3,329 | 74.9775% | 74.5269% | −0.4506 | 267 / 3,319 |
| JOB_TITLE | 3,539 | 84.0068% | 83.9220% | −0.0848 | 144 / 3,521 |

Every label has a negative point change under the fitted scalar. The local loss
counts identify where hard comparisons exist, but do not establish whether their
cause is representation, head loss, annotation ambiguity or missing interactions.

## Recommendation and questions for Aestem

**Do not promote this NLL-fitted global scalar to the production/research decoder.**
Retain the existing original P1 tau=1 research result. Keep the fitted scalar as
diagnostic evidence that probability fit and decision utility can move in opposite
directions. This is not a proof that every calibration family or every fitting
objective must fail.

The next justified research direction is a bounded test of relative scoring,
with an explicit competitor definition and unchanged candidate support. It should
not jump directly to joint training, a new encoder, or a GCN. The present evidence
shows ranking losses that scale cannot reverse; it does not show which scorer
change would repair them or whether its gains would survive new errors.

Questions for Aestem before a further implementation:
1. Can a frozen-encoder head experiment compare the existing CE objective with a
   small ranking-aware objective against **feasible competing spans/structures**,
   while keeping sampling exposure and candidate support controlled?
2. How should competitor mining remain inside FIT, and what train-only evaluation
   arrangement should be used now that this EVAL has been inspected? Avoid using
   the current EVAL to choose a new loss, margin, negative mixture or temperature.
3. What endpoint and minimum useful effect would justify a scorer change in
   **exact-slot accuracy**, including regressions, rather than only local ranking
   accuracy or structured NLL? A local win-rate increase alone is insufficient.

An alternative utility-targeted calibration proposal would test a different
objective and require its own protocol; it is not a rescue temperature sweep on
this EVAL. No follow-on method is implemented by this report.

P0's user-reported **0.805 Kaggle** remains the external reference. This experiment
does not change the submission, establish 88%/92% reachability, or create a new
promoted model version.

## Verification and evidence

Nine diagnostic tests passed, including exhaustive top-two labeled worlds across
90 tiny cases with ties, exact partition/gradient checks, analytic scalar optima,
Windows multiprocessing, ordered-slot correctness and heterogeneous-K bootstrap
weighting. Twelve existing grouping/span/neural utility tests also passed.

An independent verifier reconstructed every family assignment and checked source,
TRAIN, vocabulary, checkpoint and cache hashes. A second verifier recomputed all
EVAL ordered-slot counts, per-label metrics, MAP/gold energies and pooled NLL;
20 fixed-identity supported documents also matched the separate forward/backward
NLL implementation. All checks passed. MAP replay is identical under scaling.

- [Frozen protocol](protocol.md)
- [Technicalities and commands](technicalities.md)
- [CAL fit and hashes](../../../output/bitrase-2/algo-3/audit/calibration.json)
- [EVAL summary and bootstrap](../../../output/bitrase-2/algo-3/audit/eval_summary.json)
- [Document predictions and local ranking details](../../../output/bitrase-2/algo-3/audit/eval_documents.json)
- [Independent training verification](../../../output/bitrase-2/algo-3/independent_training_validation.json)
- [Independent audit verification](../../../output/bitrase-2/algo-3/independent_audit_validation.json)
