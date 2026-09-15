# bitrase-2.algo-2 — Posterior Concentration Audit

Completed 2026-09-15. Diagnostic only: 6,943 frozen dev documents, 85,336 slots,
temperatures 0.5, 1, 2. No new predictions, fitting, training, holdout or submission.

**Finding:** tau=2 broadens the frozen posterior substantially and changes more
decisions, but its predicted advantage for those changes disagrees with observed
correctness. The losses occur in every predefined K, populated length, gold-label
and MAP-margin stratum. This is evidence of a decision-relevant posterior mismatch;
it does not isolate a single causal defect or establish that fitting temperature
will fix it.

## Concentration and expected versus realized gain

Gains are relative to the identical saved MAP decisions (80.9354% exact-slot).
Entropy, normalized entropy, MAP structure probability and energy sacrifice below
are document means. Expected/realized gains are pooled per slot, in percentage
points. Natural logarithms; count includes all seven positive labels.

| tau | H (nats) | H/log count | MAP structure probability | Expected gain | Realized gain (95% CI) | Slot accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.8369 | 0.00813 | 0.75697 | +0.2838 pp | +0.2988 [0.1140, 0.4842] pp | 81.2342% |
| 1 | 2.3702 | 0.02312 | 0.53245 | +0.7902 pp | +0.3820 [0.0735, 0.6857] pp | 81.3174% |
| 2 | 13.7727 | 0.13785 | 0.10220 | +1.3832 pp | −3.8202 [−4.3418, −3.2921] pp | 77.1152% |

Mean log count is **98.0251**, identical at all temperatures on every document.
Entropy per slot (mean H/K) is 0.06470, 0.18415, 1.10361 respectively. Support did
not change. Entropy increasing with temperature is mathematically expected, not
independent proof of why performance changed. Even tau=2 remains far from uniform
over all valid structures; “broader” does not mean “uniform.”

The expected-minus-realized gain gap is −0.0150 pp at tau=0.5, +0.4082 pp at tau=1,
and **+5.2034 pp at tau=2**, with respective 95% intervals [−0.1954,0.1661],
[0.0959,0.7126], [4.6738,5.7509] pp. Per-document correlation between delta U and
delta C is 0.2196, 0.1153, **−0.1466**. These are descriptive associations.

Intervals use 2,000 paired document bootstrap replicates, seed 20260915, pooled
slot denominators. Small differences from P1's 10,000-replicate intervals are
expected. Repeated dev use and model selection bias are not corrected.

## Selected-slot reliability

| tau | Rule | Mean confidence | Actual accuracy | ECE (10 bins) | Brier |
|---|---|---:|---:|---:|---:|
| 0.5 | MAP | 91.2928% | 80.9354% | 0.10357 | 0.13340 |
| 0.5 | MBR | 91.5766% | 81.2342% | 0.10342 | 0.13402 |
| 1 | MAP | 81.5320% | 80.9354% | 0.01112 | 0.11659 |
| 1 | MBR | 82.3222% | 81.3174% | 0.01059 | 0.11740 |
| 2 | MAP | 53.2240% | 80.9354% | 0.27711 | 0.21188 |
| 2 | MBR | 54.6072% | 77.1152% | 0.22508 | 0.19429 |

Overall tau=2 confidence is too low on average. Nevertheless, on its **12,518
changed slots**, MAP confidence is 25.6606% against 55.3763% actual correctness;
MBR confidence rises to 35.0901% while actual correctness falls to **29.3338%**.
Thus blanket “underconfidence” misses the relative preference failure. Aggregate
ECE and Brier even improve from MAP to MBR at tau=2 while slot accuracy worsens:
the evaluated selected triples and their binary outcomes change with the rule.
Those metrics alone cannot choose the decoder.

At tau=1, changed-slot MBR confidence is 49.8390% against 43.7553% accuracy,
despite the near match in overall averages. Low pooled ECE does not establish
conditional calibration. Complete fixed confidence-bin tables and label/margin/
K/length/changed-slot reliability are in [summary.json](../../../output/bitrase-2/algo-2/summary.json).

## Churn and positional damage

| tau | Changed ordered slots | Removed/added triples (each) | Mean raw energy sacrifice | Improved/worsened docs |
|---|---:|---:|---:|---:|
| 0.5 | 1,791 | 457 | 0.02784 | 224 / 157 |
| 1 | 4,692 | 1,195 | 0.13766 | 569 / 450 |
| 2 | 12,518 | 2,804 | 0.68848 | 878 / 1,497 |

Every changed ordered slot also changes its boundary pair. Ordered-slot changes
exceed set replacements because retained spans can move to different slots.
For tau=2, 3,006 previously displaced and 666 previously absent gold entities
become correct: **3,672 repairs**. Conversely, 5,538 formerly correct gold entities
become displaced and 1,394 disappear: **6,932 new incorrect slots**, net **−3,260**.
Exact entities present at the wrong rank rise from 10,444 to 12,602. Exact entity
presence falls by 1,102 (79,511 to 78,409), while slot correctness falls by 3,260.
Both membership loss and positional displacement contribute.

Within the original MAP simple episodes, tau=2 repairs 2,721 entities, retains
5,835 displaced, and loses 437. There are 1,975 new/changed episode signatures;
this last count includes modified episodes and is not a count of new harmful
episodes. All nine disjoint transition counts and original episode diagnostics
are retained from P1 in the summary; episode repairs use a narrower denominator
than the 3,672 global repairs above.

## Fixed strata

Tau=2 realized gain is −2.2424 pp for K=1–10 (2,045 docs), −4.6316 pp for K=11–15
(3,964), and −3.1970 pp for K>=16 (934). Both populated length bins lose:
129–256 tokens −3.7526 pp (5,337 docs), >256 −4.0272 pp (1,606).
The predeclared <=128 bin has **zero documents** and cannot support a conclusion.
Expected gains remain positive in every populated K and length bin. These are
separate stratifications, not joint adjustment or causal controls.

| Gold label | Slots | tau=0.5 gain pp | tau=1 gain pp | tau=2 gain pp |
|---|---:|---:|---:|---:|
| ADDRESS | 8,226 | +0.2918 | +0.2188 | −5.2273 |
| DATE | 16,239 | +0.3079 | +0.2833 | −4.7663 |
| EMAIL | 9,723 | +0.2468 | +0.2571 | −4.1242 |
| JOB_TITLE | 8,197 | +0.2562 | +0.9638 | −1.2444 |
| NAME | 26,009 | +0.3037 | +0.3768 | −3.5565 |
| PHONE | 9,300 | +0.3333 | +0.3441 | −3.9247 |
| USERNAME | 7,642 | +0.3402 | +0.3664 | −3.4415 |

Tau=2 also loses across all raw MAP label-minus-NONE margin bins: <0 −1.4689 pp;
[0,2) −2.6072 pp; [2,5) −5.0219 pp; >=5 −3.3955 pp. This margin is a local
score, not a margin against the best competing complete structure. Fixed MAP
membership avoids reassigning these strata after MBR selection. Strata have no
separate inferential significance claims.

## Decision and next research

Close this diagnostic with the conclusion that **broadening this frozen posterior
to tau=2 increases harmful selection churn and misestimates the relative utility
of competing decisions**. The posterior supports more repairs too, but newly
introduced errors dominate. This answers the empirical mechanism more precisely
than merely saying entropy increased.

Keep P1's primary tau=1 result as the existing research result. No temperature
was selected here. P0's user-reported Kaggle 0.805 remains the external reference;
88%/92% are still unproven targets.

The next useful research question is how to separate **decision-relevant score
scale mismatch from incorrect relative energies**. A future proposal should use
train-only out-of-fold P0 scores (not scores on its own fitting documents), compare
paired competing decisions, and predeclare whether the endpoint is probability
fit or slot utility. Scalar scaling cannot change complete-structure MAP ranking,
although it can change MBR through redistribution of posterior mass. A fitted
calibration experiment and a scorer experiment therefore test different claims.
Do not conclude either is the proven bottleneck from this audit, and do not
reopen the stopped structure/GCN branch on this evidence.

Validation: four exhaustive/edge-case math tests; all 20,829 document-temperature
partition and utility comparisons; marginal normalization, entropy bounds and
monotonicity; input hashes before/after; historical P1 prediction hashes; unique
full document/slot coverage and independent correctness/transition totals.
See [protocol](protocol.md), [technicalities](technicalities.md), and
[validation](../../../output/bitrase-2/algo-2/independent_validation.json).
