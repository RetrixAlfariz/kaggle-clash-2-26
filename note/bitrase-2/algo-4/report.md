# bitrase-2.algo-4 — Frozen-Encoder Global Ranking Objective

**Result: negative.** On the frozen 3,000-document EVAL2, the ranking-loss challenger fell far below the matched-CE control on both MAP and Slot-MBR exact-slot accuracy. Neither predeclared performance gate passed. This rejects the tested training package on this split; it does not establish that the encoder is the bottleneck or that ranking objectives cannot help.

## Experiment and verification

The experiment followed the [frozen protocol](protocol.md). A fresh encoder was trained on FIT2 only; two heads shared an identical initialization and matched CE candidate, weight, order, and RNG exposure. The challenger added the frozen-lambda, epoch-wise mined global-ranking loss. No pretrained weights, dev, holdout, test, or Kaggle submission were used.

EVAL2 contains 3,000 whole-family documents and 36,794 gold slots. FIT2 contains 52,582 documents; 52,423 had representable positive-K gold structures for ranking. Nine EVAL2 documents lacked fully supported gold structures for ranking diagnostics. Exact-slot and entity metrics still score all EVAL2 slots. The bootstrap sampled the 3,000 family groups with replacement (all singleton families), using 2,000 replicates and seed `20260919`.

The final [verification record](../../../output/bitrase-2/algo-4/run/verification.json) reports `training_verified: true` and `evaluation_predictions_reconciled: true`. It also confirms matched CE exposure, lambda reconciliation, feasible incorrect competitors, and unchanged source hashes. The immutable final-score evaluation is in [eval_summary.json](../../../output/bitrase-2/algo-4/run/EVAL/evaluation/eval_summary.json); document-level predictions and diagnostics are in [eval_documents.json](../../../output/bitrase-2/algo-4/run/EVAL/evaluation/eval_documents.json).

| Decoder / metric | Matched-CE control | Ranking challenger | Paired gain (challenger − control) | 95% family-bootstrap CI | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| MAP exact-slot accuracy | 80.9887% | 26.7245% | −54.2643 pp | [−55.4022, −53.0577] pp | Fail |
| Slot-MBR τ=1 exact-slot accuracy | 81.1790% | 26.6919% | −54.4871 pp | [−55.6174, −53.3200] pp | Fail |

The predeclared gate required a gain of at least **+0.20 percentage points** and a 95% interval lower bound above zero, separately for MAP and MBR. Both comparisons instead show large, precisely negative paired differences. Entity F1 follows the same pattern: MAP control 93.3141% versus challenger 39.4059%; MBR control 92.9228% versus challenger 39.4765%.

## Ranking evidence and interpretation

For the 2,991 supported EVAL2 documents, the complete gold structure was MAP for 1,625 control documents and 47 challenger documents. The mean gold-minus-strongest-wrong energy gap was −0.604 for control and −67.441 for challenger. At the local one-slot level, gold beat the strongest feasible alternative on 34,839 of 36,685 supported control slots, compared with 16,150 of 36,685 challenger slots. These diagnostics show that the challenger scores ranked the evaluated structures much worse; they do not identify why the optimization diverged from the intended behavior.

The fixed initialization-scale factor was `λ = 1.298212`. Its initial gradient measurement approximately balanced the CE and ranking gradient norms. That balance did not persist: at the start of head epoch 2, median CE and ranking gradient norms were 0.003308 and 0.462797, so `λ × rank_norm / CE_norm ≈ 181.61`. After epoch 2 they were 0.045359 and 0.337770, giving a ratio of about 9.67. This is evidence that the initial gradient-scale estimate was not representative later in training. It is **not** a causal isolation: the experiment changed the combined package of ranking loss, epoch-wise mining, fixed lambda, and its interaction with optimization. The failed result alone cannot establish that gradient-scale drift caused the accuracy collapse.

The runtime gate passed before training. The 256-document synthetic-hidden benchmark projected about 815 seconds for one FIT mining pass; this measured execution cost, not predictive quality.

## Decision and limits

Do not promote this algo-4 challenger. Keep the matched-CE control result as an internal reference, with its exact evaluation artifact and split provenance. The experiment does not demonstrate an improvement toward the internal 82% target, does not change the user-reported Kaggle 0.805 reference, and gives no evidence that 88% or 92% Kaggle performance is reachable or unreachable.

The exposure classification and historical research-adaptation caveat are preserved separately in [exposure_audit.md](exposure_audit.md). Because EVAL2 outcomes have now been inspected, EVAL2 is adaptive development data for subsequent research, not an untouched confirmatory holdout. Any follow-up must be a separately scoped experiment; do not fold its result into this algo-4 comparison or treat EVAL2 as fresh validation again.
