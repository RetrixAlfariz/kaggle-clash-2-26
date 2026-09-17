# bitrase-2.algo-3 — Train-only Scale and Ranking Diagnostic

Frozen before training or CAL/EVAL results, 2026-09-15.

Single three-way split of prepared TRAIN only, not OOF cross-validation. Use the
existing text-only structure_families grouping (normalized exact/digit-masked
duplicates and verified minhash-retrieved 5-word Jaccard >=0.8). Grouping is
approximate; singleton groups do not prove unseen-template generalization.
Sort complete families by SHA256("20260917:"+group_id). Allocate first >=1000
documents to CAL, next >=3000 to EVAL, all remaining to FIT, keeping families whole.
Freeze assignments before training; no balance-based retries or label selection.

Build vocabulary on FIT only. Train a new randomly initialized existing ByteCNN
BiLSTM BIO encoder for exactly four epochs; freeze it, then train the existing
P0 span head for exactly two epochs. Seed 2026, AdamW lr .001, weight decay .01,
clip 1, batch 32, original AMP and P0 inverse-probability sampling. Old-model
false positives for FIT sampling come only from this new FIT-trained BIO model.
No original P0 checkpoints, caches or predictions. No checkpoint selection,
early stopping, architecture changes, dev/holdout/test access or pretrained data.
Historical architecture/epoch choices are inherited research choices, so this is
an internal diagnostic, not a pristine external generalization estimate.

Persist immutable CAL/EVAL logits over the unchanged <=16-token, seven-label
candidate universe. Complete-gold support must be checked; unsupported structures
have zero posterior probability and are reported separately, never silently
assigned finite NLL or counted as ranking failures. Evaluate exact-slot decisions
on all EVAL documents, including support failures.

Fit one scalar inverse temperature beta in [0.25,4] (tau in [0.25,4]) on CAL only.
Objective: pooled structured negative log likelihood per gold slot over fully
supported CAL documents, logZ(beta)-beta*S(gold). Use convex derivative
E_beta[S]-S(gold), endpoints plus 18 deterministic bisection steps. This is one
continuous fit, not selection among temperatures on EVAL. Record fit and hash
before opening EVAL scores/labels for the diagnostic. Never expand bounds after
observing results; boundary optimum is reported as a limitation.

EVAL comparison is tau=1 versus the single fitted tau. Endpoints:
1. Structured NLL on fully supported gold documents, baseline and fitted.
2. Gold energy versus strongest feasible incorrect complete structure: strictly
   wins, ties, loses, plus gap. Scalar scaling cannot change this raw ranking.
3. Exact-slot MBR accuracy/gain and MAP reference on ALL EVAL documents. MAP must
   replay identically under scalar scaling. Report predicted versus realized gain.
4. Supporting span diagnostic: correct span/label versus strongest distinct
   one-slot replacement feasible between its gold neighbors, holding other gold
   slots fixed. This is gold-conditioned and does not measure free decoding.

Minimum useful effects, predeclared: relative structured NLL improvement >=1%,
with paired-bootstrap lower confidence limit >0; decision gain >=0.2 percentage
points with lower confidence limit >0. Use 2000 paired-document bootstrap draws,
seed 20260917; no iid slot intervals. Evaluate both gates separately. A probability
gate alone cannot promote calibration for decoding. If neither passes, stop this
scalar-calibration branch for this run. If only NLL passes, report probability
fit improvement without decision improvement. Ranking losses motivate testing
scoring hypotheses but do not establish head or joint training as the solution.

No forced scale-versus-ranking winner. Both defects can coexist. No universal
ranking-error threshold, next model choice, temperature tuning on EVAL, Kaggle
submission, or 88%/92% reachability claim follows from this one seed/split.
Per-label results are descriptive. Bootstrap does not capture retraining or split
uncertainty. Verify tiny exact partitions/gradients, ranking competitors and fit
behavior before execution, and verify hashes/disjointness and replay afterward.
