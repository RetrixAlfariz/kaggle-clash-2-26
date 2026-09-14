# Focused next research request for Aestem

P0 discovery is a frozen epoch4 encoder plus span head, trained with IPW categorical CE, all token spans1..16 at inference, score=z(label)-z(NONE), exact-K interval MAP. Selected head epoch2:80.9354% slot,93.1740% exact entity F1. Same-encoder BIO:79.4225%/91.5581%. Previous ensemble:80.4057%/91.8428%.

Boundary error reduction is substantial: same-label overlap4083->2760 versus BIO. However exact entities at wrong rank remain10444, compared with10356 for BIO and9760 for the ensemble. Versus ensemble,1136 more exact entities translate into only452 more correct slots. This motivates selection/objective research before adding another boundary architecture. It does not prove that decoding alone is sufficient.

## Primary question: can the same span scores optimize positional utility better?

Design one bounded comparison of interval MAP against a slot-aware minimum Bayes risk decoder over the SAME complete span universe and SAME frozen model scores. Do not fit new encoder/head weights for the first comparison.

Specify a mathematically consistent distribution over exactlyK non-overlapping labelled spans, forward/backward recurrences in log space, and marginal probability of a labelled span occupying slotj. Explain how label uncertainty must enter the partition: MAP's max-label reduction cannot simply be assumed valid for marginalization. Derive the decision rule maximizing expected correct slots under that model distribution, including how it differs from maximizing entity marginals.

Address score temperature/calibration and correlated overlapping spans. IPW CE plus logit differences does not establish calibrated structured probabilities. Propose a fixed or train-internal calibration rule, no unbounded dev sweep. State complexity/memory for T<=433 train, W16,K<=31. Require exact-K/non-overlap and exhaustive toy verification, preserve all candidates initially, and no gold information in decoding.

Report a falsifiable comparison protocol with exact-slot primary, entity F1/error decomposition secondary. Do not predict88/92 from93.17% unordered F1. Distinguish a theoretical optimum under an assumed posterior from empirical improvement on this dataset.

## Secondary question, only if primary evidence points to scoring

The selected P0 head's train diagnostics show roughly48.2% NONE classification accuracy on old-model false-positive spans, versus99.99% on sampled random negatives. Old FPs are selected on train, not out-of-fold, and this is in-sample evidence, so it cannot establish generalization failure. Assess whether a bounded reweighting/calibration diagnostic can distinguish insufficient context from loss emphasis. Keep candidate universe/decoder/encoder fixed; do not bundle this with BOPN, slot queries, or joint training.

Requested deliverable: one primary protocol, assumptions, recurrences/pseudocode, numerical tests, compute estimate, decision gate, and primary-source references. Research only; no code/training/holdout inspection/Kaggle submission. Explain when the evidence would instead justify joint span training.
