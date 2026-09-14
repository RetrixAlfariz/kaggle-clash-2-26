# P0 specification

This is a frozen-encoder span-system experiment, not yet a promoted bitrase-2 model. The encoder checkpoint and preparation v1 are unchanged. A new eight-class span head receives the native contextual hidden states produced immediately before the epoch4 BIO classifier.

Canonical implementation decisions are frozen in [protocol.md](protocol.md). [aestem_proposal.md](aestem_proposal.md) preserves the supplied proposal; the protocol applies Opheline's reviewed corrections, including loss denominator equivalence, native cache precision, external hidden-state extraction, and concrete batching/ties.

Primary baseline: epoch4 BIO79.4225% exact-slot. Deployment candidate comparison: epoch4/8 neural ensemble80.4057%. No score increase is assumed. Train includes all representable positives; every dev gold remains in the metric denominator. No reserved holdout or test data enters head training/selection.

Candidate universe covers all token spans1..16. Input features1056dim, trainable head roughly275k parameters. At inference all candidates are scored, label logit minus NONE supplies interval utility, then exactlyK non-overlapping intervals are selected. Token and character intervals use half-open conventions in code.

Research-positive numeric threshold: at least+.5pp against epoch4 BIO, with compatible error evidence. Only then repeat the selected epoch count with two fixed seeds. Promotion additionally requires champion improvement and complete output validation. Kaggle upload remains Vian's action.
