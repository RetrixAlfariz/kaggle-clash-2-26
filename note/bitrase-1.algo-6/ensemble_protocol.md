# Fixed checkpoint ensemble

After epoch4 achieved 79.4225% exact-slot, test an equal-weight average of epoch4 and epoch8 token logits, followed by the identical legal BIO exact-K decoder. Weights are fixed at 0.5/0.5 before the epoch8 result is inspected; there is no weight search or fitting on dev. Vocabulary and both checkpoints must match parent provenance hashes. Evaluate all 6,943 dev documents with all gold entities retained.

This is a separate development experiment, not an independent validation set. Compare against both single checkpoints; retain the ensemble only if exact-slot improves. Text-only honorific adjustment on epoch4 has already regressed to 78.8811%, so it is rejected for this checkpoint.

Implementation: `src/evaluate_neural_ensemble.py`. Output: `output/bitrase-1.algo-6/ensemble_4_8`. Parent training sources/protocol remain unchanged.
