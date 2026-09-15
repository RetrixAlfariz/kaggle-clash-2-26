# bitrase-2 research

First completed experiment: [P0 frozen direct-span probe](p0/report.md).

Selected discovery checkpoint: seed2026, head epoch2, **80.9354% exact-slot dev**,93.1740% exact entity F1. Three-seed exact-slot range80.3412–80.9354%; consistent gain over the same encoder's BIO head, smaller and seed-sensitive gain over the old ensemble. No pretrained, external data, holdout tuning, or automatic Kaggle submission.

- [Specification](p0/spec.md)
- [Frozen protocol](p0/protocol.md)
- [Technicalities and commands](p0/technicalities.md)
- [Next research questions for Aestem](p0/research_questions.md)
- [bitrase-2.algo-1: completed Slot-MBR results](algo-1/report.md) — primary dev81.3174% (+0.3820pp); research-positive with strong temperature sensitivity. No new Kaggle score.
- [bitrase-2.algo-1: frozen protocol](algo-1/protocol.md)
- [Post-P1 research questions for Aestem](algo-1/research_questions.md)
- [Train-only document-structure diagnostic](structure-diagnostic/report.md) — separate scorer hypothesis; no useful hard-pair ordering gain, so the predefined feature branch is stopped. No production change.

Manual CSV: `output/bitrase-2/p0/seed2026/submission.csv`. Vian subsequently reported **0.805 on Kaggle**, versus the previously reported bitrase-1 score of0.705 (+10.0 percentage points). This score is user-reported;88%/92% remain unmet.

## Research status

The main path is BIO (79.4225% dev) -> frozen direct-span P0 (80.9354%) -> primary Slot-MBR P1 (81.3174%). These experiments support the tested span-scoring system and selection objective under their stated conditions; they do not establish universal architectural superiority or a new Kaggle score.

The document-structure diagnostic is a closed side branch, not another model stage. Its predefined features provided a tiny residual likelihood/discrimination signal but insufficient hard-candidate ordering value. Broader template isolation was not established. The [disfluency paper](../related-methodology/README.md) is retained as related methodology and provenance for that tested question, not a reason to continue structural modeling.

Completed: [bitrase-2.algo-2: Posterior Concentration Audit](algo-2/report.md). At tau=2, expected gain is +1.3832pp while realized gain is −3.8202pp; harmful positional churn dominates repairs across all predefined populated strata. Frozen artifacts and numerical checks pass. This is a diagnostic result, not a new model or fitted calibration. Next research should distinguish decision-relevant score scale mismatch from incorrect relative energies using a separate train-only protocol; no follow-on fitting is implemented.
