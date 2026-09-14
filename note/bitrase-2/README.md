# bitrase-2 research

First completed experiment: [P0 frozen direct-span probe](p0/report.md).

Selected discovery checkpoint: seed2026, head epoch2, **80.9354% exact-slot dev**,93.1740% exact entity F1. Three-seed exact-slot range80.3412–80.9354%; consistent gain over the same encoder's BIO head, smaller and seed-sensitive gain over the old ensemble. No pretrained, external data, holdout tuning, or automatic Kaggle submission.

- [Specification](p0/spec.md)
- [Frozen protocol](p0/protocol.md)
- [Technicalities and commands](p0/technicalities.md)
- [Next research questions for Aestem](p0/research_questions.md)

Manual CSV: `output/bitrase-2/p0/seed2026/submission.csv`. Vian subsequently reported **0.805 on Kaggle**, versus the previously reported bitrase-1 score of0.705 (+10.0 percentage points). This score is user-reported;88%/92% remain unmet.
