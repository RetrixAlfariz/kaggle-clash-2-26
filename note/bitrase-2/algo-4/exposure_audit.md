# bitrase-2.algo-4 — Exposure preflight

Archived preflight snapshot (2026-09-15). The exposure audit below preserves the decision and evidence recorded before model execution. Training and EVAL2 evaluation have since completed; see the [final algo-4 result](report.md).

**Subsequent operational decision:** Vian requested autonomous improvements toward 82%; definition 2 was adopted explicitly before selection. EVAL2 is frozen at 3,000 documents, FIT2 at 52,582; runtime benchmark passed and training was underway when this snapshot was written. No performance result had been reported at that time.

## What was verified

The registry covers all 55,582 prepared-TRAIN documents and 55,581 discovered families. Memberships and source hashes are saved under `output/bitrase-2/algo-4/exposure/`. The inspected-source inventory is `searched_sources.json`; per-document evidence is `registry.parquet`; counts are `summary.json`. The executable reconstruction is `src/audit_algo4_exposure.py`.

| Historical event | TRAIN documents | Interpretation |
| --- | ---: | --- |
| Structure probe | 6,000 | Dedicated diagnostic evaluation; exclude |
| Algo-3 CAL | 1,000 | Calibration; exclude |
| Algo-3 EVAL | 3,000 | Evaluation and research selection; exclude |
| Union of those events | 9,523 | Overlap accounted for |
| Initial baseline-probe validation | 0 | Its validation IDs are in prepared DEV |
| Historically reviewed examples | 16 | Purpose not fully classified; conservatively exclude |
| P0 training-accuracy aggregate reused in research | 55,582 | In-sample training diagnostic, requiring the distinction below |

The 16 reviewed examples overlap the excluded union by two documents. After excluding the union and unresolved examples, and enforcing eligibility on every family member, **46,045 documents in 46,044 families remain** under a policy that permits historical in-sample training diagnostics. The only non-singleton family is retained whole. These discovered groups do not establish unseen-template generalization.

## The classification issue

`src/run_span_probe.py` accumulates accuracy during optimization, separately for candidate strata. P0 epoch 2 logged 14,228,992 random-negative predictions: exactly 256 draws for each of all 55,582 training documents, reconciled against `sampling_documents.parquet`. Its random-negative accuracy is 99.9936%; old-model false-positive accuracy is 48.2430%.

`note/bitrase-2/p0/research_questions.md` subsequently uses that comparison to motivate research into context, loss emphasis, and calibration. These are **online, in-sample training metrics**, not an independent evaluation of the final checkpoint. Nevertheless, they informed later research discussion.

Two definitions therefore produce different eligibility counts:

1. If any research reuse of a training diagnostic counts as disqualifying method-selection exposure, **zero** documents qualify. The >=2,000 gate fails under that definition.
2. If historical in-sample training diagnostics are permitted alongside historical supervised fitting, while dedicated evaluation/calibration/probe exposure remains excluded, **46,045** qualify. A 3,000-document EVAL2 is feasible.

The zero count is a **policy-dependent interpretation**, not evidence that every training document received held-out evaluation. The agreed permission for prior supervised fitting does not explicitly settle post-hoc reuse of training metrics. The registry preserves both classifications; no alternative policy has been silently applied.

## Recommendation and next action

Recommend definition 2, with an explicit caveat: the experiment uses documents with less prior evaluation exposure, but its research choices were informed by historical training diagnostics involving those documents. A new encoder trained only on FIT2 prevents direct fitting leakage into EVAL2; it cannot undo historical research adaptation.

The initial recommendation was to settle the exposure definition before selection. This has now been resolved using definition 2 under the subsequent autonomous-improvement request. The [protocol](protocol.md), assignments and runtime benchmark are recorded before training. No known dedicated evaluation sets were used to fill EVAL2. The original preflight registry remains unchanged as historical evidence.

No new claim about ranking-loss efficacy, encoder limitations, or Kaggle performance follows from this preflight. No dev, holdout, or test contents were accessed. Old artifacts were checked by hash and remain unchanged.
