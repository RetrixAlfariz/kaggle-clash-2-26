# P0 frozen direct-span probe — results

Discovery seed2026 selected epoch2 at **80.9354% exact-slot dev**, versus BIO epoch4 **79.4225%** and the previous ensemble **80.4057%**. This is research-positive and a promotion candidate. It does not reach88% or92%. Encoder weights and baseline sources remained frozen; no pretrained/external data or reserved holdout was used.

## Discovery checkpoints

| Head epoch | Exact-slot | Entity F1 | Correct slots |
|---|---:|---:|---:|
| 1 | 80.1268% | 92.9561% | 68,377 |
| **2** | **80.9354%** | **93.1740%** | **69,067** |
| 3 | 80.7819% | 93.0217% | 68,936 |

All6,943 dev documents and85,336 gold entities were retained. All three saved prediction files passed independent metric/offset/count/non-overlap validation. Dev representability:85,293 entities inside width16,41 unaligned,2 too long; excluded-from-universe gold still counts against the model.

## What improved

| Diagnostic | BIO epoch4 | Previous ensemble | P0 epoch2 |
|---|---:|---:|---:|
| Correct slot | 67,776 | 68,615 | **69,067** |
| Exact entity, unordered | 78,132 | 78,375 | **79,511** |
| Same-label overlap error | 4,083 | 3,940 | **2,760** |
| Exact span, wrong slot | 10,356 | 9,760 | **10,444** |
| No overlapping prediction | 2,678 | 2,583 | 2,593 |
| Same boundary, wrong label | 134 | 104 | 390 |
| Overlap, wrong label | 309 | 334 | 82 |

Boundary errors fell32.40% versus the same encoder's BIO head, providing evidence consistent with the span hypothesis. Extra exact entities did not all become correct slots: compared with the ensemble, there are1,136 more exact entities but only452 more correct slots, while wrong-rank exact entities increased684. The label-error categories also changed, so boundary F1 must not substitute for exact entity/slot metrics.

## Paired uncertainty, selected discovery checkpoint

10,000 paired document bootstrap replicates:

| Comparison | Observed gain (pp) | 95% percentile interval (pp) |
|---|---:|---:|
| P0 vs BIO4 | **+1.5128** | **[+1.0025,+2.0191]** |
| P0 vs previous ensemble | **+0.5297** | **[+0.0129,+1.0568]** |

The improvement against BIO is clearer than against the ensemble; the latter interval barely excludes zero. These intervals condition on the selected checkpoint and this dev corpus; they do not correct repeated development selection or template dependencies and are not Kaggle evidence.

## Data, training and provenance

684,811 representable train positives,4,849,915 unique forced negatives,256 random negatives/document/epoch. Training candidate universe203,048,013 after protected-overlap exclusions; raw universe203,082,848. Random-stratum weight median13.8086,p95=17.0352,max25.8867. Negative strata can overlap in diagnostic tags but candidates are trained once.

Native FP16 cached states reproduced all baseline dev predictions and native logits exactly (maxabs0). Cache train extraction371.1s,dev55.6s. Successful discovery planning/training/evaluation/bootstrap212.5s excluding initial cache verification. Head fitting itself took28.7/24.7/24.2s; full dev scoring about21s per epoch. No AMP overflow retries were needed in the successful discovery run.

Initial runtime failures and repairs are preserved/documented in [technicalities.md](technicalities.md). No failed attempt produced a selected checkpoint or a dev score. Source/checkpoint/cache/data hashes are captured in manifests/configs; original baseline files were not edited.

## Fixed-budget replications

Both replications used the selected fixed2epochs, evaluating only the final checkpoint. No new tuning and no seed ensemble.

| Seed | Exact-slot | Entity F1 | Gain vs BIO4 (pp) | Gain vs old ensemble (pp) |
|---|---:|---:|---:|---:|
| 2026 discovery | **80.9354%** | **93.1740%** | +1.5128 | +0.5297 |
| 3407 | 80.3412% | 92.9877% | +0.9187 | -0.0645 |
| 1337 | 80.8416% | 93.0463% | +1.4191 | +0.4359 |

Mean exact-slot80.7061%, median80.8416%, range80.3412–80.9354%. All three runs exceeded the same-encoder BIO baseline by at least.5pp; all three paired95% intervals versus BIO excluded zero. Against the old ensemble, only2/3 point estimates improved, and both replication intervals included zero. Thus the head-system improvement over BIO is consistent, while superiority to the ensemble is not established across seeds.

Replication paired intervals (pp): seed3407 vsBIO [+0.3946,+1.4618], vsensemble [-0.5942,+0.4753]; seed1337 vsBIO [+0.8935,+1.9562], vsensemble [-0.1010,+0.9761]. Both replications passed independent prediction validation. There were5 dev evaluations in total:3 discovery checkpoints and2 replication finals. Concurrent replication/inference timings are not clean standalone speed benchmarks.

## Selected manual submission

The discovery seed2026/epoch2 remains the preselected candidate; no best-replication selection was added. Inference recomputed encoder states from raw dev and reproduced every selected dev prediction exactly before processing test. CSV: `output/bitrase-2/p0/seed2026/submission.csv`,285,318 rows across23,156 documents. SHA256:`587dd76b8cfc3d8b06b5c24cdea56b469a6e8affbb113300638612dccf8e0fc4`. Submission manifest binds encoder/head/vocab/test/slots/predictor hashes. No Kaggle upload occurred.

The independent submission validator is recorded at `output/bitrase-2/p0/seed2026/submission_validation.json`. This validates formatting/counts/bounds/order/labels/non-overlap, not test accuracy.

Vian subsequently reported a Kaggle score of **0.805** for bitrase-2/P0 after manual submission. This is user-reported, not independently retrieved from Kaggle. Compared with the previously reported bitrase-1 score of0.705, the increase is **0.100 (10.0 percentage points)**. The selected dev score is80.9354%, about0.44 percentage points above the reported Kaggle score; this proximity alone does not establish split equivalence. The88%/92% targets remain unmet. The no-upload statement and manifest above describe the original local generation run, before Vian's manual submission.

## Decision and next research

P0 is research-positive: direct span scoring improves this frozen representation under the tested objective, with substantial boundary-error reduction. The discovery checkpoint is eligible for manual submission; preserve the prior ensemble as a comparator because the gain over it is modest and seed-sensitive.

Do not spend the next experiment on another boundary architecture automatically. The most specific unresolved question is whether span selection aligned to expected slot correctness can convert more of the exact-entity improvement into correct slots. [research_questions.md](research_questions.md) gives Aestem a bounded request for interval MAP vs slot-aware MBR on fixed model scores, with score-calibration assumptions made explicit. Joint training remains a later question, not a conclusion from P0.
