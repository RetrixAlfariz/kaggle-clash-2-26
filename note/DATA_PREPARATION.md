# Prepared data — v1

> Update: v1 remains unchanged. The builder has since been hardened and now requires an explicit `--output`; use the pinned commands and guarded loader in [PREPARATION_HARDENING.md](PREPARATION_HARDENING.md). The original creation/validation history below describes v1's historical build.

The document-level dataset is ready at [`output/prepared/v1`](../output/prepared/v1). Raw files under `Data/` are unchanged. Preparation preserves original text and annotations, including disputed labels and unusual boundaries. No model was trained during this step.

## Frozen supervised split

| Partition | Documents | Gold entities | Intended use |
| --- | ---: | ---: | --- |
| Train | 55,582 | 685,026 | Fit model weights, dictionaries, and learned preprocessing |
| Development | 6,943 | 85,336 | Tune candidates, thresholds, decoding, and hyperparameters |
| Reserved holdout | 6,943 | 85,399 | Final evaluation after development decisions are fixed |
| Competition test | 23,156 | 285,318 exposed slots; labels unavailable | Inference and submission generation |

Seed: **2026**. The target allocation is approximately 80/10/10 within channel × domain × entity-count bands (1–10, 11–15, 16+). Allocation uses deterministic SHA-256 ordering and keeps whitespace-normalized duplicate groups together. Text normalization is used only for the grouping key, never for stored model input.

All 18 observed channel/domain combinations occur in every supervised partition. Exact counts are recorded in `split_assignments.parquet` and `manifest.json`; rounding and group constraints take priority over exact percentages.

The historical probe's 10,000 training documents remain in train, and its 2,000 evaluated documents remain in development. The holdout excludes 80 known reviewed-example document IDs, including saved annotation-review examples and the two documents printed during initial inspection. Previously trained examples take train priority; other reviewed examples go to development.

**The reserved holdout is not fully blind:** the corpus-wide data-understanding audit already used aggregate information from all gold documents. It was not used by the prior probe for fitting/scoring, but this preparation cannot undo earlier inspection. Semantic near duplicates have not been exhaustively identified.

## Files and contracts

| File | Contents |
| --- | --- |
| `train.parquet`, `dev.parquet`, `holdout.parquet` | `document_id`, exact `full_text`, sorted `entities`, `expected_entity_count` |
| `test.parquet` | `document_id`, exact `full_text`, `expected_entity_count` from submission slots; no labels |
| `test_slots.parquet` | `row_order`, original `row_id`, `document_id`, numeric `slot` |
| `split_assignments.parquet` | Document partition, duplicate-group hash, stratification fields, prior probe role, reviewed-example flag |
| `metadata_audit_only.parquet` | Original metadata plus partition; for checks and subgroup reporting |
| `weak_pool_not_for_supervised_training.parquet` | Separate weak documents, `weak_entities`, annotation source, overlap flags |
| `manifest.json` | Source/code hashes, output hashes and schemas, counts, configuration, runtime versions, limitations |

Each gold entity has `label`, `start`, `end`, and `text`. Offsets count Python Unicode characters with an exclusive end:

```python
assert document["full_text"][entity["start"]:entity["end"]] == entity["text"]
```

Gold entities are sorted by `(start, end)`, matching competition slot ordering. Blank lines, leading spaces in spans, punctuation, Unicode, and multiline addresses are retained. No stripping, case folding, relabeling, truncation, tokenization, or BIO conversion was applied. Token alignment and windowing should be implemented only after choosing the model/tokenizer.

`expected_entity_count` is legitimate inference input for **this competition**, because the sample submission discloses each test document's entity count. In gold partitions it is derived from annotations. Do not describe evaluation using this field as general NER with unknown entity counts.

Model-facing files omit operational fields such as `source_system` and `ingested_at`. Do not automatically load all Parquets as features: split and metadata files are audit artifacts, and the holdout must remain outside fitting and tuning.

For submission output, sort predicted spans within each document by `(start, end)`, assign them to the corresponding slots, and restore the exact original global order using `row_order`. The original placeholder `Predicted` values are intentionally not copied into prepared test data.

## Weak supervision remains separate

All 20,000 weak documents are preserved in a distinct pool. No weak example is included in the train/dev/holdout files. The field is deliberately named `weak_entities` rather than `entities`.

Weak document `DOC_024541` has whitespace-normalized text overlap with the supervised training partition and is flagged `overlap_partitions=["train"]`. No normalized weak-text overlap with development, holdout, or competition test was found. This flag covers exact/whitespace-normalized overlap only, not semantic similarity.

Before any weak-data experiment, decide how to handle duplicate texts and partial labels. Missing weak annotations must not automatically become `O`/negative labels; ADDRESS and JOB_TITLE are absent from the weak labels.

## Validation performed

- Reconstructed every train/test document and matched all four metadata count/length fields.
- Checked gold document coverage, label enums, bounds, exact slices, ordering, and overlap.
- Verified partition completeness/disjointness, duplicate-group isolation, historical role preservation, and known-example exclusion from holdout.
- Verified complete test-slot coverage, unique row IDs, contiguous slot numbers, and preserved source row order.
- Read back every gold/test document and span, all test-slot rows, and every weak record from Parquet and compared them to the source-level records.
- Independently compared all 92,624 prepared metadata rows against original metadata and verified their partitions.
- Verified output hashes and tested a repeated preparation run: it reused the same build without changing files.
- Seven contract tests passed, covering blank lines, Unicode offsets, invalid indices/metadata, exact boundaries, overlapping spans, deterministic grouping, and conflicting historical membership.

The eight Parquet files occupy approximately **49.91 MiB**. Source hashes are checked again before finalizing the build. Validation details are also saved in [`v1_validation.json`](../output/prepared/v1_validation.json).

## Reproduce and load

```powershell
uv run src/prepare_data.py
uv run --with pyarrow python -m unittest discover -s tests -v
```

The preparation script uses its declared PEP 723 dependency in a uv-managed environment. An identical existing build is hash-verified and reused. A changed source/configuration/code hash or modified artifact causes failure rather than silently overwriting the split. For an intentional new version:

```powershell
uv run src/prepare_data.py --output output/prepared/v2
```

Start modeling with train and development only. Example loading from the repository root:

```python
from pathlib import Path
import pyarrow.parquet as pq

data = Path("output/prepared/v1")
for batch in pq.ParquetFile(data / "train.parquet").iter_batches(batch_size=64):
    documents = batch.to_pylist()
    # Learn only from this partition. Keep character offsets unchanged.
```

Preparation decisions follow the [data-understanding report](REPORT.md). Annotation-policy adjudication remains unresolved; no inferred cleaning rules were applied.
