# Kaggle Clash 2 — Data understanding

Completed on the supplied local snapshot; no model training or competition submission in this audit.

**Decision:** reconstruction is reliable on both splits. Annotation policy and occasional unusual labels require attention before modeling; measured train/test distributions are close. Preserve exact strings, evaluate by domain/channel, and retain operational metadata for auditing rather than assuming predictive value.

Read the [executed notebook](../output/data_understanding/data_understanding.ipynb) or its [HTML preview](../output/data_understanding/data_understanding.html). Detailed examples retain document IDs and offsets in [annotations_examples.json](../output/data_understanding/annotations_examples.json).

## Scope, sources, and method

Full scans: 69,468 training documents, 23,156 test documents, 855,761 gold spans, 285,318 test slots. The optional 20,000 weak documents received structural checks only. Gold convention/context findings use training annotations only.

Source contracts: `note/overview.txt`, `note/description.txt`, `note/evaluation.txt`, `note/data_description.txt`, and `Data/README.txt`. Source-file SHA-256 hashes and runtime versions are recorded in `reconstruction.json`; the notebook manifest also hashes audit code and outputs. Synthetic ingestion dates describe generated operational metadata, not measurement freshness or real event chronology.

## 1. Reconstruction verification

**30/30 explicit integrity checks passed.** Full evidence: [reconstruction.json](../output/data_understanding/reconstruction.json).

| Property | Train | Test |
| --- | --- | --- |
| Documents | 69,468 | 23,156 |
| Segment rows | 1,094,244 | 364,845 |
| Empty segment rows | 419,265 | 139,683 |
| Documents with non-ASCII text | 1,954 | 673 |

- Metadata IDs are unique; segment/metadata joins cover every document. Segment indices are unique, contiguous, and zero-based.
- Reconstructed `n_chars`, `n_segments`, `n_words`, and `n_nonempty_segments` match metadata for every document in both splits.
- All gold document joins, label enums, and exclusive-end offsets are valid. There are no duplicate gold span rows, tied bounds, or overlapping spans (checked against the running maximum end, including nesting).
- One exact duplicate training pair exists: `DOC_121192` and `DOC_108124`; their labels match. Keep them in the same validation group. Test has no exact duplicate documents, and exact/whitespace-normalized train–test overlap is zero.
- All test documents have contiguous slots starting at 01, with unique row IDs. Counts range from 1 to 31. All sample predictions are `NAME:0:0` placeholders; they are not valid spans or hidden labels.
- Weak entity bounds, label enums, and stored text slices are structurally valid. That does not establish annotation quality. Weak labels omit ADDRESS and JOB_TITLE categories entirely.

**Implication:** preserve empty lines and original Unicode. Never strip or normalize model input without a reversible character mapping. The reconstruction-length contract is necessary but not a semantic-label test.

## 2. Annotation convention audit

Full corpus counts below describe the target, not automatically correct linguistic categories. Evidence: [annotations.json](../output/data_understanding/annotations.json).

| Label | Gold spans | Share of spans | Leading whitespace | Trailing whitespace | Multiline spans |
| --- | --- | --- | --- | --- | --- |
| ADDRESS | 81,954 | 9.58% | 1 | 0 | 1368 |
| DATE | 164,044 | 19.17% | 1 | 0 | 0 |
| EMAIL | 96,614 | 11.29% | 1 | 0 | 0 |
| JOB_TITLE | 82,472 | 9.64% | 0 | 0 | 0 |
| NAME | 260,997 | 30.50% | 8 | 0 | 0 |
| PHONE | 92,878 | 10.85% | 3 | 0 | 0 |
| USERNAME | 76,802 | 8.97% | 2 | 0 | 0 |

- The audited honorific pattern is included in 44,120/260,997 NAME spans (16.90%). The pattern and excluded-prefix counts are defined in the script/JSON; this broader pattern differs from the initial probe's narrower pattern.
- An audited honorific appears immediately before, but outside, 125,056 NAME spans. This directly establishes both inclusion and exclusion behavior; it does not establish why they differ.
- Inclusion is not universal: `DOC_010314` labels `Robert Davis` at `[684:696]` after `Mr. `, while `DOC_041205` labels `Attorney Robert Davis` at `[1005:1026]`. A blanket prefix-removal rule is unsupported.
- 1,368 ADDRESS spans contain newlines (1.67%). Restricting entities to single lines would make these spans unrecoverable.
- Leading whitespace is rare but present in gold spans; no trailing-whitespace spans were found. Trimming a predicted span blindly can still disagree with gold. Punctuation adjacency is reported separately from punctuation included inside the span.
- DATE includes time-only values: `DOC_011574` labels `10:00 AM` at `[809:817]`. The written category description does not fully express observed annotation behavior.
- 766 exact values receive more than one label across occurrences. For example, `Emily Carter` has 4,313 NAME annotations and one JOB_TITLE annotation. Cross-label reuse may be contextual or noisy; counts alone do not resolve it.
- There are 4,893 minority-label assignments relative to each exact value's most frequent label (0.57% of gold spans). This is descriptive disagreement, not an estimated annotation-error rate. The top-three cross-label examples include both majority and minority labels.
- A concrete review candidate is `Medical Clinic of Anytown`: USERNAME in `DOC_110801` `[1106:1131]` and `DOC_070407` `[1172:1197]`, but ADDRESS in `DOC_076812` `[1080:1105]`. In these examples it follows signature-role lines. This is a suspected convention/noise issue, not justification to relabel automatically.

**Repetition interpretation:** raw substring matches that differ from a gold span must be split into overlaps with existing annotations and genuinely free occurrences. All 2,231 raw mismatches in 1,990 documents overlap existing gold annotations; 0 are free occurrences. No word-boundary filter is applied. A shorter name inside a longer annotated name is not an omitted annotation. This scan only covers values already labeled somewhere in the same document, so it does not establish overall annotation completeness.

**Action:** preserve original labels; assemble a small adjudication set for prefix boundaries, time-only DATE, organization-like USERNAME/ADDRESS, and minority cross-label assignments. Do not derive a global correction policy from a few examples.

## 3. Label-context profiling

The table reports percent of each gold label assigned to explicit structural heuristics. Header = early labeled header line; opening = early document; signature/tail = closing cue or short late line. These buckets are mutually exclusive but are not human-verified document sections.

| Label | Header | Opening | Body | Signature / tail | Paragraph boundary |
| --- | --- | --- | --- | --- | --- |
| ADDRESS | 0.33% | 12.06% | 55.02% | 11.22% | 21.38% |
| DATE | 7.22% | 20.87% | 58.03% | 4.10% | 9.78% |
| EMAIL | 1.12% | 0.64% | 78.79% | 9.90% | 9.55% |
| JOB_TITLE | 3.06% | 6.04% | 29.88% | 58.63% | 2.39% |
| NAME | 9.98% | 27.51% | 39.22% | 21.94% | 1.36% |
| PHONE | 0.02% | 0.11% | 81.85% | 6.18% | 11.84% |
| USERNAME | 5.70% | 7.28% | 62.57% | 6.93% | 17.51% |

- JOB_TITLE: 58.63% in the signature/tail bucket, but 29.88% in body. A signature-only rule would discard observed targets.
- USERNAME: 34,899/76,802 (45.44%) match the explicit digit/symbol/reference-word heuristic. This is format coverage, not measured semantic accuracy; the category includes account handles and references, and some suspicious values.
- NAME, USERNAME, and JOB_TITLE have saved frequent preceding/following fragments, plus examples with domain/channel. Repeated contextual cues are candidate features, not causal explanations.
- Label-by-channel/domain tables are span counts within each label. Dataset group sizes differ; compare conditional rates rather than interpreting raw totals as group-specific accuracy.

## 4. Metadata signal audit

Evidence: [metadata.json](../output/data_understanding/metadata.json). All nine metadata columns are populated in both splits. Four count/length columns are exactly reconstructible from text. They add no new raw information, although a model could still benefit from an explicit representation.

- Strong redundancy: training Pearson r(n_chars, n_words) = 0.968; r(n_segments, n_nonempty_segments) = 0.959.
- Channel and domain are text-derived descriptors. Use them for profiling and stratified evaluation; their association with annotations is not evidence of independent provenance or causality.

| Metadata field | Bias-corrected Cramer V vs entity-count bins |
| --- | --- |
| source_system | 0.0000 |
| channel | 0.1011 |
| domain | 0.2416 |

Count bins are 0–10, 11–15, and 16+. V measures categorical association on train; a bias-corrected zero is not proof of statistical independence.
- Source-system mean entity counts and per-label span shares are retained in the artifact; no useful signal is established by the checks performed. The data README explicitly describes source_system and ingested_at as synthetic non-predictive fields.
- Ingestion dates cover January 1–March 28, 2026, with 84 distinct dates in both splits. Date-bucket associations with domain, channel, and entity-count bins are very small; no temporal validation scheme is justified by these synthetic dates alone.
- No metadata feature ablation was run. This audit cannot prove that a field has zero predictive value or discover every possible leakage mechanism. Column names alone are not a leakage test.

**Decision:** retain metadata for joins, checks, and reporting. Exclude operational fields from the initial feature set; any later inclusion needs a held-out ablation. Use domain/channel to ensure evaluation coverage.

## 5. Train/test distribution comparison

| Text descriptor | Train mean | Test mean | Standardized mean difference | KS distance |
| --- | --- | --- | --- | --- |
| n_chars | 1195.511 | 1195.224 | -0.0018 | 0.0053 |
| n_segments | 15.752 | 15.756 | 0.0011 | 0.0025 |
| n_words | 180.079 | 180.042 | -0.0015 | 0.0046 |
| n_nonempty_segments | 9.716 | 9.724 | 0.0029 | 0.0036 |

Standardized differences use the square root of the average split variances. KS distance compares empirical CDFs. The small effects support similarity in these four descriptors, not equivalence of all text or labels.

| Category | Train/test total variation distance |
| --- | --- |
| channel | 0.0061 |
| domain | 0.0078 |
| source_system | 0.0054 |

- Entity count mean/median: train 12.3188/12; test 12.3216/12. Both range from 1 to 31.
- Vocabulary: 35,894 shared distinct tokens; 0.7446% of test token occurrences are absent from train. Tokens are case-sensitive whitespace units with punctuation retained; this is not entity novelty or subword-tokenizer coverage.
- Train has three times as many documents as test. Its larger unique vocabulary alone is not evidence of distribution shift.
- Formatting summaries also compare empty segments, newlines, digits, and email-like text. Exact and whitespace-normalized document overlap is zero; semantic near-duplicate overlap was not exhaustively tested.

**Decision:** a document-level, domain/channel-aware holdout with duplicate grouping is a defensible first validation design. This is a proposed design, not one validated by this audit. Hidden test label frequencies, boundary conventions, and model generalization remain unknown.

## Risks, confidence, and next checks

| Finding | Impact / confidence | Next step |
| --- | --- | --- |
| All reconstruction checks pass | Low observed structural risk; full-scan evidence | Keep these checks as ingestion gates |
| Prefix, punctuation, whitespace and multiline conventions | High scoring impact; observed counts and examples | Review boundary-policy cases before normalization |
| Minority cross-label and organization-like values | Potential label noise; interpretation unresolved | Adjudicate examples; retain original labels |
| Operational metadata has weak measured associations | No demonstrated feature value; bounded diagnostics | Exclude initially; ablate only if justified |
| Observable train/test distributions are close | Full-scan descriptive evidence; hidden labels unknown | Use subgroup validation and inspect near duplicates |

## Reproduce and inspect

```powershell
uv run src/audit_reconstruction.py
uv run src/audit_annotations.py
uv run src/audit_metadata.py
uv run src/build_understanding_report.py
uv run src/build_understanding_notebook.py
```

The scripts declare dependencies through PEP 723 and use uv-managed environments. Original files under Data/ and note/ are unchanged. The previous baseline is separate preliminary evidence and is not used as a label-quality reference here.

Artifacts: `reconstruction.json`, `annotations.json`, `annotations_examples.json`, `metadata.json`, `manifest.json`, `data_understanding.ipynb`, `data_understanding.html`, and three notebook figures. Notebook execution independently reconciles label totals, region totals, and source-system entity-count means, and verifies hashes before loading results.
