# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Assemble the reviewed five-stage narrative from completed local audit outputs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/data_understanding'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                     *['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows]])


def main():
    r = json.loads((OUT / 'reconstruction.json').read_text())
    a = json.loads((OUT / 'annotations.json').read_text())
    m = json.loads((OUT / 'metadata.json').read_text())
    n = a['denominators']['annotations']
    profile, boundary = a['label_profile'], a['boundary_by_label']
    pct = lambda x, y: f'{100*x/y:.2f}%'
    lines = ['# Kaggle Clash 2 — Data understanding', '',
        'Completed on the supplied local snapshot; no model training or competition submission in this audit.', '',
        '**Decision:** reconstruction is reliable on both splits. Annotation policy and occasional unusual labels require attention before modeling; '
        'measured train/test distributions are close. Preserve exact strings, evaluate by domain/channel, and retain operational metadata for auditing rather than assuming predictive value.', '',
        'Read the [executed notebook](../output/data_understanding/data_understanding.ipynb) or its [HTML preview](../output/data_understanding/data_understanding.html). '
        'Detailed examples retain document IDs and offsets in [annotations_examples.json](../output/data_understanding/annotations_examples.json).', '',
        '## Scope, sources, and method', '',
        f"Full scans: {r['splits']['train']['documents']:,} training documents, {r['splits']['test']['documents']:,} test documents, "
        f"{n:,} gold spans, {r['submission']['rows']:,} test slots. The optional 20,000 weak documents received structural checks only. "
        'Gold convention/context findings use training annotations only.', '',
        'Source contracts: `note/overview.txt`, `note/description.txt`, `note/evaluation.txt`, `note/data_description.txt`, and `Data/README.txt`. '
        'Source-file SHA-256 hashes and runtime versions are recorded in `reconstruction.json`; the notebook manifest also hashes audit code and outputs. '
        'Synthetic ingestion dates describe generated operational metadata, not measurement freshness or real event chronology.', '',
        '## 1. Reconstruction verification', '',
        f"**{r['summary']['passed']}/{r['summary']['checks']} explicit integrity checks passed.** Full evidence: [reconstruction.json](../output/data_understanding/reconstruction.json).", '',
        table(['Property', 'Train', 'Test'], [
            [label, f"{r['splits']['train'][key]:,}", f"{r['splits']['test'][key]:,}"]
            for key, label in [('documents','Documents'),('segments','Segment rows'),('empty_segments','Empty segment rows'),('documents_with_non_ascii','Documents with non-ASCII text')]]), '',
        '- Metadata IDs are unique; segment/metadata joins cover every document. Segment indices are unique, contiguous, and zero-based.',
        '- Reconstructed `n_chars`, `n_segments`, `n_words`, and `n_nonempty_segments` match metadata for every document in both splits.',
        '- All gold document joins, label enums, and exclusive-end offsets are valid. There are no duplicate gold span rows, tied bounds, or overlapping spans (checked against the running maximum end, including nesting).',
        '- One exact duplicate training pair exists: `DOC_121192` and `DOC_108124`; their labels match. Keep them in the same validation group. Test has no exact duplicate documents, and exact/whitespace-normalized train–test overlap is zero.',
        '- All test documents have contiguous slots starting at 01, with unique row IDs. Counts range from 1 to 31. All sample predictions are `NAME:0:0` placeholders; they are not valid spans or hidden labels.',
        '- Weak entity bounds, label enums, and stored text slices are structurally valid. That does not establish annotation quality. Weak labels omit ADDRESS and JOB_TITLE categories entirely.', '',
        '**Implication:** preserve empty lines and original Unicode. Never strip or normalize model input without a reversible character mapping. '
        'The reconstruction-length contract is necessary but not a semantic-label test.', '',
        '## 2. Annotation convention audit', '',
        'Full corpus counts below describe the target, not automatically correct linguistic categories. Evidence: [annotations.json](../output/data_understanding/annotations.json).', '',
        table(['Label', 'Gold spans', 'Share of spans', 'Leading whitespace', 'Trailing whitespace', 'Multiline spans'], [
            [label, f"{profile[label]['count']:,}", pct(profile[label]['count'],n), boundary[label]['leading_whitespace'],
             boundary[label]['trailing_whitespace'], boundary[label]['internal_newline']] for label in sorted(profile)]), '',
        f"- The audited honorific pattern is included in {profile['NAME']['honorific_included']:,}/{profile['NAME']['count']:,} NAME spans "
        f"({pct(profile['NAME']['honorific_included'],profile['NAME']['count'])}). The pattern and excluded-prefix counts are defined in the script/JSON; this broader pattern differs from the initial probe's narrower pattern.",
        f"- An audited honorific appears immediately before, but outside, {profile['NAME']['honorific_excluded']:,} NAME spans. This directly establishes both inclusion and exclusion behavior; it does not establish why they differ.",
        '- Inclusion is not universal: `DOC_010314` labels `Robert Davis` at `[684:696]` after `Mr. `, while `DOC_041205` labels `Attorney Robert Davis` at `[1005:1026]`. A blanket prefix-removal rule is unsupported.',
        f"- {boundary['ADDRESS']['internal_newline']:,} ADDRESS spans contain newlines "
        f"({pct(boundary['ADDRESS']['internal_newline'],profile['ADDRESS']['count'])}). Restricting entities to single lines would make these spans unrecoverable.",
        '- Leading whitespace is rare but present in gold spans; no trailing-whitespace spans were found. Trimming a predicted span blindly can still disagree with gold. Punctuation adjacency is reported separately from punctuation included inside the span.',
        '- DATE includes time-only values: `DOC_011574` labels `10:00 AM` at `[809:817]`. The written category description does not fully express observed annotation behavior.',
        f"- {a['cross_label_distinct_values']:,} exact values receive more than one label across occurrences. For example, `Emily Carter` has 4,313 NAME annotations and one JOB_TITLE annotation. Cross-label reuse may be contextual or noisy; counts alone do not resolve it.",
        f"- There are {a['cross_label_minority_assignments']:,} minority-label assignments relative to each exact value's most frequent label ({pct(a['cross_label_minority_assignments'],n)} of gold spans). This is descriptive disagreement, not an estimated annotation-error rate. The top-three cross-label examples include both majority and minority labels.",
        '- A concrete review candidate is `Medical Clinic of Anytown`: USERNAME in `DOC_110801` `[1106:1131]` and `DOC_070407` `[1172:1197]`, but ADDRESS in `DOC_076812` `[1080:1105]`. In these examples it follows signature-role lines. This is a suspected convention/noise issue, not justification to relabel automatically.', '',
        '**Repetition interpretation:** raw substring matches that differ from a gold span must be split into overlaps with existing annotations and genuinely free occurrences. '
        f"All {a['repetition']['unlabelled_exact_occurrences']:,} raw mismatches in {a['repetition']['documents_with_unlabelled_repeat']:,} documents overlap existing gold annotations; "
        f"{a['repetition']['free_unlabelled_occurrences']:,} are free occurrences. No word-boundary filter is applied. A shorter name inside a longer annotated name is not an omitted annotation. "
        'This scan only covers values already labeled somewhere in the same document, so it does not establish overall annotation completeness.', '',
        '**Action:** preserve original labels; assemble a small adjudication set for prefix boundaries, time-only DATE, organization-like USERNAME/ADDRESS, and minority cross-label assignments. '
        'Do not derive a global correction policy from a few examples.', '',
        '## 3. Label-context profiling', '',
        'The table reports percent of each gold label assigned to explicit structural heuristics. Header = early labeled header line; opening = early document; '
        'signature/tail = closing cue or short late line. These buckets are mutually exclusive but are not human-verified document sections.', '',
        table(['Label','Header','Opening','Body','Signature / tail','Paragraph boundary'], [
            [label, *[pct(a['region_by_label'][label].get(region,0),profile[label]['count'])
             for region in ['header','opening','body','signature_or_tail','paragraph_boundary']]] for label in sorted(profile)]), '',
        f"- JOB_TITLE: {pct(a['region_by_label']['JOB_TITLE'].get('signature_or_tail',0),profile['JOB_TITLE']['count'])} in the signature/tail bucket, "
        f"but {pct(a['region_by_label']['JOB_TITLE'].get('body',0),profile['JOB_TITLE']['count'])} in body. A signature-only rule would discard observed targets.",
        f"- USERNAME: {profile['USERNAME']['numeric_or_reference_like']:,}/{profile['USERNAME']['count']:,} "
        f"({pct(profile['USERNAME']['numeric_or_reference_like'],profile['USERNAME']['count'])}) match the explicit digit/symbol/reference-word heuristic. "
        'This is format coverage, not measured semantic accuracy; the category includes account handles and references, and some suspicious values.',
        '- NAME, USERNAME, and JOB_TITLE have saved frequent preceding/following fragments, plus examples with domain/channel. '
        'Repeated contextual cues are candidate features, not causal explanations.',
        '- Label-by-channel/domain tables are span counts within each label. Dataset group sizes differ; compare conditional rates rather than interpreting raw totals as group-specific accuracy.', '',
        '## 4. Metadata signal audit', '',
        'Evidence: [metadata.json](../output/data_understanding/metadata.json). All nine metadata columns are populated in both splits. '
        'Four count/length columns are exactly reconstructible from text. They add no new raw information, although a model could still benefit from an explicit representation.', '',
        f"- Strong redundancy: training Pearson r(n_chars, n_words) = {m['metadata_redundancy']['numeric_pearson']['n_chars']['n_words']:.3f}; "
        f"r(n_segments, n_nonempty_segments) = {m['metadata_redundancy']['numeric_pearson']['n_segments']['n_nonempty_segments']:.3f}.",
        '- Channel and domain are text-derived descriptors. Use them for profiling and stratified evaluation; their association with annotations is not evidence of independent provenance or causality.', '',
        table(['Metadata field','Bias-corrected Cramer V vs entity-count bins'], [
            [field, f"{m['operational_signal_summary']['categorical_entity_count_association'][field]['entity_count_bin_cramers_v_bias_corrected']:.4f}"]
            for field in ['source_system','channel','domain']]), '',
        'Count bins are 0–10, 11–15, and 16+. V measures categorical association on train; a bias-corrected zero is not proof of statistical independence.',
        '- Source-system mean entity counts and per-label span shares are retained in the artifact; no useful signal is established by the checks performed. '
        'The data README explicitly describes source_system and ingested_at as synthetic non-predictive fields.',
        '- Ingestion dates cover January 1–March 28, 2026, with 84 distinct dates in both splits. Date-bucket associations with domain, channel, and entity-count bins are very small; no temporal validation scheme is justified by these synthetic dates alone.',
        '- No metadata feature ablation was run. This audit cannot prove that a field has zero predictive value or discover every possible leakage mechanism. '
        'Column names alone are not a leakage test.', '',
        '**Decision:** retain metadata for joins, checks, and reporting. Exclude operational fields from the initial feature set; any later inclusion needs a held-out ablation. '
        'Use domain/channel to ensure evaluation coverage.', '',
        '## 5. Train/test distribution comparison', '',
        table(['Text descriptor','Train mean','Test mean','Standardized mean difference','KS distance'], [
            [field, f"{v['train']['mean']:.3f}", f"{v['test']['mean']:.3f}", f"{v['standardized_mean_difference']:.4f}", f"{v['ks_statistic']:.4f}"]
            for field,v in m['numeric_comparison'].items()]), '',
        'Standardized differences use the square root of the average split variances. KS distance compares empirical CDFs. '
        'The small effects support similarity in these four descriptors, not equivalence of all text or labels.', '',
        table(['Category','Train/test total variation distance'], [[field,f"{v['total_variation_distance']:.4f}"] for field,v in m['categorical_comparison'].items()]), '',
        f"- Entity count mean/median: train {m['entity_count_comparison']['train_gold']['mean']:.4f}/12; "
        f"test {m['entity_count_comparison']['test_submission_slots']['mean']:.4f}/12. Both range from 1 to 31.",
        f"- Vocabulary: {m['text_overlap']['case_sensitive_whitespace_token_vocab_overlap']:,} shared distinct tokens; "
        f"{100*m['text_overlap']['test_token_occurrence_oov_rate_vs_train']:.4f}% of test token occurrences are absent from train. "
        'Tokens are case-sensitive whitespace units with punctuation retained; this is not entity novelty or subword-tokenizer coverage.',
        '- Train has three times as many documents as test. Its larger unique vocabulary alone is not evidence of distribution shift.',
        '- Formatting summaries also compare empty segments, newlines, digits, and email-like text. Exact and whitespace-normalized document overlap is zero; semantic near-duplicate overlap was not exhaustively tested.', '',
        '**Decision:** a document-level, domain/channel-aware holdout with duplicate grouping is a defensible first validation design. '
        'This is a proposed design, not one validated by this audit. Hidden test label frequencies, boundary conventions, and model generalization remain unknown.', '',
        '## Risks, confidence, and next checks', '',
        table(['Finding','Impact / confidence','Next step'], [
            ['All reconstruction checks pass','Low observed structural risk; full-scan evidence','Keep these checks as ingestion gates'],
            ['Prefix, punctuation, whitespace and multiline conventions','High scoring impact; observed counts and examples','Review boundary-policy cases before normalization'],
            ['Minority cross-label and organization-like values','Potential label noise; interpretation unresolved','Adjudicate examples; retain original labels'],
            ['Operational metadata has weak measured associations','No demonstrated feature value; bounded diagnostics','Exclude initially; ablate only if justified'],
            ['Observable train/test distributions are close','Full-scan descriptive evidence; hidden labels unknown','Use subgroup validation and inspect near duplicates'],
        ]), '',
        '## Reproduce and inspect', '',
        '```powershell\nuv run src/audit_reconstruction.py\nuv run src/audit_annotations.py\nuv run src/audit_metadata.py\nuv run src/build_understanding_report.py\nuv run src/build_understanding_notebook.py\n```', '',
        'The scripts declare dependencies through PEP 723 and use uv-managed environments. Original files under Data/ and note/ are unchanged. '
        'The previous baseline is separate preliminary evidence and is not used as a label-quality reference here.', '',
        'Artifacts: `reconstruction.json`, `annotations.json`, `annotations_examples.json`, `metadata.json`, `manifest.json`, '
        '`data_understanding.ipynb`, `data_understanding.html`, and three notebook figures. '
        'Notebook execution independently reconciles label totals, region totals, and source-system entity-count means, and verifies hashes before loading results.', '']
    (ROOT / 'note' / 'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(ROOT / 'note' / 'REPORT.md')


if __name__ == '__main__':
    main()
