# Train-only document-structure diagnostic — frozen protocol

This is a separate future-scorer diagnostic, not bitrase-2.algo-2, calibration, Slot-MBR, or joint training. Freeze this protocol and feature schema before inspecting real-data probe outcomes. P0 remains unchanged. No dev, holdout, test, or Kaggle data is used for feature choice, fitting, grouping, or evaluation.

## Question and evidence limits

Do the six predefined structural feature families add held-out probe information beyond selected P0 logits, candidate margins, basic shape features, and P0's existing newline flags?

P0's encoder/head already saw every train document. Holding out probe folds does not make these documents unseen by P0. Findings are conditional residual-information diagnostics on train, not estimates of new-document model gains. The controls are finite low-capacity functions of scores/shape, not a claim to control every semantic property of text.

## Train families and bounded sample

Use only prepared train. Union normalized exact duplicates, then approximate template families using label-independent lowercased whitespace-normalized text, digit masking, and word5-shingle MinHash retrieval with the existing fixed8 permutations/four2-hash bands. Verify retrieved pairs at Jaccard>=0.8 before union; connected components define grouping proxies. Include no annotation masking, metadata IDs, split outcomes, or dev/holdout cross-split edges. The retrieval is approximate and generator-family identity is not known.

Sort components by SHA256 of their smallest document ID with fixed seed20260916. Include complete components until at least6,000 train documents are selected. If the resulting sample exceeds12,000 documents, or three balanced folds cannot each contain at least30 components, stop as a feasibility/inconclusive outcome without changing the policy based on scores. Assign complete sampled components greedily to three folds by descending size, deterministic hash tie order, minimizing fold document counts. Preserve all duplicate/template family memberships and report sizes, candidate-edge counts, and fold balance.

## Candidate population and exact target

Define a bounded diagnostic candidate mixture, not the full span-universe population:

- Every token-aligned gold labeled span of width<=16.
- Up to two uniformly sampled distinct valid start-only/end-only boundary perturbations (+/-1,+/-2 tokens) for each representable gold span, with the same candidate label.
- Every old BIO train prediction with its predicted label.
- At each representable gold boundary, the highest-P0-logit alternative entity label.
- Sixteen uniformly sampled width<=16 token intervals per document, labeled by P0's best entity label.

Use seed20260916 and document ID for sampling. Deduplicate labeled triples, retaining source bitmasks. Exact gold triple membership determines correctness; positives override any negative source attribution. Report unrepresentable gold exclusions. Every sampled labeled candidate has weight1. Report AUC/log-loss for this explicitly enriched mixture, not as an estimate of natural all-span prevalence. No oversampling, class weighting, or full-universe accuracy claim.

Score the required interval union using the selected frozen P0 head and verified train hidden cache. No P0 training or updated embeddings. Preserve logits, candidate labels, targets, grouping, fold assignment, features, and source hashes.

## Frozen controls and structural additions

Fit a separate binary logistic probe for each candidate label; its target is whether the entire (start,end,label) triple is exact. Both probes receive identical controls:

- All eight raw P0 logits.
- Candidate entity-minus-NONE margin and candidate-minus-best-other-entity margin.
- Fixed hinge terms of candidate-minus-NONE margin at -4,0,+4, to reduce simple score-nonlinearity confounding.
- Basic shape: log token/character length, upper/digit/punctuation fractions, first/last-token title/upper/digit flags.
- Existing P0 newline-before/newline-after flags at the span endpoints and their span-token means; existing endpoint token-adjacency flags.

Only the augmented probe receives the six structural families:

1. Normalized physical start/end line indices.
2. Normalized start/end paragraph indices and paragraph-start/end flags; paragraphs are separated by blank physical lines.
3. Span-crosses-line flag.
4. Same-line preceding/following colon indicators, normalized distance to a preceding colon with an explicit absence indicator, and colon-inside-span flag.
5. Normalized character start/end positions within the document.
6. Previous/next physical-line nonblank indicators, colon presence in nearest previous/next nonblank lines, and normalized distances to those lines.

All features derive from raw text and offsets. No inferred signature/header labels, keyword lexicon, parser, pretrained model, graph, learned line embeddings, family IDs, or label counts enter the probes. Final scalar names/formulas are recorded in feature_schema.json and source hashes before extraction; no post-outcome additions or removals.

## Probe fit and paired evaluation

Use the same standardized ridge-logistic model for control and augmented inputs. Fit standardization only on the two training folds. Mean binary cross-entropy plus0.001/2 times squared non-intercept coefficients; no class weights or hyperparameter search. Deterministic float64 solver, maximum100 iterations, documented convergence checks. Require both classes in training; unsupported labels/folds are explicitly inconclusive rather than imputed as successful fits.

Produce out-of-fold predictions for all sampled candidates. Report pooled and per-label candidate AUC and log-loss, plus each fold. Also report candidate-source counts to distinguish hard errors from easy random negatives.

Pairwise task: for each representable gold labeled span, compare against up to four highest-P0-energy incorrect hard candidates of the same candidate label in the same document, drawn from boundary perturbations, old-model predictions, or alternative-label candidates. Exclude random-only negatives. This fixed hard-pair list is identical for both probes. Win=1, tie=0.5, loss=0. Every pair has weight1. Report eligible gold/pairs and label coverage; this is not a naturally weighted document metric.

## Uncertainty, minimum useful effect, and stopping

Use1,000 paired family-component bootstrap resamples, seed20260916, of the fixed out-of-fold predictions. Recompute weighted exact AUC and log-loss with family multiplicities, and pairwise accuracy from family win/total counts. Report95% percentile intervals for improvements: augmented-minus-control AUC and pairwise accuracy, control-minus-augmented log-loss (absolute and relative). Resampling does not refit probes or account for P0's train exposure. Per-label intervals are descriptive and not independently sufficient for promotion.

Advance only if all pooled point estimates meet these predeclared minima:

- AUC improvement>=0.005 (0.5 percentage points).
- Relative log-loss reduction>=1%.
- Pairwise ordering improvement>=0.01 (1 percentage point).

Additionally require positive lower95% bounds for all three improvements, positive log-loss and pairwise improvements in all three folds, and positive pairwise point estimates in at least five of seven labels with at least50 positive candidates,50 negative candidates,50 hard pairs, and20 families in the pooled evaluation.

If the gate fails, do not pursue a dedicated structure-aware scorer from this diagnostic or search for alternative features on dev. Distinguish a sufficiently precise failure to reach the useful-effect threshold from wide/inconclusive intervals or poor family support. A failure of this feature/probe specification is not proof that every possible structural architecture is useless.

If the gate passes, recommend only a future controlled addition of the same small features to a span scorer before considering a graph/GCN. No production-model change is implemented here.
