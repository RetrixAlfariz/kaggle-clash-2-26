# Annotation convention audit and label-context profile

Scope: train gold corpus only; reconstructed text and exact character offsets.

Documents: 69,468; annotated documents: 69,468; annotations: 855,761.

- **ADDRESS**: 81,954 spans; left/right whitespace 97.07%/16.26%; internal newline 1.67%; punctuation adjacency 2.93%/78.91%.
- **DATE**: 164,044 spans; left/right whitespace 96.35%/9.07%; internal newline 0.00%; punctuation adjacency 3.17%/87.48%.
- **EMAIL**: 96,614 spans; left/right whitespace 86.42%/39.10%; internal newline 0.00%; punctuation adjacency 13.57%/56.44%.
- **JOB_TITLE**: 82,472 spans; left/right whitespace 96.52%/63.70%; internal newline 0.00%; punctuation adjacency 3.45%/28.75%.
- **NAME**: 260,997 spans; left/right whitespace 98.38%/40.66%; internal newline 0.00%; punctuation adjacency 1.61%/58.72%.
- **PHONE**: 92,878 spans; left/right whitespace 96.18%/37.85%; internal newline 0.00%; punctuation adjacency 3.79%/58.48%.
- **USERNAME**: 76,802 spans; left/right whitespace 60.41%/12.57%; internal newline 0.00%; punctuation adjacency 39.54%/82.47%.

- **NAME honorifics**: 44,120/260,997 (16.90%) begin with the audited pattern; 125,056 NAME spans have a nearby preceding honorific pattern. These are boundary observations, not correctness claims.
- **USERNAME numeric/reference-like**: 34,899/76,802 (45.44%) match the explicit format heuristic.
- **Exact repeats**: 32,887 documents contain repeated annotated values; 1,990 have raw exact matches at other offsets, but all 2,231 such matches overlap a larger gold span and 0 are free. This scan provides no evidence of a free unlabelled repeat.
- **Cross-label exact values**: 766 distinct values occur with more than one label; 4,893 assignments are minority-label occurrences among those values. Inspect cross_label_examples_top3 before using exact-value dictionaries.

Full aggregates are in annotations.json; representative IDs/offset contexts are in annotations_examples.json. Structural regions and context fragments are descriptive heuristics.
