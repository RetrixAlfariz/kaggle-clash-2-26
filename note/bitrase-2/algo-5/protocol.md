# Algo-5 — Conservative CE continuation and refreshed hard negatives

Fixed before training. This is a new adaptive development experiment after the failed algo-4 ranking package. EVAL2 has already informed this choice; do not present these results as a pristine held-out test. No original dev/holdout/test access, pretrained model or submission.

Both arms reuse the verified algo-4 FIT2-only frozen encoder and start from its stable CE control epoch-2 head. Exactly two additional epochs (3 and 4), batch 32, AdamW lr 0.0003, weight decay 0.01, clip 1, FP32/TF32 off, seed 2026. Final epoch only. No temperature sweep or checkpoint selection.

Control continues original P0 IPW CE sampling. Challenger refreshes FIT2 MAP false positives before each epoch and unions them with the original FIT2 BIO false positives. Full width <=16, seven-label candidate universe; no boundary generation changes. Gold boundaries retain their correct positive label; they cannot become NONE negatives due to a wrong predicted label. Unsupported gold retains original protected-gold handling.

Challenger uses weighted CE: positive candidates x4; forced false-positive candidates x4; near-boundary-only candidates x2; remaining candidates x1. For overlapping negative categories use the maximum, not a product. The inverse-probability correction remains, and the denominator is the exact weighted population. This is a bounded weighting choice fixed before results; no weight search.

Batch order and RNG seed rule match by epoch across arms. Samples intentionally differ due to mining. There is no additional ranking loss or lambda. Compare the entire continuation/mining/weighting package, not an isolated causal effect of any one component. The continuation control distinguishes gains obtainable from extra lower-rate CE training alone.

Score fixed final heads once on the same 3,000-document EVAL2 development cohort. Report MAP and Slot-MBR tau=1, exact-slot and entity F1, with paired family bootstrap (2,000 replicates, seed 20260919). A gain >=0.20pp and CI lower >0 is a descriptive useful-effect gate, not fresh confirmatory evidence after adaptation. Report previous control 81.1790% separately from this new continuation control. Crossing 82% means observed internal development accuracy only; replication/external validation remains necessary.
