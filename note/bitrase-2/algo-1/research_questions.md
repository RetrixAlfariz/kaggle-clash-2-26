# Follow-up for Aestem after bitrase-2.algo-1

P1 is complete. See the [report](report.md). Research only: do not launch new training, fit calibration, sweep temperature, inspect holdout, or submit to Kaggle as part of this request.

## Observed evidence

- Frozen P0 MAP:80.9354% slot,93.1740% entity F1.
- Primary Slot-MBR tau=1:81.3174% slot (+0.3820pp; paired95% CI[+0.0690,+0.6877]),92.9080% entity F1. Performance gate passed.
- Fixed tau=0.5:81.2342% slot; fixed tau=2:77.1152%. No temperature was selected from these results.
- Primary correct-slot transitions:1,620 misranked exact entities repaired;433 previously non-exact entities now correct;1,317 formerly correct entities displaced;410 formerly correct entities lost. Net+326.
- Simple MAP episode references:1,472 repaired,7,234 still displaced,287 lost;197 episodes completely repaired. Total simple episodes increased despite fewer entities inside them.
- Average posterior expected MBR slot fractions at tau0.5/1/2 are91.5766%/82.3222%/54.6072%, versus observed81.2342%/81.3174%/77.1152%. Aggregate agreement is not calibration proof.
- Exhaustive verification and complete MAP replay passed. Numerical residuals are below2.3e-13. No current evidence suggests an inference bug.

## Primary research question

Why does broadening this posterior at tau=2 damage slot decisions so sharply, and what bounded experiment would distinguish posterior concentration from wrong relative energies or missing span interactions?

First propose descriptive diagnostics using the already saved fixed-temperature outputs. Distinguish confidence reliability, changes in selected spans, and expected-utility gain under the model from realized slot gain. Consider stratification by K, token length, label, local competing-span ambiguity, and rank displacement. Freeze bins and comparisons before new analysis; do not mine dev for a new selection rule.

Explain how the number of competing labeled interval structures can affect posterior mass when temperature increases. This is a possible mechanism to investigate, not an established cause. Clarify whether any proposed diagnostic can actually distinguish this effect from local score miscalibration.

## If proposing fitted calibration

Design one properly separated experiment with a calibration set unseen by every supervised encoder/head parameter. The current P0 model used the full training partition; taking a subset out only after training is insufficient. Define treatment of exact duplicate/related documents using the existing preparation/split policy.

Specify one calibration objective and a fixed evaluation plan. Structured NLL is infinite when the entire gold sequence is outside the fixed universe. State how unsupported documents are handled, exclusions and their representativeness, and whether the resulting objective applies only to a subset. Do not quietly expand P1's candidate universe or fit on dev. Quantify the cost of rebuilding an honest calibration setup before recommending it.

Do not assume temperature can reverse a wrong local label/energy ordering. State which outcome would instead motivate a separate head/scoring or joint-training experiment. P1 does not identify the encoder as the bottleneck.

## Deliverable and target boundary

Return one prioritized next experiment, its falsifiable question, fixed budget, provenance/split requirements, decision gate, and what positive and negative outcomes would permit us to conclude. Keep performance evidence separate from the explanatory mechanism.

88% remains a research target and92% a stretch target. P1 adds326 correct dev slots; it does not recover the6,029 or9,443 additional slots that were needed from P0 to reach those targets. Other error categories could improve, but no extrapolation of P1's gain establishes reachability. P0's user-reported Kaggle0.805 remains the external reference.
