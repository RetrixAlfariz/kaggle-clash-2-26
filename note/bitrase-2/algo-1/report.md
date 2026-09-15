# bitrase-2.algo-1: Frozen Span Slot-MBR — results

**Completed. Primary tau=1 is research-positive: exact-slot dev accuracy increased from80.9354% to81.3174% (+0.3820 percentage points; +326 correct slots).** The paired95% interval is[+0.0690,+0.6877] points, passing the predeclared+0.20-point/positive-lower-bound gate. However, tau=2 reverses the gain sharply. This is positive performance evidence with strong posterior-concentration sensitivity, not a robust improvement across all fixed temperatures.

The encoder, span head, tokenizer, candidates, and raw logits remained frozen. P0's user-reported **Kaggle0.805 remains the external reference**. P1 has no Kaggle score: no retraining, holdout access, test inference, submission generation, or upload occurred.88% and92% remain unmet.

## Exact verification before comparison

14 tests passed, including1,000 exhaustive randomized decoder fixtures and200 randomized positional-accounting cases. Brute-force enumeration, independently indexed labeled-candidate DP, and grouped-label DP agreed on partitions and full labeled slot marginals; MAP/MBR optima and adversarial cases passed.

All6,943 P0 dev documents replayed exactly at prediction level: identical K, ordered character triples, and complete coverage. Independently canonicalized MAP artifacts share SHA256 `0f79816fa564ae4db43cd94d16e50611ae85062e0cca9aa76dac94a217e64b51`. Matching the aggregate score was not used as the replay criterion.

The frozen artifact contains25,347,544 candidate intervals, eight logits each, for all dev documents. Maximum observed dev length was449 tokens and K was30. Every decoder consumed the same hash-verified artifact.

## Performance and fixed sensitivity

| Decoder | Exact-slot | Exact entity F1 | Correct slots | Delta vs MAP (pp) | Paired95% interval (pp) |
|---|---:|---:|---:|---:|---:|
| P0 MAP |80.9354%|93.1740%|69,067|—|—|
| **Slot-MBR tau=1, primary** |**81.3174%**|92.9080%|**69,393**|**+0.3820**|**[+0.0690,+0.6877]**|
| Slot-MBR tau=0.5, diagnostic |81.2342%|93.1166%|69,322|+0.2988|[+0.1153,+0.4814]|
| Slot-MBR tau=2, diagnostic |77.1152%|91.8827%|65,807|-3.8202|[-4.3583,-3.2955]|

Bootstrap:10,000 paired document resamples, seed20260915, pooled correct counts divided by pooled slots. All85,336 gold slots remain in the denominator. Percentile intervals do not correct repeated development use, checkpoint selection, or dependence between related documents. There were three MBR dev comparisons, plus the mandatory MAP replay.

Tau=1 was primary before results and remains the only primary condition. Tau=0.5 and2 are not selectable alternatives. `research_positive=false` in a sensitivity artifact means it is not eligible as the primary experiment; it does not label the numerically positive tau=0.5 comparison as a loss.

For primary MBR,569 documents gained correct slots,450 lost correct slots, and5,924 retained their correct-slot count. Predictions changed in1,139 documents. All seven labels improved in slot accuracy, while whole-document exactness declined from3,741/6,943 (53.8816%) to3,588/6,943 (51.6779%). Improving expected slot utility can trade away entire-sequence correctness and unordered entity accuracy.

## What changed in the errors

| Diagnostic | P0 MAP | Primary MBR | Change |
|---|---:|---:|---:|
| Correct slots |69,067|69,393|+326|
| Exact unordered entities |79,511|79,284|-227|
| Exact entity, wrong slot |10,444|9,891|-553|
| Same-label overlapping boundary error |2,760|2,763|+3|
| No overlapping prediction |2,593|2,808|+215|
| Same boundary, wrong label |390|395|+5|
| Overlap, wrong label |82|86|+4|

The complete disjoint transition table explains the net gain:

| P0 status of a gold entity | MBR correct slot | MBR exact but displaced | MBR absent as exact triple |
|---|---:|---:|---:|
| Correct slot |67,340|1,317|410|
| Exact but displaced |1,620|8,495|329|
| Absent as exact triple |433|79|5,313|

MBR recovered1,620 previously misplaced exact entities and433 previously non-exact entities into correct slots. It displaced1,317 and lost410 previously correct-slot entities. Thus **1,620+433−1,317−410=326** net additional correct slots. The553 reduction in wrong-rank entities is not all repair:329 such entities were lost, and1,396 newly displaced exact entities appeared.

## Simple-shift mechanism evidence

MAP contained1,890 simple shift episodes encompassing8,993 exact entities. Following those fixed MAP references under primary MBR:

| Outcome | Entities |
|---|---:|
| Repaired to correct slot |1,472|
| Retained, still displaced |7,234|
| Lost as exact triple |287|

There were197 completely repaired episodes and255 partially repaired episodes. Partial repair means some original reference entities became correct; losses remain separately counted and never count as repairs.

The overall pattern is mixed. Simple-episode entities decreased from8,993 to8,802, but the number of simple episodes increased from1,890 to2,015. Documents containing any displaced exact entity increased from1,983 to2,052. MBR therefore repairs some longer shifts while introducing or reorganizing other errors; it does not uniformly clean up every positional diagnostic.

The728 new/changed MBR episode signatures include episodes with modified reference sets, including shortened episodes. This statistic must not be interpreted as728 newly harmed regions. Actual new harm is captured by the correct-to-displaced and correct-to-lost transitions above. Detailed gap audits, episode records, and length histograms are preserved.

**Mechanism conclusion:** there is direct evidence of rank repair contributing to the net slot gain, accompanied by substantial regressions and entity loss. The performance gate passes independently of whether all positional diagnostics improve.

## Posterior sensitivity and next research

Both tau=0.5 and1 improve slots, but tau=2 loses3.82 points. This fails the protocol's stronger all-temperatures-same-direction criterion. It supports studying posterior concentration/calibration; it does not establish that calibration is the sole cause or that fitting a temperature will improve performance.

Existing posterior diagnostics give another descriptive view:

| Temperature | Posterior expected MBR correct-slot fraction | Observed dev correct-slot fraction |
|---|---:|---:|
|0.5|91.5766%|81.2342%|
|1|82.3222%|81.3174%|
|2|54.6072%|77.1152%|

These are aggregate means computed from saved outputs, not a calibration fit or a proof of calibration. Tau=1's close aggregate agreement can conceal errors by confidence range, label, K, or document length. No additional temperature was tried.

The focused follow-up is to explain sensitivity using saved outputs and design a clean calibration experiment if warranted. Any fitted calibration subset must be unseen by both supervised encoder and head, and structured NLL must handle unsupported gold sequences explicitly. Do not jump directly to joint training or a larger encoder on this evidence. The [research brief](research_questions.md) specifies the open questions.

## Validation, runtime, and artifacts

All three prediction artifacts independently passed document coverage, exact-K, ordered unique non-overlapping spans, character bounds, and recomputed slot/entity counts. A separate primary audit reproduced all nine transition counts and matched the new decoder's MAP diagnostic to saved P0 predictions for every document. Maximum partition and slot-normalization residuals across all runs were below2.3e-13. MBR expected utility never fell below MAP utility under its own posterior.

Measured four-worker decoding took60.78s (tau=1),61.03s (tau=0.5), and60.78s (tau=2). Primary median document decoding time was34.9ms under concurrent execution. The largest counted decoder array workspace was12,149,696 bytes; maximum single-worker peak working set across runs was340,615,168 bytes. Frozen extraction took11.64s, used504,531,456 peak CUDA allocation bytes, and produced1,027,734,737 bytes of data files. These are observed local timings, not cross-system benchmarks.

Evidence is under `output/bitrase-2/algo-1/`: `summary.json`, `verification.json`, `verification.log`, `frozen/manifest.json`, `map_replay.json`, and each temperature's predictions, performance, positional diagnostics, and final audit. Primary additionally includes `independent_validation.json`. Primary prediction SHA256: `157c8f049b03b8a23940726a9c6c3cc0a9d2ff045b013e77fafb7f154e7b8c76`.

See [technicalities and commands](technicalities.md) and the [unchanged frozen protocol](protocol.md). P1 establishes a modest dev improvement, not attainment of88%/92% or a new Kaggle result.
