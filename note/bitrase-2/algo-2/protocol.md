# bitrase-2.algo-2 — Posterior Concentration Audit

Frozen before execution, 2026-09-15. Diagnostic only on P1's frozen dev candidates,
logits and saved MAP/MBR decisions. Temperatures: 0.5, 1, 2. No fitting, selection,
training, candidate changes, holdout access or submission.

For exactly K chronologically ordered nonoverlapping labeled intervals, raw
energy is the sum of positive-label logit minus NONE logit. The posterior is
exp(energy/tau)/Z. Count all labeled structures on this same support, including
all seven labels. Natural logarithms throughout. Compute H=logZ-E[energy]/tau,
log count through the zero-energy partition, H/log count, H/K, and MAP structure
probability. Entropy growth with temperature is expected mathematically; it
cannot alone establish the cause of an error.

For both saved decision rules, slot confidence is the posterior marginal of the
selected complete (start,end,label) at its chronological slot. Actual correctness
requires equality to gold at that slot. Report ten fixed confidence bins
[0,.1),...,[.9,1], pooled ECE, Brier and observed accuracy. These measure the
selected slots, not calibration of every possible candidate or whole structures.

Per document, delta U=sum(MBR selected marginals)-sum(MAP selected marginals);
delta C=MBR correct slots-MAP correct slots. Report both per slot and per document,
their correlation, document outcomes and raw-energy sacrifice E_MAP-E_MBR.
Churn includes changed ordered triples, boundary changes, set removals/additions,
and the nine gold states correct/displaced/absent from the frozen P1 mechanism.
Changed episode signatures must not be described as necessarily newly harmful.

Fixed strata: K 1–10,11–15,16+; token length <=128,129–256,>256;
selected label for reliability; gold label for paired correctness changes.
Local margin means selected label logit minus NONE, in raw units, bins
<0,[0,2),[2,5),>=5. For paired gain strata use the MAP selection's margin and gold
label so membership cannot change across temperatures. Reliability uses each
rule's own selection margin. Also report changed versus unchanged slots.

Uncertainty: 2,000 paired document bootstrap replicates, seed 20260915, percentile
95% intervals for pooled delta C/K, delta U/K and their difference. No iid slot
intervals. Other strata are descriptive, not multiple hypothesis tests.

Before full execution require exhaustive tiny-case entropy/count/marginal tests.
For every document and temperature require partition and saved MAP/MBR expected
utilities to agree within 1e-8, normalized marginals within 1e-8, entropy within
[0,log count] to tolerance, and entropy monotonic across fixed temperatures.
Hash input artifacts and new code before and after execution; abort on changes.

Decision boundary: describe whether tau=2 introduces errors despite predicted
utility gains, where concentration differs, and whether those patterns persist
within fixed K/length strata. This does not isolate calibration, scorer quality,
missing interactions or support as a causal explanation. No next model or optimal
temperature may be chosen from this audit.
