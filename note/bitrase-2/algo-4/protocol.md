# bitrase-2.algo-4 — Frozen-Encoder Global Ranking Objective

Status: execution specification fixed before training, following Vian's request for autonomous improvements toward 82%. No performance results exist at this freeze. See [exposure preflight](report.md).

## Operational decisions fixed before execution

Historical in-sample training diagnostics are permitted, with their research-adaptation caveat retained. Prior dedicated evaluation/calibration/probe exposure and unresolved reviewed examples remain excluded. This resolves the preflight classification question under the user's subsequent autonomous-improvement instruction; it does not erase the recorded historical exposure.

Use existing discovered family membership, sorting eligible whole families by SHA256("20260919:eval2:" + family_id), then family ID. Take whole families until >=3,000 documents; preserve prepared TRAIN order within FIT2/EVAL2. Freeze assignments/configuration before runtime measurement. Seed 2026 for encoder and heads. Benchmark and lambda document order use the same hash construction with tags "runtime" and "lambda" respectively.

Runtime uses 256 FIT2 documents' actual token lengths with synthetic normal hidden states and a random head; it tests scoring/mining costs without using labels for method selection or an old model. Four CPU workers. A projected full-FIT mining pass above 7,200 seconds stops before training. Exact DP may be optimized without changing feasible structures or energies; deterministic transition ties prefer skip, interval start, top label (energy then label ID), predecessor rank. Exhaustive tiny-case checks precede training. Final evaluation uses the existing MAP/Slot-MBR tie conventions.

Lambda uses 512 fully supported, positive-K FIT2 documents at initialization, fixed batches of 32, CE sample epoch 1. Rank norm <=1e-12 is effectively zero. Every epoch's monitor uses those same CE samples, before and after training, against that epoch's frozen competitors. Report the component norms, not a claim of equal AdamW influence.

Head CE batch order is shuffled with Python Random(2026 + epoch). Persist orders and hash all candidate indices, targets, weights and per-batch RNG seeds for both arms. Reset CE RNG to 2026 + epoch*100000 + batch_index. Heads use FP32 with TF32 disabled; encoder uses the inherited AMP overflow-retry implementation. Compute final MAP/MBR tau=1 metrics once on EVAL2 with 2,000 whole-family bootstrap replicates, seed 20260919, pooled slot accuracy. Do not use EVAL2 to select checkpoints, temperatures or further hyperparameters within algo-4.

The 82% target refers to observed internal exact-slot accuracy; it is not a promised Kaggle score. A positive algo-4 remains a research candidate pending replication.

## Data and initialization

Prepared TRAIN only. Exclude prior dedicated evaluation, calibration, structure-probe and unresolved-history documents at whole-family level. Permit historical in-sample training diagnostics as specified above. Target EVAL2 3,000; if fewer are available, retain the largest eligible whole-family set only if >=2,000, otherwise stop. FIT2 is its complement. Persist membership, source hashes, grouping rules and deterministic split seed before training. No balance-based retries.

Build vocabulary and a fresh randomly initialized encoder using FIT2 only. Four encoder epochs; batch 32; AdamW lr 0.001, weight decay 0.01, clip 1. Freeze encoder. Two heads start from identical initialization and train for two epochs with the same optimizer settings, FP32, final epoch only. No checkpoint selection. No pretrained model, extra data, original checkpoint reuse, dev/holdout/test access, or submission.

## Treatment

Control: original inverse-probability-weighted CE. Challenger: that CE plus lambda times document-averaged ranking loss over supported documents in the minibatch:

`softplus((S(strongest_incorrect) - S(gold)) / K)`.

No additional cost weight. Unsupported documents retain the original protected-gold CE treatment and are excluded from ranking; a batch with no supported documents contributes zero ranking loss. Energies are positive-label logits minus NONE, over the unchanged width <=16, seven-label, exact-K nonoverlapping candidate universe.

At the beginning of each challenger epoch, the strongest incorrect feasible structure is mined using the current head. Its identity is frozen for that epoch. It is therefore an epoch-wise mined competitor, not necessarily the strongest competitor after subsequent parameter updates.

Use exact top-two labeled-structure DP: top one if incorrect, otherwise top two. Mining is eval mode with dropout off. Rank forward passes also have dropout off but gradients enabled; identities are detached. Re-mine from each final head for the final ranking diagnostic.

## Lambda and CE matching

Before optimizer steps, measure gradients on 512 deterministic supported FIT2 documents, in batches of 32, at common head initialization. All head parameters, FP32, dropout off, encoder excluded. The estimator is the ratio of medians:

`lambda = median_batch(||gradient CE||) / (median_batch(||gradient rank||) + 1e-12)`.

Nonfinite values, effectively zero rank norms or lambda outside [0.01,100] stop for configuration review. Do not clip or sweep lambda. Freeze lambda once and monitor component losses/norms using the same measurement batches each epoch, with tolerance and competitor timing as specified above.

Persist identical CE candidates, inverse-probability weights and batch orders for both arms. Reset CPU/CUDA CE RNG per minibatch from global seed, epoch and batch index so mining/ranking does not consume CE dropout draws. **CE exposure is matched**; total gradient exposure intentionally differs. FP32 is not a claim of universal bitwise GPU reproducibility.

## Runtime and verification gates

Before full training, benchmark scoring and exact top-two mining on 256 deterministic FIT2 documents. Report timings, peak RAM/VRAM and projected full-FIT mining time. No label-based method selection; no heuristic substitution. If runtime is impractical, amend the protocol before full training.

Verify exact top-two including label alternatives, ties and unsupported gold on exhaustive tiny cases. Verify CE candidate/weight/order/RNG parity, rank gradients, final-head re-mining and immutable evaluation coverage before evaluation. Record implementation hashes and operational constants before running.

## Evaluation and interpretation

One final EVAL2 evaluation, paired by document/family. Separate gates: MAP exact-slot improvement >=0.20 percentage points with bootstrap 95% lower bound >0, and MBR tau=1 exact-slot improvement under the same threshold/interval requirement. Bootstrap settings are fixed above. Neither gate substitutes for the other.

Report gold-versus-strongest-wrong ranking, local replacements, entity F1, exact-but-displaced entities, repaired/displaced/lost slots and margins. No temperature fitting, EVAL checkpoint selection or candidate changes.

A positive result supports this package of CE, ranking loss, epoch-wise mining, fixed lambda and optimization settings on this research evaluation. It does not isolate global structure as the cause. Require seed replication before external promotion. A negative result rejects this tested package; it does not prove an encoder bottleneck. No inference that 88% or 92% Kaggle is reachable follows in advance.
