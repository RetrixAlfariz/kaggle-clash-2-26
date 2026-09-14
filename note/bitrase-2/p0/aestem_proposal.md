# Frozen Direct-Span Scorer Experimental Protocol

**Status:** protocol proposal, no implementation or training performed
**Scope:** frozen Byte-CNN + 2-layer BiLSTM encoder, no pretrained model, no external data, no reserved holdout
**Primary milestone:** 88% exact-slot
**Stretch target:** 92% exact-slot
**Purpose of this experiment:** determine whether direct span classification makes better use of the existing epoch-4 encoder representation than its original BIO head.

---

# 1. Decision Page

## 1.1 Research question

The experiment tests exactly one primary hypothesis:

> Given the same frozen epoch-4 Byte-CNN + BiLSTM encoder, does replacing token-level BIO classification with direct span classification improve exact-slot performance after exact-K non-overlapping decoding?

This is deliberately narrower than testing whether span-based NER in general is superior.

A failure therefore rejects only:

> **this frozen-encoder direct-span probe**

and does **not** prove that joint span training, biaffine scoring, or span-oriented encoders are ineffective.

The literature provides a credible basis for testing span enumeration: exhaustive span-based NER has previously represented enumerated spans using shared BiLSTM outputs, while biaffine NER explicitly scores start/end pairs. Global Span Selection additionally demonstrates that span scores can be combined with global structured selection instead of treating every span independently.

These papers motivate the experiment; **their benchmark gains are not predictions for this competition**.

---

## 1.2 Frozen component

Use:

**epoch-4 Byte-CNN + 2-layer BiLSTM encoder**

and freeze every parameter belonging to:

* word embeddings;
* byte embeddings;
* byte CNN;
* token feature inputs;
* both BiLSTM layers.

The current repository constructs a 2-layer bidirectional LSTM with hidden size 128 per direction and passes its output directly to a 15-way BIO linear classifier. Therefore the span experiment should use the **256-dimensional output of the second BiLSTM layer immediately before `self.out`**.

Do **not** use:

* BIO logits;
* intermediate word embeddings alone;
* first-layer LSTM states;
* epoch-8 averaging;
* ensemble hidden states.

Using epoch 4 alone is necessary for the direct comparison against the **79.4225% BIO epoch-4 baseline**.

Before span training, the extracted frozen hidden states must reproduce the original BIO logits when passed through the frozen `self.out` layer. This is an implementation integrity test, not a model experiment.

---

## 1.3 Main span representation

For token span \((i,j)\), inclusive:

$$
r_{ij} =
[
h_i;
h_j;
h_i\odot h_j;
\operatorname{mean}(h_i,\ldots,h_j);
e_{\text{width}(i,j)}
]
$$

where:

$$
h_i,h_j\in\mathbb{R}^{256}
$$

and:

$$
e_{\text{width}}\in\mathbb{R}^{32}.
$$

Thus:

$$
\dim(r)=4(256)+32=1056.
$$

### Why retain all five components?

`h_start` and `h_end` are essential because exact boundaries are the target.

`mean(h_span)` exposes information from the internal tokens rather than expecting the endpoints to summarize everything.

`width_embedding` supplies an inexpensive learned length prior.

`h_start ⊙ h_end` is retained because it introduces an explicit multiplicative endpoint interaction at negligible cost compared with replacing the head by a full biaffine scorer. This is our lightweight analogue of the start/end interaction motivated by biaffine span scoring.

No feature ablation is performed in the primary run.

---

## 1.4 Main head

Use:

$$
1056
\rightarrow
256
\rightarrow
8
$$

with:

```text
LayerNorm(1056)
Linear(1056, 256)
GELU
Dropout(0.20)
Linear(256, 8)
```

Classes:

```text
NONE
NAME
DATE
EMAIL
PHONE
ADDRESS
USERNAME
JOB_TITLE
```

Estimated trainable parameters:

$$
\approx275\,000
$$

including the width embedding and LayerNorm.

This is intentionally small relative to the frozen encoder.

### Optimization

Use:

* AdamW;
* learning rate `1e-3`;
* weight decay `0.01`;
* gradient clipping `1.0`;
* mixed precision;
* no scheduler;
* no focal loss;
* no ranking loss;
* no boundary-regression loss;
* no segmental loss.

Those optimizer values match the scale already used by the neural sequence experiment, reducing unnecessary new degrees of freedom. The existing sequence training uses AdamW at `1e-3`, weight decay `.01`, clipping `1.0`, batch size 32, and mixed precision.

---

## 1.5 Candidate width

Fix:

$$
W_{\max}=16
$$

before looking at probe results.

Reason:

$$
99.9686\%
$$

of gold train entities are representable using aligned spans of at most 16 tokens.

Do **not** tune `Wmax` on dev.

The 187 unaligned train entities and 28 aligned entities longer than 16 tokens remain outside the model's representable gold universe.

They are not silently repaired or deleted.

---

## 1.6 Primary decoding

At inference:

1. enumerate **every** token-aligned span with width \(1\ldots16\);
2. score all seven entity labels;
3. reduce each span to its best entity label;
4. run exact-K weighted interval DP;
5. select exactly \(K\) mutually non-overlapping spans;
6. map selected token boundaries back to original Python character offsets;
7. sort by `(start_offset, end_offset)`.

No score threshold and no span pruning are allowed in P0.

This ensures that a performance difference is attributable primarily to span scoring, not a newly introduced candidate-recall heuristic.

---

# 2. Token and Character Contract

The repository tokenizer is:

```text
\w+|[^\w\s]
```

and returns each token as:

```text
(char_start, char_end, token_text)
```

where the character end is already Python-style exclusive.

For span token indices:

$$
(i,j)
$$

the character prediction is therefore exactly:

$$
start = token_i.start
$$

$$
end = token_j.end.
$$

No reconstructed offsets and no normalization are allowed.

The invariant is:

```text
full_text[start:end]
```

must exactly equal the predicted textual span.

---

# 3. Efficient Span Representation

Computing:

$$
\operatorname{mean}(h_i,\ldots,h_j)
$$

by looping through every span would be wasteful.

For each document define prefix hidden-state sums:

$$
P_0=0
$$

$$
P_{t+1}=P_t+h_t.
$$

Then:

$$
\operatorname{mean}_{i:j}
=
\frac{P_{j+1}-P_i}{j-i+1}.
$$

Use FP32 for the prefix accumulation, even if cached hidden states are FP16.

The resulting span feature may then be cast to the mixed-precision dtype used by the MLP.

Complexity for means becomes effectively:

$$
O(1)
$$

per span after an:

$$
O(T)
$$

prefix construction.

---

# 4. Frozen Encoder Cache

Because the encoder never updates, hidden states should be generated **once** and treated as a derived immutable artifact.

For every document cache:

```text
document_id
token_start_char[]
token_end_char[]
token_hidden[T, 256]
token_count
```

Recommended hidden-state storage:

```text
float16
```

while original token character offsets remain integers.

The cache manifest must contain hashes of:

* epoch-4 checkpoint;
* `vocab.json`;
* `neural_sequence.py`;
* `sequence_crf.py`;
* prepared train/dev artifacts;
* cache generator source.

The current model code does not expose pre-classifier hidden states directly; it returns `self.out(x)`. Implementation will therefore need an explicit encoder-output path or equivalent refactoring.

Required integrity check:

$$
self.out(cached\_h)
$$

must reproduce the original epoch-4 BIO logits within the expected numerical tolerance and produce identical BIO decisions/known-K predictions.

No span experiment proceeds if this fails.

---

# 5. Training Candidate Universe

For a document with \(T\) tokens:

$$
U_d=
\{
(i,j):
0\le i\le j<T,\;
j-i+1\le16
\}.
$$

For \(T\ge16\):

$$
|U_d|=16T-120.
$$

Every candidate is a **span**, not `(span, label)`.

Its target is one of the eight classes.

---

# 6. Positive Spans

Every gold entity satisfying both conditions:

1. start and end align exactly to tokenizer boundaries;
2. token width \(\le16\);

is included with probability:

$$
q=1.
$$

Target:

```text
gold label
```

All representable gold positives are used every epoch.

If a candidate is selected by any negative-generation procedure but exactly matches a gold span, **gold wins**.

Thus a current model prediction with:

```text
same start
same end
wrong label
```

is trained using the **correct gold label**, never `NONE`.

---

# 7. Protected Unrepresentable Gold

The following gold entities cannot be represented by the P0 candidate universe:

* unaligned entities;
* aligned width \(>16\).

During training, candidate spans that overlap one of these protected entities should be removed from the **negative sampling pool**.

Reason:

otherwise the model could be explicitly taught that a truncated representation of an entity is `NONE`, even though the failure arises from our candidate definition rather than the annotation.

These protected entities remain fully present in dev evaluation.

There is no equivalent protection at test inference, because gold is unavailable.

---

# 8. Near-Boundary Hard Negatives

G2 shows that a large fraction of observed boundary errors are local, particularly for NAME, JOB_TITLE, USERNAME, EMAIL, and PHONE.

The primary protocol therefore uses a deliberately narrow perturbation scheme.

For gold token span:

$$
(i,j)
$$

generate:

### Start-only perturbations

$$
(i+\delta,j)
$$

for:

$$
\delta\in\{-2,-1,+1,+2\}
$$

and:

### End-only perturbations

$$
(i,j+\delta)
$$

for the same \(\delta\).

Retain only spans that:

* remain inside document boundaries;
* have width \(1\ldots16\);
* are not exact gold spans;
* do not overlap protected unrepresentable gold.

Maximum:

$$
8
$$

near-boundary negatives per gold entity before deduplication.

This specifically covers:

* the observed two-token NAME start-boundary issue;
* JOB_TITLE end-boundary shortening/extension;
* other local boundary alternatives.

It intentionally **does not** enumerate the full \(5\times5\) boundary grid.

The objective is to make negatives difficult without filling the sample with every trivial perturbation.

ADDRESS is not given special additional local candidates in P0 because only 29.1% of its G2 pairs had both boundaries within ±2 tokens. The diagnostic does not support assuming ADDRESS is mainly a local-offset problem.

---

# 9. Existing-Model False Positives

Use epoch-4 Byte-CNN/BiLSTM predictions on **train only**.

Inference should use the existing epoch-4 known-K decoder because that matches the competition regime.

For every train document:

```text
old_predictions = epoch4_exact_K(document)
```

A predicted span becomes a hard candidate when it is not an exact gold entity.

However:

* if its boundaries equal a gold entity but its predicted label is wrong, assign the true gold label;
* if it is already in the near-boundary pool, store it once;
* retain multi-source tags for diagnostics.

### Repository requirement

No existing train-prediction artifact was found in the repository search.

Therefore implementation will need to generate a frozen epoch-4 **train inference artifact** before span-head training.

Required sources:

```text
output/bitrase-1.algo-6/v1/epoch4.pt
output/bitrase-1.algo-6/v1/vocab.json
output/prepared/v1/train.parquet
src/neural_sequence.py
src/sequence_crf.py
```

This inference artifact must not read dev or holdout labels.

---

# 10. Random Negatives

After removing:

* gold positives;
* protected spans;
* forced near-boundary spans;
* forced old-model false positives;

let the remaining `NONE` pool contain:

$$
R_d
$$

spans.

Sample uniformly without replacement:

$$
m_d=\min(256,R_d)
$$

per document, per epoch.

Random sampling is refreshed deterministically from the experiment seed and epoch.

This gives broad negative coverage while keeping computation substantially below exhaustive span training.

---

# 11. Deduplication and Group Priority

Operational precedence:

```text
1. exact representable gold
2. protected-unrepresentable overlap → excluded from negative sampling
3. epoch-4 train false positive
4. near-boundary negative
5. random negative
```

A span stored under categories 3 and 4 appears only once in training but may retain both diagnostic tags.

The target is determined by gold first.

Therefore:

> A candidate that exactly matches any gold entity can never become `NONE` because an older model or hard-negative generator happened to nominate it.

---

# 12. Sampling Probability

The protocol uses unequal inclusion probabilities, so they must be recorded explicitly.

If several independent sampling stages \(g\) could nominate candidate \(c\), the combined inclusion probability would be:

$$
q(c)
=
1-
\prod_g
(1-q_g(c)).
$$

This follows directly from the probability that at least one selection mechanism includes the candidate.

In the chosen protocol, however, stages are made disjoint by precedence.

Thus the actual inclusion probability simplifies to:

$$
q(c)=
\begin{cases}
1 & c\in F_d\\[4pt]
m_d/R_d & c\in R_d,\ R_d>m_d\\[4pt]
1 & c\in R_d,\ R_d\le m_d
\end{cases}
$$

where \(F_d\) is the forced set:

```text
gold + near-boundary + old false positives.
```

Horvitz and Thompson established that inverse-inclusion weighting can recover an unbiased estimate of a finite-population total under unequal-probability sampling.

---

# 13. Primary Loss

Use **inclusion-probability-weighted categorical cross-entropy**.

For candidate \(c\):

$$
\ell_c
=
-\log p_\theta(y_c\mid r_c).
$$

For sampled candidates in mini-batch documents \(B\):

$$
L_B
=
\frac{
\sum_{d\in B}
\sum_{c\in S_d}
\frac{\ell_c}{q(c)}
}{
\sum_{d\in B}|U_d^{train}|
}.
$$

Here \(U_d^{train}\) excludes protected candidates described earlier.

This is the Horvitz-Thompson-style estimator of the full training-candidate loss for those batch documents.

Do **not** normalize using:

$$
\sum_{c\in S}1/q(c)
$$

in the primary implementation, because that creates a ratio estimator rather than the direct HT total divided by the known population size.

Do not clip weights in P0.

Instead report:

```text
min q
median q
P95 weight
max weight
```

to expose whether variance becomes extreme.

One known property of inverse-probability estimators is that very small inclusion probabilities can cause high variance.

No additional entity-class weighting is used.

---

# 14. Why Not Train on Every Span?

Exhaustive span classification is a legitimate NER approach. Sohrab and Miwa explicitly enumerate candidate spans and classify them using shared BiLSTM representations.

But for this probe, exhaustive training wastes most computation on easy `NONE` spans.

For \(T\ge16\):

$$
M(T)=16T-120.
$$

At an illustrative \(T=256\):

$$
M=3976
$$

spans/document.

By contrast, before old-model false positives, our average sampled count is approximately:

$$
12.3246
+
8(12.3246)
+
256
\approx367
$$

spans/document before deduplication.

Thus sampling reduces the span-head training workload by roughly an order of magnitude under this illustrative token length.

**This is a compute comparison, not a score prediction.**

---

# 15. Decoder Score

For candidate span \(s\) and non-NONE label \(l\), use:

$$
score(s,l)
=
z_l(s)-z_{NONE}(s)
$$

where \(z\) denotes raw logits.

Under an eight-class softmax:

$$
z_l-z_{NONE}
=
\log
\frac
{p(l\mid s)}
{p(NONE\mid s)}
$$

for the model's own fitted distribution.

The score is therefore interpretable as model evidence for an entity label relative to `NONE`.

### Important limitation

This does **not** mean:

* the score is perfectly calibrated;
* spans are probabilistically independent;
* negative sampling bias disappears automatically.

The inclusion-weighted loss is used precisely because negative sampling can otherwise distort the learned candidate distribution.

Even with weighting, model misspecification and overlapping correlated candidates remain.

For exact-K inference, an additive constant applied equally to every selected span would cancel because every feasible solution contains exactly \(K\) spans. Span-dependent distortions do **not** cancel.

---

# 16. Label Reduction Before DP

Because P0 has:

* no label transition scores;
* no label-specific overlap rules;
* no slot-conditioned scores;

the label can be maximized independently for each span:

$$
l_s^*
=
\arg\max_{l\ne NONE} score(s,l)
$$

and:

$$
v_s
=
\max_{l\ne NONE} score(s,l).
$$

The interval decoder therefore receives one:

```text
(start, end, best_label, best_score)
```

candidate per span.

This reduction would no longer be valid if future models introduced inter-label transitions or slot-specific label interactions.

---

# 17. Exact-K Weighted Interval Decoder

Let candidates be sorted by increasing token end position.

For candidate \(i\):

```text
start_i
end_i
score_i
label_i
```

define:

$$
p(i)
=
\max\{j<i:end_j<start_i\}.
$$

The strict inequality:

$$
end_j<start_i
$$

allows adjacent spans:

```text
previous ends at token 8
next starts at token 9
```

while preventing token overlap.

Define:

$$
DP[i,k]
$$

as the maximum score obtainable from the first \(i\) candidates using exactly \(k\) spans.

Initialization:

$$
DP[0,0]=0
$$

$$
DP[0,k>0]=-\infty.
$$

Recurrence:

$$
DP[i,k]
=
\max
\left(
DP[i-1,k],
DP[p(i),k-1]+v_i
\right).
$$

Final solution:

$$
DP[M,K].
$$

Store backpointers to recover selected spans.

### Complexity

Sorting and predecessor computation:

$$
O(M\log M)
$$

DP:

$$
O(MK)
$$

Memory:

$$
O(MK)
$$

with straightforward backpointers.

Because:

$$
M=O(16T)
$$

for fixed maximum width, this remains modest.

Global structured span selection via dynamic programming has precedent in NER literature, although the exact objective here is competition-specific rather than a reproduction of that paper.

---

# 18. Decoder Infeasibility

P0 must never invent placeholder spans.

Before decoding verify:

$$
0\le K\le T.
$$

Because every one-token span belongs to the candidate universe, \(K\le T\) guarantees at least \(K\) mutually non-overlapping candidate intervals.

If:

$$
DP[M,K]=-\infty
$$

despite this invariant, treat the run as an implementation failure.

Do not:

* lower K;
* reuse a span;
* inject dummy spans;
* invoke gold;
* silently fall back to BIO.

A BIO fallback would contaminate the experiment because the resulting score would no longer measure the direct-span system.

---

# 19. Inference Candidate Scoring

Inference uses **all** width-\(\le16\) spans.

No negative sampling.

For memory safety, score candidates in fixed chunks of:

```text
32,768 spans
```

per forward pass of the head.

Store only:

```text
span start token
span end token
best label
best score
```

after label maximization.

Then run interval DP.

---

# 20. Boundary Strategy in P0

G2 provides strong evidence that local alternatives matter for many labels:

* NAME: 89.0% within ±2;
* JOB_TITLE: 85.3%;
* USERNAME: 80.9%;
* EMAIL: 87.3%;
* PHONE: 83.3%;
* DATE: 71.2%.

Therefore local boundary alternatives are explicitly represented in hard-negative sampling.

But G2 is **not** evidence that a boundary-regression model will successfully learn the correct offset.

BOPN's primary contribution is to predict offsets from candidate spans toward nearby entity spans, providing a relevant future mechanism if direct span scoring still leaves concentrated local errors.

P0 does **not** implement BOPN.

ADDRESS is particularly important:

$$
29.1\%
$$

of its G2 boundary errors fall within ±2 tokens.

Therefore an ADDRESS-specific local correction is not justified by current diagnostics.

---

# 21. Future Boundary Experiment, Only If P0 Is Positive

If P0 succeeds but local boundary errors remain large, one follow-up may test either:

1. two-dimensional local boundary perturbations; or
2. an explicit offset-regression/BOPN-like auxiliary head.

Do not add both simultaneously.

Long aligned spans and the 187 unaligned training entities remain a different representational issue.

A token-boundary correction cannot solve character boundaries that the tokenizer cannot express.

---

# 22. Training Schedule

## Discovery run

One initial seed:

```text
2026
```

Train for exactly:

```text
3 epochs
```

Evaluate dev after:

```text
epoch 1
epoch 2
epoch 3
```

Select the best epoch by exact-slot accuracy.

No other hyperparameter sweep is permitted in this experiment.

Specifically frozen:

```text
max width        16
random negatives 256/document
MLP hidden       256
width embedding  32
dropout           0.20
learning rate     1e-3
weight decay      0.01
```

The three dev checkpoint evaluations must be reported as three development looks.

Dev is already repeatedly used and must never be described as blind validation.

---

# 23. Replication Budget

Replication happens **only if the discovery run is research-positive**.

Then freeze the selected epoch count and repeat with:

```text
seed 3407
seed 1337
```

No new tuning.

Do not ensemble these models during the frozen-head experiment.

An ensemble would answer a different question:

> Does an ensemble of span heads outperform the current ensemble?

The primary research question is instead:

> Is the span head intrinsically better than BIO given the same encoder?

---

# 24. Required Comparisons

## Comparison A — representation/head question

Compare:

$$
SpanHead_{epoch4}
$$

against:

$$
BIO_{epoch4}=79.4225\%.
$$

This is the primary research comparison.

The encoder checkpoint is identical.

Caveat:

the encoder was originally trained under BIO supervision. Therefore even this comparison is not perfectly symmetric: the frozen representation itself is BIO-optimized.

A span-head failure may therefore reflect representation mismatch rather than an inherent weakness of direct span classification.

---

## Comparison B — promotion question

Compare the span probe against:

$$
Ensemble_{4/8}=80.4057\%.
$$

This answers whether the single frozen span system is strong enough to replace the current champion.

No head/encoder equivalence exists here because the champion averages two checkpoints.

---

# 25. Evaluation Metrics

Primary:

$$
\boxed{\text{exact-slot accuracy}}
$$

Secondary diagnostics:

```text
unordered exact entity F1
boundary-only F1
exact-document rate
per-label exact-slot
per-label entity F1
correct_slot
exact_span_wrong_slot
overlap_same_label
overlap_wrong_label
same_boundary_wrong_label
no_overlapping_prediction
```

Also report:

```text
representable-gold rate under width 16
unaligned gold count
long-span gold count
```

These diagnostic misses remain in the metric denominator.

---

# 26. Research-Positive Gate

The initial probe is **research-positive** when:

$$
\Delta_{\text{Span vs BIO4}}
\ge
+0.5
\text{ percentage points}
$$

and the error decomposition is compatible with the hypothesis.

The `+0.5 pp` value is an **engineering decision threshold**, not a statistical significance claim.

Compatible error movement means the report should show whether gains came from:

* more exact entities;
* fewer local boundary errors;
* fewer composition/rank failures;

rather than an unexplained metric fluctuation.

Do not require every diagnostic category to improve.

---

# 27. Promotion-Positive Gate

The probe is **promotion-positive** when:

$$
ExactSlot_{\text{span}}
>
80.4057\%
$$

and every output contract passes:

```text
exact K
valid labels
start < end
offsets in bounds
non-overlap
correct sorting
complete dev coverage
```

Promotion does not imply the 88% milestone or 92% stretch target has been reached.

---

# 28. Paired Uncertainty Estimate

Use paired document-level bootstrap.

For document \(d\), retain:

```text
K_d
correct_span_d
correct_baseline_d
```

Draw:

```text
B = 10,000
```

bootstrap samples of the 6,943 dev documents with replacement.

For bootstrap sample \(b\):

$$
Acc^{span}_b
=
\frac{\sum_d correct^{span}_d}
{\sum_d K_d}
$$

and analogously for baseline.

Then:

$$
\Delta_b
=
Acc^{span}_b-Acc^{baseline}_b.
$$

Report:

```text
observed delta
bootstrap median delta
2.5th percentile
97.5th percentile
```

Do this separately for:

1. span vs BIO epoch 4;
2. span vs ensemble champion.

The resampling must be paired: the same sampled documents are used for both systems.

### Interpretation

This interval measures document-level sampling variability **within the repeatedly used dev set**.

It does not repair:

* repeated-development selection bias;
* template dependence;
* lack of a blind holdout evaluation.

Therefore:

> `CI > 0` means the difference is stable under resampling this dev corpus, not that it is proven to generalize to Kaggle test.

---

# 29. Stability Evidence

Engineering gate and stability evidence are separate.

### Engineering threshold

$$
\Delta\ge+0.5pp
$$

against BIO4.

### Stability evidence

After replication, report:

```text
seed 2026 delta
seed 3407 delta
seed 1337 delta
mean
median
range
paired bootstrap CI for each seed
```

A stronger result is one where the direction of improvement is consistent across all three seeds.

No seed ensemble is used for this conclusion.

---

# 30. Compute and Memory Estimate

The following numbers are planning estimates, not measured runtime.

Current frozen BiLSTM output:

$$
256
$$

dimensions/token.

Span feature:

$$
1056
$$

dimensions.

| Component                         | Planning estimate | Assumption                                                                         |
| --------------------------------- | ----------------: | ---------------------------------------------------------------------------------- |
| Span-head trainable parameters    |             ~275k | 1056→256→8, LayerNorm, width embedding                                             |
| Parameter + Adam states           |      < ~5 MB FP32 | head only                                                                          |
| Mean-span prefix state            |      ~0.25 MB/doc | example T=256, 256 dims FP32                                                       |
| Base sampled spans/doc            |              ~367 | 12.3246 gold average, max 8 local negatives/gold, 256 random; before old FPs/dedup |
| Sampled feature tensor, batch 32  |       ~24 MB FP16 | ~367 spans/doc, 1056 dims                                                          |
| All spans/doc at T=256            |             3,976 | Wmax=16                                                                            |
| Full candidate features, batch 32 |      ~256 MB FP16 | if all candidate features were materialized together                               |
| Recommended scoring chunk         |      32,768 spans | inference                                                                          |
| Feature memory/chunk              |       ~66 MB FP16 | 32,768×1056                                                                        |
| MLP hidden/chunk                  |       ~16 MB FP16 | 32,768×256                                                                         |

Thus 16 GB GPU memory is not expected to be the limiting factor for the span head.

The stronger reason to sample during training is **compute efficiency and hard-negative concentration**, not VRAM survival.

---

# 31. Hidden-State Cache Size

Do not invent a fixed cache size before measuring the actual token count.

For \(N_{tok}\) cached tokens:

$$
Storage_{hidden}
\approx
N_{tok}\times256\times2
\text{ bytes}
$$

for FP16 hidden states, plus token-offset metadata.

The existing neural training script records `train_tokens` in its final report construction.

Before implementation, retrieve exactly:

```text
output/bitrase-1.algo-6/v1/report.json
    ["train_tokens"]
```

and separately compute dev token count using the unchanged tokenizer.

Those two statistics are sufficient for an exact cache-size estimate.

---

# 32. Pseudocode — Candidate Sampling

```text
for document in TRAIN:

    tokens = tokenize(document.text)
    U = all token spans with width 1..16

    positives = {}
    protected = {}

    for gold in document.gold:

        if token_aligned(gold) and token_width(gold) <= 16:
            positives[gold.span] = gold.label
        else:
            protected.add(gold)

    # Remove candidates whose training target would be ambiguous
    eligible_negative =
        U
        - positives.keys()
        - spans_overlapping(protected)

    near = set()

    for gold_span in positives:

        i, j = gold_span

        for delta in [-2, -1, +1, +2]:

            candidate1 = (i + delta, j)
            candidate2 = (i, j + delta)

            for candidate in [candidate1, candidate2]:

                if valid_width_1_to_16(candidate)
                   and inside_document(candidate)
                   and candidate not in positives
                   and not overlaps_protected(candidate):

                    near.add(candidate)

    old_fp = set()

    for old_prediction in epoch4_train_exactK_predictions[document]:

        span = token_span(old_prediction)

        if span in positives:
            continue

        if span in eligible_negative:
            old_fp.add(span)

    forced_negative = near union old_fp

    remaining =
        eligible_negative
        - forced_negative

    m = min(256, len(remaining))

    random_negative =
        uniform_sample_without_replacement(remaining, m)

    sample =
        positives
        union forced_negative
        union random_negative

    for candidate in sample:

        if candidate in positives:
            target = positives[candidate]
            q = 1

        else if candidate in forced_negative:
            target = NONE
            q = 1

        else:
            target = NONE
            q = m / len(remaining)

        emit(candidate, target, q, source_tags)
```

---

# 33. Pseudocode — Loss

```text
numerator   = 0
denominator = 0

for document in batch:

    for sampled_candidate in document.samples:

        logits = span_head(span_representation(candidate))

        ce = cross_entropy(
            logits,
            candidate.target
        )

        numerator += ce / candidate.inclusion_probability

    denominator += document.training_candidate_universe_size

loss = numerator / denominator
```

No class-frequency weighting.

No focal factor.

No ranking term.

---

# 34. Pseudocode — Inference

```text
hidden = frozen_encoder(document)

prefix = prefix_sum_fp32(hidden)

candidates = []

for start in range(T):

    for width in 1..16:

        end = start + width - 1

        if end >= T:
            break

        mean = (
            prefix[end + 1] - prefix[start]
        ) / width

        feature = concat(
            hidden[start],
            hidden[end],
            hidden[start] * hidden[end],
            mean,
            width_embedding[width]
        )

        logits = span_head(feature)

        best_label = argmax(logits[ENTITY_LABELS])

        score = (
            logits[best_label]
            - logits[NONE]
        )

        candidates.append(
            start,
            end,
            best_label,
            score
        )

selected = exact_K_interval_DP(
    candidates,
    K=document.expected_entity_count
)

predictions = [
    (
        tokens[s.start].char_start,
        tokens[s.end].char_end,
        s.label
    )
    for s in selected
]

sort(predictions, by=(start, end))
```

---

# 35. Pseudocode — Exact-K Interval DP

```text
sort candidates by:
    end_token,
    start_token

for i in candidates:

    p[i] =
        last candidate j < i
        where end[j] < start[i]

DP[0][0] = 0

for k > 0:
    DP[0][k] = -INF

for i in 1..M:

    DP[i][0] = 0

    for k in 1..K:

        skip =
            DP[i - 1][k]

        take =
            DP[p[i]][k - 1]
            + score[i]

        if take > skip:
            DP[i][k] = take
            back[i][k] = TAKE
        else:
            DP[i][k] = skip
            back[i][k] = SKIP

if DP[M][K] == -INF:
    HARD FAIL

return backtrack(DP, back, M, K)
```

---

# 36. If the Probe Fails

A failure does not identify a single cause.

Possible explanations include:

### Frozen representation mismatch

The epoch-4 encoder was trained jointly with BIO supervision.

Its hidden states may therefore encode information in a form that works well for token classification but is suboptimal for direct span scoring.

### Head limitation

The one-hidden-layer MLP may not model endpoint interaction strongly enough.

### Sampling/loss issue

Hard-negative sampling, inverse-probability weighting, or extreme `NONE` imbalance may distort optimization.

### Representational coverage

Approximately 0.0314% of train gold is outside the chosen token-span universe, and unaligned spans cannot be predicted at all.

This ceiling is small globally but could concentrate in particular labels.

### Composition/decoder error

Correct local span scores do not guarantee that additive exact-K interval decoding selects the globally correct composition.

---

# 37. Maximum Two Cheap Checks After Failure

If P0 is not research-positive, perform at most two follow-ups before deciding whether frozen span scoring deserves further work.

## Cheap check A — Sampling sanity

On a fixed, train-only subset:

* enumerate **all** width-\(\le16\) candidates;
* train/evaluate the same frozen span head using full candidate CE;
* compare against the sampled/IPW version on a fixed train-internal evaluation subset.

Purpose:

> determine whether the sampling/loss estimator is masking a useful span representation.

No dev-based hyperparameter selection is allowed in this sanity check.

## Cheap check B — Biaffine endpoint head

Keep:

* exact same frozen hidden states;
* exact same candidate universe;
* exact same negative protocol;
* exact same decoder.

Replace only the MLP span interaction with a simple biaffine start/end scorer.

Yu et al. demonstrate that biaffine scoring can directly model start/end token pairs for NER.

Purpose:

> test whether the first MLP head is too weak, not whether a larger encoder is needed.

No joint encoder training follows automatically.

---

# 38. What Failure Can Legitimately Conclude

If P0 and both cheap checks fail:

Supported conclusion:

> The existing BIO-trained frozen encoder does not provide compelling evidence that direct span rescoring is superior under the tested heads and training objectives.

Unsupported conclusions:

> Span-based NER is bad for this dataset.

> Joint span training would fail.

> BOPN would fail.

> Semi-Markov models would fail.

> A different tokenizer could not improve representability.

Those questions were not tested.

---

# 39. Source/Evidence Separation

## Primary literature findings

**Sohrab & Miwa 2018:** exhaustive NER can enumerate spans and represent them from shared BiLSTM outputs without external knowledge resources.

**Yu et al. 2020:** start/end pairs can be scored directly with biaffine modeling for NER.

**Zaratiana et al. 2022:** global span selection can optimize/select a segmentation with structured dynamic programming.

**Tang et al. 2023:** boundary-offset prediction explicitly connects candidate spans to nearby entity boundaries and addresses limitations of treating all non-entity spans identically.

**Horvitz & Thompson 1952:** inverse first-order inclusion probabilities can yield an unbiased finite-population total estimator under unequal-probability sampling.

---

## Reasoning specific to this protocol

The following are **our design decisions**, not claims made by those papers:

* width 16;
* use of epoch-4 hidden states;
* `[start; end; product; mean; width]`;
* 256-dimensional MLP;
* eight local perturbations per gold;
* 256 random negatives/document;
* `logit(label)-logit(NONE)`;
* exact-K interval recurrence;
* +0.5 pp research gate;
* three-epoch discovery budget;
* paired document bootstrap;
* two-follow-up failure budget.

---

## Dataset-specific hypotheses

G2 motivates the hypothesis that local boundary alternatives are valuable hard negatives.

G3 motivates the hypothesis that better composition may recover multiple slot positions in some documents.

Neither diagnosis proves that the proposed direct-span classifier will improve exact-slot accuracy.

Only the frozen probe can answer that.

---

# 40. Final Protocol Decision

The approved P0 experiment is therefore:

$$
\boxed{
\text{Epoch-4 frozen contextual states}
\rightarrow
\text{direct span representation}
\rightarrow
\text{8-class MLP}
\rightarrow
\text{exact-K interval DP}
}
$$

with:

$$
W_{\max}=16
$$

and training candidates consisting of:

$$
\boxed{
\text{all representable gold}
+
\text{local boundary negatives}
+
\text{epoch-4 train false positives}
+
\text{256 random negatives/doc}
}
$$

under explicit inclusion probabilities.

The experiment changes **one conceptual component**:

$$
\text{BIO token scoring}
\rightarrow
\text{direct span scoring}.
$$

It does not change:

* encoder representation;
* tokenizer;
* training corpus;
* expected K;
* competition offset convention;
* pretrained/external-data policy.

This makes it sufficiently isolated to answer the research question.

**88% remains a milestone and 92% a stretch target. Neither is implied by this protocol, and no expected score gain should be stated before the experiment is executed.**
