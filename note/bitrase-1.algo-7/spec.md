# bitrase-1.algo-7 — Specification

Input: frozen algo6 epoch4 token logits and the same train-only vocabulary/token boundaries. Learned component:15x15 BIO transition matrix, optimized with legal-path CRF negative log likelihood. The neural encoder/head remain frozen. Output: non-overlapping character spans with seven competition labels, exactly the exposed K count, sorted by start/end.

Training uses4.000 deterministic prepared-train documents,3epochs,Adam lr0.03,batch32,gradient clip5. No pretrained, new dataset, dev-gradient fitting, holdout, or test data. Initial I labels are forbidden; no learned start/end scores. Illegal transitions are masked in training and decoding.

Selection criterion: exact-slot dev against the frozen neural parent and the existing ensemble. Validated result80,0670% improves the parent but loses to ensemble80,4057%; no replacement submission is generated. See [report.md](report.md), [protocol.md](protocol.md), and [technicalities.md](technicalities.md).
