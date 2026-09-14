# bitrase-1.algo-2 — protocol

Parent is `bitrase-1` (M1 classifier plus slot-MBR decoder). This experiment changes only candidate scoring: a LightGBM binary classifier is fit from scratch on v1 train candidates. The frozen M1 proposal function, exact train/dev split, labels, exposed K, and decoder remain fixed. No pretrained weights, external data, holdout, test data, or dev tuning are used.

To bound memory and class imbalance, all positive candidates and at most 30 deterministic negatives per training document are used. The cap is part of the protocol. Frequency features are computed from the training candidate pool only.

Primary metric is exact slot accuracy on dev; entity F1, candidate recall, and exact-document rate are diagnostic. This is a development experiment, not an unbiased final evaluation.
