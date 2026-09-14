# bitrase-1.algo-5 document honorific style

Fixed TF-IDF word 1-2 gram max_features=150000 min_df=2 and SGD logistic classifier, random_state=2026. Fit uses only prepared v1 train documents with at least two classified honorific observations; ties resolve to included. Dev style accuracy is descriptive. The sole adjustment uses predicted probability >= 0.5 for included and applies only to NAME spans in CRF known_k predictions. No holdout/test, refit, threshold sweep, or gold-based adjustment. Mixed documents remain a caveat.
