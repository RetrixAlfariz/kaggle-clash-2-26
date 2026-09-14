# bitrase-1.algo-7 — Technicalities

`src/run_neural_transition.py` verifies the frozen parent checkpoint/vocabulary/source hashes, caches neural emissions in length-sorted GPU batches, fits only225 transition parameters on CPU, then evaluates every dev document with legal BIO exact-K decoding. Source/protocol/train/dev hashes are captured before fitting and rechecked at completion. Parent neural parameters receive no gradients.

The likelihood sums all legal BIO paths with logsumexp; initial I states are impossible, and no learned start/end scores are introduced. Padded timesteps preserve the forward state and do not contribute to the gold score. Adam lr0.03, batch32,3epochs, clip5; each batch performs one update. The deterministic4,000-document train subset uses ascending SHA1 of document ID. Unaligned training gold is counted under the parent's tokenization convention; every dev gold entity remains evaluated.

Three tests in `tests/test_neural_transition.py` passed: legality/finite gradients, initial-state restriction, and exhaustive three-token partition compared with both single-sequence and padded-batch loss. CUDA inference uses the already installed `.venv-neural` runtime; transitions train on CPU with one thread.

```powershell
.venv-neural/Scripts/python.exe src/run_neural_transition.py
uv run src/verify_dev_predictions.py --predictions output/bitrase-1.algo-7/v1/dev_predictions.parquet --report output/bitrase-1.algo-7/v1/report.json --method known_k --output output/bitrase-1.algo-7/v1/independent_check.json
```

The output directory is immutable after creation. `transitions.npy` is a numeric NumPy matrix; illegal transitions are masked again by the shared decoder. Reports identify parent epoch, selected train count, unaligned entities, losses, timing, and artifact hashes. This transition-only fit is not joint end-to-end BiLSTM-CRF training; a failure does not establish that joint training cannot help.
