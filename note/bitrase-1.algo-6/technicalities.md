# bitrase-1.algo-6 — Technicalities

## Files

- [neural_sequence.py](../../src/neural_sequence.py): vocabulary, compact byte/token arrays, flags, masked batch collation, model architecture, decode interface.
- [run_neural_sequence.py](../../src/run_neural_sequence.py): GPU fitting, bucketing, epoch4/8 evaluation, checkpoint selection/provenance.
- [predict_neural_sequence.py](../../src/predict_neural_sequence.py): hash-verified test inference and CSV generation.
- [test_neural_sequence.py](../../tests/test_neural_sequence.py): Unicode/cap, train vocab/UNK, padmask/ignore targets, padding-invariant valid logits.
- [sequence_crf.py](../../src/sequence_crf.py): shared exact-K BIO Viterbi, independently tested against exhaustive small examples.

## Local runtime

Environment `.venv-neural` is separate from frozen preparation/project dependencies. Installed with uv: Python3.13.2, PyTorch2.8.0+cu128, NumPy2.5.3, PyArrow25.0.1. Torch came from the [official CUDA128 package index](https://download.pytorch.org/whl/cu128), with installation version supported by the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/).

GPU: RTX5060Ti16GB. CUDA matrix operation smoke test succeeded before training. Dependency installation installs software, not pretrained model weights.

```powershell
.venv-neural/Scripts/python.exe -m unittest discover -s tests -p test_neural_sequence.py
.venv-neural/Scripts/python.exe src/run_neural_sequence.py
# Only after completed run / accepted checkpoint:
.venv-neural/Scripts/python.exe src/predict_neural_sequence.py
uv run src/verify_submission.py --submission output/bitrase-1.algo-6/v1/submission.csv --report output/bitrase-1.algo-6/v1/submission_validation.json
```

Commands describe first run; training refuses existing output and inference refuses CSV overwrite. New runs must use a new `--output` and preserve predecessor artifacts. CUDA reproducibility is not claimed byte-identical; hashes identify actual selected artifacts.

## Validation and provenance

Four tests pass, including checking that batch padding does not alter logits on valid tokens. Root and independent source review confirmed that dev labels are not passed into model/decoder, only evaluation. Source/data hashes are captured before training and checked again after completion.

Config contains runtime/device and training parameters. `vocab.json` is train-only; `epoch4.pt`/`epoch8.pt` are state_dict checkpoints. Each snapshot has raw/known-K Parquet predictions and a report with model/prediction hashes. Final report selects one of the two checkpoints. Candidate recall from dictionary baselines is not reported for this sequence model.

Inference uses `torch.load(..., weights_only=True)` after model/vocab/source hash checks, follows the same AMP and exact-K path as evaluation, and writes a new manifest. Final CSV is independently checked against original sample submission and prepared test text.

## Ensemble follow-up

`src/evaluate_neural_ensemble.py` averages epoch4/8 logits equally and uses the same decoder. The config binds the parent report and evaluation source hashes. This achieved80.4057% exact-slot and passed independent verification. On Windows all JSON reads explicitly specify UTF-8, including Unicode vocabulary.

```powershell
.venv-neural/Scripts/python.exe src/evaluate_neural_ensemble.py
.venv-neural/Scripts/python.exe src/predict_neural_sequence.py --ensemble
uv run src/verify_submission.py --submission output/bitrase-1.algo-6/ensemble_4_8/submission.csv --report output/bitrase-1.algo-6/ensemble_4_8/submission_validation.json
```

These commands have completed. Outputs are preserved and reruns refuse overwrite. Ensemble inference took133.1seconds excluding initial test encoding. Submission285,318rows passed validation, no upload performed.
