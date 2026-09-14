# bitrase-1.algo-3 — Technicalities

## Komponen

- [sequence_crf.py](../../src/sequence_crf.py): tokenization, fitur, gold BIO conversion, reconstruction spans, export CRF weights, exactly-K Viterbi.
- [run_sequence_crf.py](../../src/run_sequence_crf.py): train sampling, training, raw/count-constrained dev evaluation dan provenance.
- [predict_sequence_crf.py](../../src/predict_sequence_crf.py): inference CSV dari checkpoint terverifikasi.
- [test_sequence_crf.py](../../tests/test_sequence_crf.py): Unicode/alignment tests dan exhaustive verification terhadap constrained paths pada 30 kasus kecil.
- [verify_dev_predictions.py](../../src/verify_dev_predictions.py): independent metric recomputation dari saved predictions tanpa import model.

CRFsuite dibaca melalui [python-crfsuite API](https://python-crfsuite.readthedocs.io/en/latest/pycrfsuite.html), versi package 0.9.12. Runtime lain: Python 3.13.2, PyArrow 25.0.1, NumPy 2.5.3.

## Count-constrained Viterbi

Emission dan transition weights diambil dari checkpoint lokal. State menyimpan skor maksimum pada posisi token t, jumlah entity starts k, dan label BIO terakhir l. Transisi ke B-label menambah k; transisi ke O/I tidak. Illegal BIO transitions diberi skor negatif tak hingga. Array backpointers memulihkan path terbaik dengan k=K.

Kompleksitas O(T K L²) untuk T tokens dan L BIO labels, dengan backpointer memory O(T K L). Untuk dokumen raw Viterbi yang sudah legal dan memiliki K, decode ulang tidak diperlukan. Exported weights dari model diuji pada dua toy sequences terhadap output native CRFsuite; production predictions divalidasi terpisah.

Panjang label strings dan token offsets tidak dinormalisasi. CRFsuite feature names dibangun sebagai daftar strings berbobot 1, sehingga exported state weights bisa dijumlahkan langsung menjadi emission.

## Run awal

```powershell
uv run src/run_sequence_crf.py
```

Output default `output/bitrase-1.algo-3/crf12k`. Script menolak direktori yang sudah ada. Scaling, jika dilakukan, menggunakan output baru dan explicit `--train-docs` / `--iterations`; bukan overwrite run awal.

Setelah checkpoint dan hasil dev diverifikasi serta metode dipilih:

```powershell
uv run src/predict_sequence_crf.py --run output/bitrase-1.algo-3/crf12k
uv run src/verify_submission.py --submission output/bitrase-1.algo-3/crf12k/submission.csv --report output/bitrase-1.algo-3/crf12k/submission_validation.json
```

## Provenance dan batasan

Config menyimpan sumber/data hashes dan versions. `train_ids.json` merekam subset train; `model.crfsuite` memuat bobot baru dari nol; `report.json` menyimpan hasil dan hashes model/predictions. `raw_dev_predictions.parquet` dan `known_k_dev_predictions.parquet` menyimpan hasil yang bisa dihitung ulang.

Hash checks sebelum/sesudah run menolak perubahan sumber. Source/run baseline bitrase-1 tidak dimodifikasi. File inference memverifikasi checkpoint dan sumber sebelum memprediksi test, kemudian menghasilkan CSV/manifest baru. Reproduksi byte-identical training belum diuji.

Dev telah digunakan untuk pemilihan metode, sehingga evaluasi bukan final holdout. Token boundaries membatasi beberapa entitas yang tidak align; CRF berpotensi gagal mengenali format baru atau anotasi tak konsisten. Known-K performance tidak sama dengan unknown-count NER.
