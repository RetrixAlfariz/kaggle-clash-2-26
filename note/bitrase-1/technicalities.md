# bitrase-1 — Technicalities

## Peta implementasi

| Komponen | Implementasi |
|---|---|
| Loader dan schema/hash guards | [prepared_loader.py](../../src/prepared_loader.py) |
| Train dictionary/candidate rules M0 | [run_m0.py](../../src/run_m0.py) |
| Trie matching dan interval decoder | [baseline_probe.py](../../src/baseline_probe.py): `matches`, `decode` |
| Training dan inference bitrase-1 | [train_m1.py](../../src/train_m1.py): `propose`, `features`, `score`, `train`, `submit_file` |
| Dictionary loading / CSV helpers | [submit_m0.py](../../src/submit_m0.py) |
| Dev metrics | [ner_evaluation.py](../../src/ner_evaluation.py) |
| Full CSV verification | [verify_submission.py](../../src/verify_submission.py) |
| Unit tests M1 | [test_m1.py](../../tests/test_m1.py) |

Nama file tetap M1 agar source hashes frozen run tetap cocok. `bitrase-1` adalah kode baseline untuk run tersebut.

## Candidate generation dan features

Dictionary M0 memuat 71.494 frasa train dengan label dan occurrence prior. Prior dihitung menggunakan smoothing `(positive + 1) / (occurrences + 2)`; tidak dihitung ulang pada dev/test. Kandidat format berasal dari regex EMAIL, PHONE, DATE.

Tambahan USERNAME mencari konteks `username`, `user name`, `user id`, atau `login id`, diikuti separator opsional dan identifier ASCII 3–64 karakter. Trailing punctuation dipangkas; stopwords tetap difilter. Proposal tambahan menggunakan prior 0,5 bila triple belum ada. Pola persis tersedia dalam `USERNAME_CONTEXT` dan `STOP_USER` pada script.

Fitur diberi prefix label dan di-hash menjadi sparse matrix:

- Prior logit dibagi 4; panjang span dibatasi 150 lalu dibagi 30; jumlah kata dibatasi 15 lalu dibagi 3; posisi relatif; jumlah newline dibatasi 3.
- Flag title, whitespace, digit, underscore, `@`, casing, serta identifier characters di kiri/kanan.
- Frasa lowercase, kata pertama/terakhir, karakter tetangga, dan honorific kiri.
- Lima token kiri dari jendela 100 karakter dan empat token kanan dari jendela 70 karakter, termasuk bigram terdekat.

Semua lexical weights dipelajari dari train. Hash collision mungkin terjadi; baseline tidak memakai vocabulary embedding. Score adalah `predict_proba(... )[:, 1]`; tidak ada calibration stage terpisah.

## Training dan decoder

Train documents diacak deterministik dengan seed `2026 + epoch_index`; estimator juga memakai random state 2026. Setiap batch ditransform ke sparse features dan diproses dengan `partial_fit`, classes `[0, 1]`. Semua kandidat positif/negatif dilibatkan.

Known-K decoder memakai weighted interval dynamic programming. Kandidat diurutkan menurut `(end, start, label)`; predecessor adalah kandidat terakhir yang kompatibel (`end <= start`). State menyimpan skor terbaik untuk prefix kandidat dan jumlah pilihan k. Transisi take menambahkan **probabilitas**, bukan logit, ke skor predecessor untuk k−1. Ties mempertahankan solusi sebelumnya. Hasil akhir disortir menurut posisi.

Decoder umum dapat mengembalikan jumlah feasible terbesar di bawah K. `submit_file` memeriksa panjang hasil terhadap K dan menghentikan run jika kurang. Pada baseline ini semua dokumen memenuhi K. Tidak ada optimisasi end-to-end langsung atas exact-slot loss: classifier belajar candidate correctness, decoder mengoptimalkan jumlah skor kandidat.

## Runtime dan reproduksi

Run menggunakan Python 3.13.2, PyArrow 25.0.1, scikit-learn 1.9.1, NumPy 2.5.3. [Script lockfile](../../src/train_m1.py.lock) mengunci environment training melalui uv, terpisah dari dependency preparation.

Jalankan dari root repository. Gunakan direktori baru yang belum ada:

```powershell
uv run --locked src/train_m1.py --output output/bitrase-1/reproduction-01
uv run --locked src/train_m1.py --output output/bitrase-1/reproduction-01 --submission-only
uv run src/verify_submission.py --submission output/bitrase-1/reproduction-01/submission.csv --report output/bitrase-1/reproduction-01/submission_validation.json
```

Run ini masih bergantung pada prepared v1 dan frozen M0 dictionary yang sudah tersedia; command di atas tidak membangun raw data/dictionary. Source changes yang terdeteksi selama training atau sebelum inference akan menyebabkan hash guard gagal. Training menolak output directory yang sudah ada; inference menolak overwrite CSV.

M1 belum dilatih ulang untuk membuktikan byte-identical reproduction. Seed dan lockfile mendukung reproduksi; klaim yang sudah terbukti adalah validasi run dan metrik independen. Checkpoint berupa pickle lokal, dimuat setelah hash cocok dengan report run; gunakan hanya checkpoint yang provenance-nya dipercaya.

## Artefak frozen dan provenance

Root artefak: `output/m1/v1`.

| Artefak | Fungsi |
|---|---|
| [config.json](../../output/m1/v1/config.json) | Hyperparameters, runtime, source/data hashes |
| [model.pkl](../../output/m1/v1/model.pkl) | Classifier dan hasher terlatih |
| [report.json](../../output/m1/v1/report.json) | Training counts, dev metrics, model/prediction hashes |
| [dev_predictions.parquet](../../output/m1/v1/dev_predictions.parquet) | Saved spans per dev document |
| [independent_dev_check.json](../../output/m1/v1/independent_dev_check.json) | Recomputed dev metrics |
| [submission.csv](../../output/m1/v1/submission.csv) | File yang diupload manual |
| [submission_manifest.json](../../output/m1/v1/submission_manifest.json) | CSV/model/test/slot hashes |
| [submission_validation.json](../../output/m1/v1/submission_validation.json) | Independent full CSV checks |

SHA-256 identitas baseline:

```text
model.pkl
3b744c3c8bad6070f5831aa30f808891dbbf8c9f5a9f69200979feae39f09d61
dev_predictions.parquet
80c5be8193523cd4b926577a2ab523605db165c426baf8582be40a4e937f2a3c
submission.csv
795acaf78cf60a2a91549afe5d10917c03a5c7dd436d1e7eeaa955b5fdb0b8e8
M0 dictionary.parquet
67f6d511d062669410e5a2890c17abce02c39292fad73853af2490bf9c3e8954
```

Tidak ada pemindahan atau salinan model/data besar ke folder note. Ketiga dokumen ini merujuk artefak asli agar baseline memiliki satu sumber hasil.
