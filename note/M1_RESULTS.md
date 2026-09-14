# M1: baseline tanpa pretrained

Run ini ditetapkan sebagai baseline **bitrase-1**. Dokumentasi baseline: [report](bitrase-1/report.md), [spec](bitrase-1/spec.md), dan [technicalities](bitrase-1/technicalities.md). Dokumen ini mempertahankan catatan awal M1.

Tanggal run: 15 September 2026. **M1 dipilih sebagai file submission untuk diupload manual oleh Vian. Tidak ada submission yang dikirim Opheline.**

## Model dan alasan

M1 adalah binary logistic classifier, dilatih dari nol dengan SGD untuk menilai apakah kandidat `(start, end, label)` merupakan entitas yang benar. Kandidat berasal dari dictionary M0 yang dibangun hanya dari train, regex format, dan satu tambahan regex konteks USERNAME. Classifier belajar dari frasa, konteks kiri/kanan, bentuk karakter, honorific, dan batas identifier. Decoder memilih tepat K span tanpa overlap menggunakan jumlah slot yang memang disediakan kompetisi.

Pendekatan ini menargetkan masalah M0: frasa yang sama bisa salah dalam konteks tertentu, potongan nama/username bersaing dengan span lengkap, dan satu kesalahan span dapat menggeser slot berikutnya. Model memakai character offsets langsung sehingga tidak perlu alignment tokenizer.

Tidak ada pretrained weights, embeddings, transformer, LLM inference, external dataset, atau external gazetteer. Bobot lexical dipelajari hanya dari train kompetisi. Weak labels dan reserved holdout tidak digunakan.

## Training dan hasil dev

Konfigurasi dibekukan sebelum scoring dalam [M1_PROTOCOL.md](M1_PROTOCOL.md). Training menggunakan 55.582 dokumen, dua epoch, masing-masing 3.824.302 kandidat dengan 681.100 positif. Waktu training beserta evaluasi sekitar 216 detik. Dev berisi 6.943 dokumen dan 85.336 entitas. Tidak ada hyperparameter sweep atau refit dengan dev.

| Metrik dev | M0 | M1 | Perubahan |
|---|---:|---:|---:|
| Exact-slot accuracy | 64,0023% | **71,4528%** | +7,4505 pp |
| Unordered exact entity F1 | 85,8220% | **89,2144%** | +3,3925 pp |
| Candidate recall | 94,7033% | **95,8915%** | +1,1882 pp |
| Dokumen seluruh span/slot benar | 1.672 | **2.397** | +725 |

M1 mendapat 60.975 slot benar dan 76.132 entitas benar tanpa memperhitungkan slot. Seluruh dokumen memenuhi K. Exact-slot menjadi dasar pemilihan karena entitas yang ditempatkan pada slot yang salah tidak mendapat kredit pada kompetisi. Angka 89,21% bukan perkiraan langsung skor leaderboard.

Independent check menghitung ulang positional slot accuracy dan unordered entity F1 dari saved predictions terhadap dev. Keduanya cocok persis dengan report; tidak ada kesalahan coverage, jumlah, urutan, offset, atau overlap. Tiga unit tests untuk USERNAME offsets, boundary/context features, dan classifier probabilities lulus.

## Batas interpretasi dan langkah berikutnya

- Peningkatan mencakup classifier **dan** tambahan kandidat USERNAME; belum ada ablation yang memisahkan kontribusinya.
- Masih ada 3.506 gold spans di luar candidate pool. Classifier tidak dapat memilih span yang tidak diusulkan. Candidate recall 95,89% adalah batas recall entitas untuk pool ini, bukan batas exact-slot yang otomatis dapat dicapai.
- Sebanyak 5.698 gold spans tersedia sebagai kandidat tetapi tidak dipilih. Perlu diagnosis scoring/decoding sebelum memperbesar model.
- Sebanyak 15.157 entitas terdeteksi tepat tetapi bergeser slot. Prioritaskan sumber missing/extra span dan batas NAME/USERNAME/ADDRESS karena efeknya merambat ke slot berikutnya.
- Dictionary dan prior dibuat dari train yang sama dengan classifier; training fit bukan bukti generalisasi. Dev belum merupakan final holdout dan kemiripan template tetap dapat membuat hasil tampak lebih baik.
- Jangan ubah banyak hal berdasarkan satu public score. Simpan baseline ini, lalu lakukan eksperimen kandidat/boundary dan ablation dengan protokol dev yang sama.

## Artefak dan reproduksi

- Model dan konfigurasi: `output/m1/v1/model.pkl`, `config.json`.
- Evaluasi: `output/m1/v1/report.json`, `dev_predictions.parquet`, `independent_dev_check.json`.
- Submission manual: `output/m1/v1/submission.csv`.
- Validasi submission: `output/m1/v1/submission_manifest.json`, `submission_validation.json`.
- Script: `src/train_m1.py` dan lockfile script untuk environment terpisah dari preparation.

CSV final lolos validasi seluruh **285.318 baris / 23.156 dokumen**, header, row IDs dan urutan persis sample submission, label, batas karakter, jumlah slot, serta non-overlap. Tidak ada placeholder span. SHA-256 CSV: `795acaf78cf60a2a91549afe5d10917c03a5c7dd436d1e7eeaa955b5fdb0b8e8`.

Untuk upload manual, pilih `output/m1/v1/submission.csv` pada Submit Prediction di kompetisi `kaggle-clash-2-iris-2026`. Deskripsi yang bisa dipakai: `M1 scratch logistic contextual span classifier; train-only dictionary + regex; no pretrained; dev slot accuracy 0.714528; prepared v1`. Public score belum tersedia karena belum diupload. Form browser ditutup setelah Vian memilih untuk upload sendiri.

```powershell
# Untuk run baru; output yang sudah ada tidak ditimpa.
uv run --locked src/train_m1.py --output output/m1/reproduction
uv run --locked src/train_m1.py --output output/m1/reproduction --submission-only
uv run src/verify_submission.py --submission output/m1/reproduction/submission.csv --report output/m1/reproduction/submission_validation.json
```

Run baru belum dilakukan untuk membuktikan byte-identical reproduction M1. Source, model, prediction, dan CSV hashes tersedia pada artefak run. Model tidak diubah setelah dev scoring.
