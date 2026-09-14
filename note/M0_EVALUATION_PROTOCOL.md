# Protokol evaluasi M0 — dibekukan sebelum run

Run: `m0-v1`, seed/tie rules deterministik, 15 September 2026 WIB.

## Populasi dan batas akses

Gunakan prepared v1: seluruh train untuk fitting kamus dan occurrence precision; seluruh dev untuk evaluasi dan diagnosis. Loader eksplisit `PreparedData.load_train()` dan `load_dev("evaluation")`. Tidak membaca holdout, competition test, weak pool, atau metadata audit. Tidak melakukan hyperparameter search pada run ini.

Kamus: exact phrase case-sensitive dari anotasi train, satu label per phrase berdasarkan observed matched positives (fallback frekuensi anotasi), tie frekuensi diselesaikan secara alfabetis. Confidence `(positive + 1)/(all matched occurrences + 2)` dihitung hanya pada train. Matching memakai batas alphanumeric seperti probe historis, tanpa mengubah teks/offset.

Fallback tetap: EMAIL regex confidence 0,85; PHONE parenthesized US format 0,8; DATE nama bulan lengkap atau ISO date 0,8. Tidak mengklaim regex ini mencakup semua format. Jika kandidat exact span/label sudah ditemukan kamus, score kamus dipertahankan. Regex dan decoder menggunakan implementasi probe historis yang dicatat hash-nya.

## Dua decoder, satu konfigurasi tetap

1. **Known count — utama:** weighted non-overlapping interval DP, target K dari `expected_entity_count`, memaksimalkan jumlah confidence. Jika K tidak feasible, ambil jumlah feasible terbesar; tidak membuat span palsu untuk mengisi slot.
2. **Unconstrained — diagnostik:** interval DP dengan log-odds score, tanpa K. Ini mengukur sensitivitas terhadap informasi count, bukan kandidat model lain yang dituning.

Known count sah untuk konteks kompetisi karena slot test mengungkap jumlah entitas. Pada dev, field ini berasal dari gold; jangan menyebut hasilnya unknown-count NER. Prediction menerima teks dan K saja; gold spans dipakai setelah prediction untuk scoring/diagnosis.

## Metrik

- **Utama: exact slot accuracy** = jumlah gold slots dengan `(start,end,label)` identik pada posisi sama / seluruh gold slots. Spans diurutkan `(start,end,label)`; pada non-overlapping output tidak ada tie posisi. Missing prediction pada posisi gold bernilai salah. Laporkan count mismatch dan excess predictions pada decoder unconstrained secara terpisah.
- **Entity micro precision/recall/F1:** intersection himpunan `(document_id,start,end,label)`, dibagi total predicted/gold. Ini diagnostik unordered, bukan pengganti slot metric.
- **Boundary-only micro P/R/F1:** intersection `(document_id,start,end)`, mengabaikan label, untuk memisahkan lokalisasi dari klasifikasi.
- Per label: gold/predicted counts, exact-span precision/recall/F1, slot accuracy dengan denominator gold label, candidate recall.
- Candidate recall: proporsi gold triples yang tersedia sebelum decoding. Exact-document rate dan count-correct-document rate memakai denominator semua dev documents.

Untuk precision/recall dengan denominator nol, laporkan nol dan raw counts. Tidak ada partial-overlap credit pada exact metrics. Satu kesalahan dapat menggeser rank banyak slot; karena itu laporkan `exact_span_wrong_slot` secara terpisah.

## Diagnosis, bukan causal attribution

Setiap gold entity masuk tepat satu kategori candidate-stage: `exact_candidate`, `same_boundary_wrong_label`, `overlap_same_label`, `overlap_wrong_label`, atau `no_overlapping_candidate`, dengan prioritas sesuai urutan itu.

Setiap gold entity masuk tepat satu kategori output-stage: `correct_slot`, `exact_span_wrong_slot`, `same_boundary_wrong_label`, `overlap_same_label`, `overlap_wrong_label`, atau `no_overlapping_prediction`. Exact candidate yang tidak dipilih decoder dihitung terpisah. Overlap hanya berarti interval beririsan dan dapat many-to-one; ini petunjuk diagnosis, bukan matching one-to-one untuk F1 atau bukti penyebab tunggal.

Simpan contoh dev terbatas per kategori beserta gold/prediction/candidates lokal. Pilih deterministik berdasarkan urutan document ID, bukan berdasarkan contoh yang mendukung kesimpulan. Contoh ini tidak mewakili prevalensi; prevalensi berasal dari full-dev counts.

## Reproducibility dan keputusan berikutnya

Simpan frozen config, hashes input/implementasi/protokol, dictionary hasil fitting, prediksi kedua decoder, metrics, dan error examples. Output existing ditolak agar run lama tidak ditimpa.

Sesudah run: tentukan apakah prioritas berikutnya candidate coverage, label/context disambiguation, boundary handling, atau decoding. Tidak otomatis mengubah regex, tuning confidence, membuka holdout, memilih tokenizer, atau melatih neural model pada tahap ini. Dev sudah merupakan development set dan hasil ini bukan estimasi fully blind atau generalisasi lintas template.
