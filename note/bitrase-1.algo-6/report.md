# bitrase-1.algo-6 — Hasil neural dari nol

Byte-CNN + BiLSTM yang dilatih pada seluruh 55.582 dokumen train mencapai **80,4057% exact-slot dev** dengan average logits checkpoint4/8. Checkpoint tunggal terbaik adalah epoch4 pada79,4225%; epoch8 menghasilkan79,4085%. Target88% belum tercapai; angka entity F1 bukan pengganti exact-slot.

| Metode | Exact-slot | Entity F1 | Slot benar |
|---|---:|---:|---:|
| Epoch4, exact-K | 79,4225% | 91,5581% | 67.776/85.336 |
| Epoch8, exact-K | 79,4085% | 91,4948% | 67.764/85.336 |
| Epoch4 + text-only honorific adjustment | 78,8811% | 90,9147% | 67.314/85.336 |
| Average logits epoch4/8, exact-K | **80,4057%** | **91,8428%** | **68.615/85.336** |

Bobot ensemble0,5/0,5 ditetapkan sebelum hasil epoch8 diperiksa, tanpa weight search. Ensemble menambah839 slot benar dibanding epoch4 dan7.640 dibanding baseline; sebanyak3.423 dokumen diprediksi sempurna. Validator independen pada `ensemble_4_8/independent_check.json` lolos. Ensemble dipilih untuk pembuatan submission manual.

## Submission ensemble

File `output/bitrase-1.algo-6/ensemble_4_8/submission.csv` sudah dibuat dan diverifikasi:285.318rows,23.156dokumen, IDs/order sama dengan sample, seluruh count/bounds/labels/non-overlap valid, errors kosong, ready=true. SHA256:`5bd13ceb311db06cc779ce574c7be5b0bde99ff9b1e61c034df9e0430d5a6206`. Belum diunggah; Vian melakukan submission sendiri. Manifest mencatat kedua checkpoint dan test/slots/source hashes.

Perbandingan: baseline bitrase-1 dev71,4528%; CRF+style77,6366%. Neural epoch4 menambah 6.801 slot benar dari baseline atau 7,9697 percentage points. Skor Kaggle0,705 tetap hanya hasil baseline yang dilaporkan Vian; hasil neural belum dinilai Kaggle.

## Temuan

Pada epoch4 terdapat 10.356 entitas tepat yang salah posisi slot, 4.083 overlap dengan label sama (terutama masalah boundary), dan443 kasus confusion label. NAME F1 88,63%; USERNAME F1 84,44% lebih lemah dibanding EMAIL98,53% danPHONE98,17%. Exact-K memperbaiki jumlah prediksi, tetapi tidak otomatis memilih entitas yang benar pada setiap urutan.

Training loss turun dari0,1040 ke0,03958, sementara exact-slot dev tidak meningkat antaraepoch4 dan8. Memperpanjang jadwal yang sama belum didukung hasil. Honorific adjustment yang membantu CRF malah menurunkan neural sehingga ditolak untuk neural. Uji lanjutan terpisah: average logits checkpoint4/8 dan skor transisi yang dilatih pada emisi neural frozen.

## Validasi dan batas kesimpulan

Kedua checkpoint serta varian style dihitung ulang dari saved Parquet dengan validator independen. Seluruh6.943 dokumen/85.336 slot tercakup, label/bounds/non-overlap/urutan valid, hitungan sesuai report. Empat unit tests neural lulus. Runtime training+evaluasi sekitar896,8detik padaRTX5060Ti16GB; sumber/data/checkpoint/vocabulary diidentifikasi SHA256.

Tidak memakai pretrained, external data, holdout, atau test labels. Sebanyak187 entitas train tidak representable pada token boundary; seluruh41 entitas dev yang demikian tetap masuk evaluasi. Dev telah dipakai untuk memilih metode sehingga ini hasil development selection. Belum ada bukti88% maupun skor Kaggle baru.

Artifacts: `output/bitrase-1.algo-6/v1/report.json`, `epoch4_independent_check.json`, `epoch8_independent_check.json`, serta `output/bitrase-1.algo-6/style_epoch4/independent_check.json`.
