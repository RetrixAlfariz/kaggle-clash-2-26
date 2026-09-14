# bitrase-1 — Baseline report

**Kode baseline: `bitrase-1`.** Ditetapkan pada 15 September 2026 untuk Kaggle Clash 2 — IRIS 2026. Baseline ini merujuk tepat pada model M1 run `output/m1/v1`; pemberian kode tidak mengubah model, data, konfigurasi, atau predictions.

Dokumen pendamping: [spec](spec.md) menjelaskan kontrak baseline; [technicalities](technicalities.md) menjelaskan implementasi, provenance, dan cara menjalankan ulang.

Eksperimen turunan: [bitrase-1.algo-1](../bitrase-1.algo-1/report.md) menguji perubahan decoder dengan classifier dan kandidat tetap; exact-slot dev 72,3024%. Artefak baseline bitrase-1 tetap dipertahankan.

## Status

Training, dev evaluation, independent metric verification, dan pembuatan CSV selesai. Vian melaporkan hasil submission Kaggle **0,705** pada percakapan 15 September 2026. Sumber skor adalah laporan Vian; receipt, submission ID, dan angka dengan presisi lebih tinggi belum diperiksa langsung oleh Opheline. Upload dilakukan manual oleh Vian.

Model adalah logistic classifier kandidat span yang dilatih dari nol. Tidak menggunakan pretrained weights, pretrained embeddings, transformer, inference LLM, atau external dataset. Tujuannya memberi baseline yang dapat diukur sebelum eksperimen berikutnya.

## Hasil

Training memakai 55.582 dokumen prepared v1 selama dua epoch. Setiap epoch memuat 3.824.302 kandidat, termasuk 681.100 positif. Training beserta dev evaluation memerlukan sekitar 216 detik pada run ini; ini bukan benchmark hardware.

Dev: 6.943 dokumen, 85.336 entitas. M0 memakai dictionary/regex; bitrase-1 menambahkan contextual scoring dan kandidat USERNAME berbasis konteks.

| Metrik dev | M0 | bitrase-1 | Selisih |
|---|---:|---:|---:|
| Exact-slot accuracy | 64,0023% | **71,4528%** | +7,4505 pp |
| Unordered exact entity F1 | 85,8220% | **89,2144%** | +3,3925 pp |
| Candidate recall | 94,7033% | **95,8915%** | +1,1882 pp |
| Dokumen seluruh entitas dan slot benar | 1.672 | **2.397** | +725 |

Sebanyak 60.975 slot benar; 76.132 entitas benar jika posisi slot diabaikan. Seluruh dokumen memenuhi jumlah entitas K. **Exact-slot menjadi metrik utama pemilihan baseline**, karena entitas benar pada slot yang salah tidak mendapat kredit. Entity F1 89,21% bukan skor leaderboard.

Hasil ini cocok dengan [independent dev check](../../output/m1/v1/independent_dev_check.json), yang menghitung ulang metrik dari predictions tersimpan dan gold dev. Tidak ada error coverage, urutan, offset, overlap, atau jumlah. Tiga unit tests M1 juga lulus pada pengerjaan awal.

## Submission

**Skor Kaggle yang dilaporkan: 0,705 (70,5%).** Dibanding exact-slot dev 0,714528, selisihnya sekitar **−0,009528 / −0,9528 poin persentase**. Selisih dihitung dari presisi skor yang diberikan Vian. Kedekatan ini menjadi indikasi awal bahwa performa dev cukup sejalan dengan hasil Kaggle; satu submission belum membuktikan kestabilan generalisasi atau menjelaskan penyebab selisih. Skor ini tidak digunakan untuk mengubah konfigurasi bitrase-1.

File final: [submission.csv](../../output/m1/v1/submission.csv).

[Validasi independen CSV](../../output/m1/v1/submission_validation.json) lulus untuk 285.318 baris / 23.156 dokumen: header, row IDs dan urutan sample submission, label, offsets, jumlah slot, dan non-overlap. Tidak ada placeholder span.

Deskripsi upload yang disarankan:

```text
bitrase-1 | scratch logistic contextual span classifier | train-only dictionary + regex | no pretrained | dev slot accuracy 0.714528
```

## Pertimbangan dan rekomendasi

**Pembaruan setelah audit oracle:** [batas exact-slot candidate pool](slot-oracle.md) mencapai **95,7369%**, dibanding aktual **71,4528%**. Prioritas berikutnya adalah diagnosis pemilihan kandidat/scoring/decoder, sebelum perluasan USERNAME. Daftar awal berikut tetap menjadi inventaris masalah, bukan urutan implementasi yang sudah dipastikan.

1. **Pertahankan run ini sebagai pembanding.** Model dipilih dari perbandingan M0/M1 pada dev; tidak ada sweep atau refit dengan dev. Eksperimen berikutnya perlu kode/output baru agar bukti baseline tetap dapat dirujuk.
2. **Audit candidate coverage.** Sebanyak 3.506 gold spans tidak tersedia dalam pool; scoring yang lebih baik tidak dapat memilih kandidat yang tidak ada. Fokus pada proposal USERNAME dan batas span, lalu ukur dampak lintas label.
3. **Audit scoring dan decoding.** Sebanyak 5.698 gold spans ada dalam pool tetapi tidak dipilih. Ada pula 15.157 entitas tepat dengan slot bergeser. Missing/extra spans dan boundary errors perlu dipisahkan sebelum mengganti model.
4. **Lakukan ablation.** Peningkatan sekarang mencakup classifier dan tambahan regex USERNAME sekaligus. Belum ada bukti terpisah tentang kontribusi masing-masing.
5. **Batasi klaim generalisasi.** Dev adalah development split; template dapat mirip antarsplit. Reserved holdout belum digunakan. Prior dictionary dihitung pada train yang juga dipakai classifier, sehingga training performance tidak menjadi estimasi generalisasi.

Sumber angka: [run report](../../output/m1/v1/report.json), [hasil M0](../M0_BASELINE_RESULTS.md), dan independent check di atas. Riwayat awal run: [M1_RESULTS](../M1_RESULTS.md). Status bitrase-1 dan dokumentasi baseline selanjutnya dirujuk dari folder ini.
