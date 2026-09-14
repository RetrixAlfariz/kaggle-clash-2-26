# bitrase-1.algo-1 — Hasil perubahan decoder

**Keputusan dev: pilih Slot MBR.** Exact-slot meningkat dari **71,4528% menjadi 72,3024%**, selisih **+0,8496 poin persentase / +725 slot benar**. Ini improvement development yang terukur; belum merupakan skor Kaggle baru atau pengganti baseline utama bitrase-1.

Parent: bitrase-1 / M1 v1. Kandidat, fitur, classifier weights, prepared data, dan jumlah slot K tetap sama. Tidak retraining atau memakai pretrained. Perubahan hanya pada pemilihan kombinasi span. [Protokol](protocol.md) ditetapkan sebelum scoring kedua challenger.

## Hasil perbandingan

Dev yang sama: 6.943 dokumen, 85.336 gold slots.

| Metode | Exact-slot | Unordered entity F1 | Dokumen sempurna |
|---|---:|---:|---:|
| Parent: jumlah probabilitas kandidat | 71,4528% | 89,2144% | 2.397 |
| MAP: kombinasi paling mungkin | 71,4575% | 89,2214% | 2.398 |
| **Slot MBR: peluang kandidat pada posisi slot** | **72,3024%** | **88,9320%** | **2.283** |

MAP hanya menambah 4 slot benar dan mengubah 14 dokumen. Mengganti jumlah probabilitas menjadi jumlah log-odds hampir tidak mengubah keputusan parent pada dev ini.

Slot MBR menambah 725 slot benar bersih: 692 dokumen membaik, 471 memburuk, 5.780 memiliki jumlah slot benar yang sama. Kenaikan per dokumen berjumlah 2.219 slot, penurunan berjumlah 1.494 slot. Prediksi berubah pada 1.386 dokumen; sebagian perubahan tidak mengubah total slot benar.

### Trade-off

Entity F1 turun **0,2824 pp**, jumlah entitas benar turun 241, dan dokumen sempurna turun 114. Improvement bukan dominasi semua metrik. Slot MBR dipilih karena protokol kompetisi menilai exact-slot; jika kebutuhan berubah menjadi unordered NER atau seluruh dokumen harus sempurna, parent bisa lebih sesuai.

Di 4.572 dokumen yang seluruh gold tersedia dalam candidate pool, slot benar bertambah **373**. Di 2.371 dokumen dengan kandidat gold hilang, slot benar bertambah **352**. Candidate recall tetap **95,8915%**, karena tidak ada kandidat baru.

| Label gold | Exact-slot parent | Exact-slot Slot MBR |
|---|---:|---:|
| NAME | 69,2260% | 69,9066% |
| DATE | 75,1339% | 75,9222% |
| EMAIL | 75,3883% | 76,5710% |
| PHONE | 73,0860% | 74,4516% |
| ADDRESS | 66,6910% | 67,4082% |
| USERNAME | 59,7226% | 60,2853% |
| JOB_TITLE | 80,4197% | 81,3468% |

Metrik per label menghitung slot menurut label gold, sehingga perubahan posisi dapat memengaruhi semua kategori. Peningkatan ini tidak berarti kemampuan mengenali tiap kategori secara unordered semuanya meningkat.

## Rasionalisasi dan batasan

Parent memaksimalkan penjumlahan probabilitas kandidat tanpa reward khusus untuk posisi slot. Challenger Slot MBR membentuk distribusi kombinasi span valid dan menghitung peluang sebuah kandidat berada pada slot tertentu. Decoder kemudian memilih kombinasi valid dengan expected jumlah slot benar tertinggi di bawah distribusi tersebut.

Model distribusi ini mengasumsikan independent Bernoulli candidate probabilities sebelum dikondisikan pada tepat K/non-overlap. Probabilitas classifier belum dikalibrasi secara terpisah dan kandidat dapat saling bergantung. Karena itu optimalitas matematis pada distribusi asumsi tidak menjamin optimalitas terhadap data nyata.

Oracle dev sebelumnya 95,7369%; hasil baru 72,3024%, masih tertinggal **23,4344 pp**. Hasil ini membuktikan manfaat terbatas dari perubahan keputusan, bukan bahwa seluruh gap dapat diselesaikan decoder. Diagnosis berikutnya perlu memeriksa ranking/kalibrasi skor kandidat pada train sebelum perubahan baru; belum dilakukan pada eksperimen ini.

Dev telah dipakai untuk analisis dan memilih decoder. Tidak ada sweep, holdout evaluation, atau klaim signifikansi independen. Kemiripan template tetap caveat. Skor Kaggle parent 0,705 berasal dari laporan Vian; eksperimen baru belum diupload.

## Validasi dan artefak

- Prediksi parent direplay dari checkpoint dan cocok persis dengan saved predictions pada semua dev documents.
- Tiga tests decoder lulus, mencakup exhaustive verification untuk 180 kasus kecil acak deterministik, marginals, keputusan MAP/MBR, extreme probabilities, touching intervals, K tidak feasible, dan input tidak sah.
- Forward/backward partition cocok; probabilitas setiap slot berjumlah satu. Hasil dev memenuhi K, kandidat asal, bounds, labels, sorting, dan non-overlap.
- Source review independen tidak menemukan bug actionable pada DP atau tests.

Independent recomputation dari kedua saved dev prediction files cocok persis dengan report: coverage 6.943 dokumen, 85.336 entitas, exact-slot, entity F1, dan boundary F1. Bukti: [independent_check.json](../../output/bitrase-1.algo-1/v1/independent_check.json).

Hasil mesin: [report.json](../../output/bitrase-1.algo-1/v1/report.json), [paired documents](../../output/bitrase-1.algo-1/v1/paired_documents.json). [Spec](spec.md) dan [technicalities](technicalities.md) menjelaskan kontrak dan implementasi.

## Submission manual

File terpilih: [submission.csv](../../output/bitrase-1.algo-1/v1/submission.csv). **Belum diupload oleh Opheline; belum ada skor Kaggle untuk eksperimen ini.**

[Validasi CSV](../../output/bitrase-1.algo-1/v1/submission_validation.json) lulus seluruh 285.318 baris / 23.156 dokumen, ID dan order sample submission, labels, offsets, K, sorting, dan non-overlap. [Manifest](../../output/bitrase-1.algo-1/v1/submission_manifest.json) merekam decoder Slot MBR dan parent model hash yang tidak berubah.

SHA-256 CSV: `702c009f73c2411c2c891cf3ce06380454ee1379bbee3d48cee357b16ee2bb3e`.

Deskripsi upload yang disarankan:

```text
bitrase-1.algo-1 | slot-MBR decoder | frozen bitrase-1 classifier/candidates | no pretrained | dev exact-slot 0.723024
```
