# bitrase-1.algo-1 — Specification

## Identitas

- Parent: `bitrase-1`, frozen `output/m1/v1`.
- Eksperimen: `bitrase-1.algo-1`, output `output/bitrase-1.algo-1/v1`.
- Perubahan: decoder saja. Metode terpilih berdasarkan dev: `slot_mbr`.
- Tidak mengubah prepared data, candidate generation, fitur, learned weights, atau baseline CSV.
- Tidak memakai pretrained, external corpus, reserved holdout, gold saat inference, atau refit dengan dev.

## Input/output

Mengikuti [data contract parent](../bitrase-1/spec.md): reconstructed full text, karakter Unicode 0-based end-exclusive, tujuh labels, exposed K. Classifier parent menyediakan probabilitas correctness untuk setiap candidate triple `(start,end,label)`.

Decoder menerima candidate probabilities dan K saja, tanpa gold. Probabilitas harus finite dalam [0,1]; span harus panjang positif. Untuk pembentukan distribusi, p diklip pada [1e-6, 1-1e-6].

Hasil wajib tepat K, non-overlapping, sorted by position, dan seluruh span berasal dari input candidate pool. Tidak ada penambahan/fallback placeholder jika tidak feasible; proses gagal.

CSV memiliki kolom `row_id,Predicted`, row IDs/order persis sample submission, serta value `LABEL:start:end`. Upload dilakukan manual oleh Vian setelah local validation.

## Keputusan decoder

Distribusi kombinasi valid S diberi bobot sebanding dengan `exp(sum(logit(p_i), i in S))`, dengan constraint `|S|=K` dan non-overlap.

- Pembanding MAP memilih kombinasi dengan bobot tertinggi.
- Slot MBR menghitung `P(candidate i occupies slot k)` untuk setiap candidate/slot, lalu memilih kombinasi valid yang memaksimalkan jumlah marginal pada posisi masing-masing.

Slot MBR meminimalkan expected positional slot error **di bawah distribusi asumsi tersebut**. Ini tidak menggunakan oracle gold dan bukan jaminan terhadap distribusi gold sebenarnya.

## Acceptance dan versioning

Kriteria awal: exact-slot dev lebih tinggi daripada parent; semua output contracts valid. Entity F1/dokumen sempurna dilaporkan sebagai trade-off. Pemilihan antara dua challenger telah menggunakan dev dan tidak dianggap final unbiased evaluation.

Terpilih di dev bukan promosi otomatis menjadi bitrase-2. Public score baru dicatat setelah Vian melaporkan hasil upload. Perubahan algoritma berikutnya menggunakan kode eksperimen baru, bukan overwrite artefak ini.
