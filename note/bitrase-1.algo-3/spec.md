# bitrase-1.algo-3 — Specification

Keluarga metode: sequence labelling dengan linear-chain Conditional Random Field (CRF), dilatih dari nol. Eksperimen mengganti classifier kandidat dengan model urutan BIO; fitur dan representasi token ikut berubah. Karena itu hasilnya adalah perbandingan metode lengkap, bukan bukti kontribusi satu komponen tunggal.

Data sumber tetap prepared v1. Run awal `crf12k` menggunakan 12.000 dokumen dari train setelah shuffle seed 2026. Dev penuh digunakan untuk evaluasi, reserved holdout tidak dibuka. Tidak memakai weak labels, data luar, pretrained weights, pretrained tokenizer, atau external inference.

## Representasi

Token adalah kata/angka/underscore atau satu karakter punctuation menurut regex `\w+|[^\w\s]`. Span menggunakan start/end indeks karakter pada teks asli. Whitespace tidak menjadi token, tetapi gap dan newline menjadi fitur; whitespace di dalam entitas dipertahankan saat span direkonstruksi.

Setiap token diberi O, B-label, atau I-label untuk tujuh label kompetisi. Gold entity yang tidak tepat align ke batas token dilewati untuk target training dan dihitung sebagai keterbatasan. Semua gold dev tetap dievaluasi tanpa penghapusan.

## Model dan keputusan

Fitur mencakup lowercase token, character shape, panjang, casing/digit, prefix/suffix, dua token kiri/kanan, serta joined/whitespace/newline boundaries. Seluruh bobot dipelajari dari train.

Training memakai L-BFGS CRFsuite: L1 c1=.1, L2 c2=.1, max_iterations=60, minimum feature frequency 2, possible transition features. Tidak melakukan early stopping berdasarkan dev.

Raw prediction memakai Viterbi dari CRFsuite. Prediction untuk submission memakai maximum-score BIO path dengan tepat K B-tags, dengan K dari exposed entity slots. I-label hanya boleh mengikuti B/I dari label yang sama. Jika raw path sudah valid dan memiliki K entitas, path itu langsung memenuhi constrained optimum.

Tidak ada gold dalam input decoder. Jika constraint tidak feasible, gagal; tidak membuat placeholder spans.

## Acceptance

Exact-slot dev adalah metrik utama. Laporkan pula unordered entity F1, count correctness, dan perfect-document count. Karena metode ini tidak menggunakan candidate pool dictionary, candidate recall parent tidak digunakan sebagai metrik CRF.

CSV akhir wajib memiliki 285.318 baris, header `row_id,Predicted`, row IDs dan urutan sample submission, label sah, offsets valid, K terpenuhi, dan tanpa overlap. Public score hanya dicatat setelah hasil upload manual Vian tersedia. Kode ini tidak otomatis menjadi baseline utama bitrase-2.
