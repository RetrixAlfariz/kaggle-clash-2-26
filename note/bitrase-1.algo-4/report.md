# CRF dictionary24k — hasil eksperimen internal algo-4

Paket eksperimen menambahkan fitur B/I/end dari frozen M0 candidate dictionary ke CRF, sekaligus scaling train12k→24k dan iterations60→80. Menurut konvensi pengguna, bagian fitur termasuk pembaruan representasi data; kode internal output algo-4 dipertahankan sebagai identitas run. Eksperimen ini tidak dipromosikan menjadi baseline/data revision yang diterima.

| Decoder | Exact-slot dev | Unordered entity F1 |
|---|---:|---:|
| Raw Viterbi | 52,9976% | 88,8420% |
| Legal BIO, tepat K | **71,5571%** | **88,8793%** |

Hasil lebih rendah daripada CRF12k76,9968%, sehingga **ditolak untuk pemilihan model**. Training sekitar654 detik, train+dev sekitar754 detik. Tambahan data/fitur tidak otomatis memperbaiki hasil.

[Independent check](../../output/bitrase-1.algo-4/crf24k/independent_check.json) merekonsiliasi61.064 slot benar /85.336 dan output contracts. Regresi bukan formatting error yang ditemukan oleh validator; penyebab modelnya belum diisolasi. Karena fitur, jumlah train, dan iterations berubah sekaligus, tidak menyimpulkan satu komponen tertentu sebagai penyebab.

[Protocol](protocol.md), [script](../../src/run_sequence_crf_dictionary.py), [feature module](../../src/sequence_crf_dictionary.py), [report](../../output/bitrase-1.algo-4/crf24k/report.json). Tidak ada pretrained, label edits, atau holdout access. Tidak menghasilkan submission baru dari run ini.
