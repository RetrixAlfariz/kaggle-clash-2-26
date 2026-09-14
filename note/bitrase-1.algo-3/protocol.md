# bitrase-1.algo-3 — Sequence CRF dari nol

Uji keluarga metode alternatif: linear-chain CRF dengan BIO labels langsung dari word/punctuation tokens. Fitur lexical, shape, prefix/suffix, dua token konteks kiri/kanan, dan whitespace/newline gaps. Tidak ada pretrained atau external corpus. Ini membandingkan paket metode/representasi baru, bukan ablation satu variabel terhadap classifier kandidat.

Train awal: 12.000 dokumen dipilih shuffle seed 2026 hanya dari prepared v1 train. CRFsuite L-BFGS, c1=.1, c2=.1, max_iterations=60, feature.minfreq=2, all transition features. Jika hasil membutuhkan peningkatan dan resource memungkinkan, scaling jumlah train/iterations dicatat sebagai run terpisah sebelum evaluasi berikutnya. Tidak membuka reserved holdout.

Tokenization tetap regex `\w+|[^\w\s]`. Audit seluruh train menemukan 187/685.026 entitas tidak align dengan batas token; entitas tersebut tidak dapat direpresentasikan sempurna. Jangan menggunakan gold untuk mengubah tokenizer per dokumen. Unaligned train entities tidak diberi BIO target; semua dev gold tetap dihitung dalam denominator.

Bandingkan raw Viterbi dengan CRF path maximum score yang memenuhi legal BIO dan tepat exposed K entity starts. Gold tidak menjadi input decoding. Positional exact-slot adalah metric utama; laporkan unordered F1 dan count correctness. Dev dipakai untuk pemilihan eksperimen, bukan klaim unbiased final score.

Model membaca urutan teks untuk menentukan spans sehingga tidak memiliki candidate recall yang dapat dibandingkan langsung dengan dictionary pool parent. Candidate-specific diagnostics dari generic evaluator tidak dilaporkan.
