# bitrase-1.algo-1 — Protokol sebelum dev scoring

Parent: bitrase-1 / frozen M1 v1. Eksperimen hanya mengubah decoder; prepared data, candidate pool, features, classifier weights, dan exposed K sama. Tidak memakai pretrained, gold saat inference, reserved holdout, atau feedback Kaggle untuk tuning.

Hipotesis: penjumlahan probabilitas kandidat pada parent menargetkan jumlah entitas benar secara unordered. Scoring yang mempertimbangkan posisi kandidat dalam urutan dapat memperbaiki exact-slot.

Uji dua metode yang ditetapkan sebelum evaluasi:

1. **MAP:** maksimum jumlah log-odds pada kombinasi tepat K span non-overlapping. Ini memilih kombinasi paling mungkin dalam asumsi independent Bernoulli yang dikondisikan pada K/non-overlap.
2. **Slot MBR:** gunakan distribusi kombinasi yang sama untuk menghitung marginal peluang kandidat pada setiap posisi slot, lalu pilih kombinasi valid yang memaksimalkan jumlah marginal tersebut. Ini meminimalkan expected slot error di bawah distribusi asumsi tersebut.

Probabilitas diklip tetap pada [1e-6, 1-1e-6] untuk kestabilan numerik. Tidak ada temperature sweep atau fitting calibration. Independence/calibration merupakan asumsi, bukan fakta yang dibuktikan; kandidat overlap dan lexical dependencies dapat melanggarnya.

Reproduce prediction parent di dev, bandingkan kedua metode dengan exact-slot sebagai metrik utama, serta entity F1, dokumen sempurna, per-label, dan paired document differences. Pilih metode dengan exact-slot tertinggi; tie antar challenger memilih slot MBR. Jika tidak mengalahkan parent, eksperimen dicatat ditolak dan tidak dibuat submission baru. Bila menang, buat satu CSV dengan model parent yang sama untuk upload manual Vian. Public score belum menjadi bagian eksperimen ini.

Dev sudah dipakai untuk diagnosis sebelumnya dan sekarang untuk memilih decoder; hasil adalah development selection, bukan unbiased final evaluation. Tidak melakukan refit atau membuka holdout setelah pemilihan.
