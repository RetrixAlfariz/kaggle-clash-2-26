# M1 — model tanpa pretrained

Konfigurasi ditetapkan sebelum dev scoring, 15 September 2026 WIB.

- Logistic classifier biner dilatih dari nol dengan SGD dan fitur sparse hashed, bukan pretrained transformer/embedding/LLM.
- Train hanya prepared v1 train. Kandidat dari frozen M0 dictionary + regex, ditambah satu regex context USERNAME yang tetap. Dictionary M0 juga dibuat hanya dari train.
- Target training: kandidat exact `(start,end,label)` gold = 1; kandidat lain = 0. Seluruh kandidat dipakai, tanpa negative subsampling.
- Fitur: score M0, label, frasa, bentuk karakter, jumlah kata, posisi, konteks token kiri/kanan, batas identifier dan honorific. Bobot lexical/context dipelajari hanya dari competition train; tidak ada external dataset.
- Dua epoch, seed 2026, batch 256 dokumen, 2^20 hash features, logistic loss, L2 alpha 1e-6, constant learning rate 0,02, averaged weights. Shuffle dokumen deterministik per epoch. Tidak melakukan sweep hyperparameter atau early stopping berdasarkan dev.
- Evaluasi sekali pada dev menggunakan protokol exact slot M0 dan known-count non-overlapping decoding. Input tetap character spans, sehingga tidak ada neural-tokenizer alignment atau BIO conversion.
- Bandingkan exact slot dev terhadap M0. Upload satu baseline yang lebih baik dan memiliki submission valid; jika M1 menurun, M0 tetap menjadi baseline upload. Jangan menggunakan public score untuk memilih hyperparameter dalam run ini.
- Tidak membuka reserved holdout, menambah data luar, atau refit dengan dev sebelum submission ini.

Skor dev adalah development score. Fitur phrase-global prior dihitung pada train yang sama, sehingga training performance bukan estimasi generalisasi; keputusan perbandingan menggunakan dev. Kemiripan template tetap caveat.
