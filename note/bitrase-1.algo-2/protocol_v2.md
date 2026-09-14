# Boosted candidates v2 — protokol koreksi sebelum scoring

V1 interrupted/aborted, tidak memiliki hasil sah. V2 adalah implementasi baru dengan bounded context tokenization sekali per dokumen, fitur categorical lexical/context yang diencode hanya pada train, dan probabilitas via predict_proba.

Gunakan frozen M1 candidate pool. Seluruh 55.582 train docs, seluruh kandidat positif, maksimum40 negatif per dokumen dipilih deterministik. Negatif diberi inverse inclusion probability weight (N_neg/40 bila N_neg>40); positive weight1. Tidak menggunakan probabilitas yang tidak dikoreksi akibat downsampling. Objective weighted loss mendekati distribusi semua kandidat, tanpa klaim calibration sempurna.

LightGBM500trees, learning_rate .06,63leaves,minchild40,colsample.9,lambda2,cat_smooth10,maxcatthreshold32,seed2026,njobs8. Tidak ada dev early stopping/sweep. Bandingkan sum-probability decoder dan Slot-MBR. Dev digunakan untuk pemilihan, bukan final holdout.

Fitur kategori: label, frasa, kata awal/akhir, dua token kiri/kanan, bigram, shape, honorific. Fitur numerik: prior logit, panjang/posisi, whitespace/newlines, casing/digits, identifier boundaries dan prefix. Tidak ada pretrained atau data luar. Report menyimpan model, candidate counts, metode terpilih, dan semua sumber hash.
