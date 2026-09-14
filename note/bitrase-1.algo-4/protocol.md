# bitrase-1.algo-4 — CRF dengan fitur batas kandidat kamus train

Follow-up setelah CRF12k menghasilkan 76,9968% exact-slot dev. Hipotesis: context sequence features saja kurang memanfaatkan informasi lexical/boundary yang sudah ada pada train dictionary.

Gabungkan fitur CRF sebelumnya dengan indikator B/I/end per label dari frozen M0 candidates, serta bin prior confidence. Dictionary hanya dari prepared train; tidak memperluas dengan dev/test. Kandidat yang tidak align token dilewati sebagai fitur. CRF tetap dapat menghasilkan spans di luar dictionary melalui sequence prediction.

Run: 24.000 dokumen shuffle seed2026; CRF L-BFGS c1=.1,c2=.1,max_iterations80,minfreq2, possible transitions. Evaluasi raw dan exact-K pada dev penuh. Ini paket perubahan fitur + scaling data/iterations, bukan ablation yang membuktikan kontribusi tiap perubahan.

Tidak memakai pretrained, external corpus, gold sebagai feature saat inference, atau reserved holdout. Dev sudah dipakai untuk memilih arah eksperimen; semua hasil dinyatakan development scores. Target tetap88% exact-slot dev sebelum validasi Kaggle manual.
