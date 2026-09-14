# bitrase-1.algo-7 — Transition-only CRF

Skor exact-slot dev menjadi **80,0670%**, naik0,6445 percentage points dari frozen neural epoch4 (79,4225%). Entity F1 menjadi92,0081%. Hasil ini belum mengungguli ensemble checkpoint4/8 pada80,4057%, sehingga submission terbaik tetap ensemble algo6. Target88% belum tercapai.

Matriks transisi dilatih pada4.000 train documents,3epochs, tanpa mengubah neural weights. Ada9 entitas train yang tidak representable pada batas token; konsisten dengan parent, jumlahnya dicatat. Training negative log likelihood per document turun6,9485 →6,7185 →6,6911. Dev hanya dievaluasi setelah fit selesai.

Seluruh6.943 dokumen dev dan85.336 gold slots dievaluasi. Saved predictions lolos validator independen; report dan matrix/prediction hashes tersimpan di `output/bitrase-1.algo-7/v1`. Tidak ada pretrained, external data, reserved holdout, test label, atau submission otomatis.

Kesimpulan: skor transisi memberi manfaat kecil, tetapi penambahan transisi saja tidak menjembatani gap menuju88%. Ini tidak menguji joint BiLSTM-CRF training; emisi masih frozen. Dokumen/spec implementasi ada di [protocol.md](protocol.md) dan [technicalities.md](technicalities.md).
