# bitrase-1.algo-6 — Neural sequence specification

Model sequence BIO dari nol: learned random word embeddings + byte-character CNN + 2-layer bidirectional LSTM. Tidak ada pretrained weights, pretrained tokenizer, pretrained embeddings, atau external data.

## Data dan representasi

Semua55.582 prepared train documents dipakai. Lowercase word vocabulary dibangun hanya dari train, maksimum50.000 entries termasuk PAD/UNK. Tokenization tetap word/punctuation regex. Original character offsets tidak diubah.

Setiap token diberi byte UTF-8 maksimum32 bytes serta tujuh flags: uppercase, digit, titlecase, newline kiri/kanan, joined token kiri/kanan. Unknown word menjadi UNK tetapi karakter aslinya tetap tersedia. Unicode byte prefix bisa terpotong pada32 bytes; byte CNN memproses bytes, bukan mendekode prefix itu kembali menjadi string.

Gold BIO dibuat hanya pada batas token yang representable. Unaligned train entities tidak diberi target parsial; jumlahnya dicatat. Semua dev gold tetap dievaluasi. Dev tidak membentuk vocabulary atau gradient updates.

## Arsitektur dan training

- Word embedding128dim, random initialization.
- Byte embedding16dim, Conv1D kernel3/channels64, ReLU/maxpool.
- BiLSTM2layers, hidden128 per arah, dropout.2 antarlayer.
- Linear head15 BIO logits untuk tujuh labels dan O.
- AdamW lr.001, weight_decay.01, gradient clip1.0,8epochs, seed2026,batch32.
- Padded targets=-100 diabaikan crossentropy. Packed sequences memakai panjang asli.
- CUDA float16 autocast dengan GradScaler. Mini-batches dikelompokkan menurut panjang dalam shuffled blocks untuk mengurangi padding.

## Evaluasi dan inference

Snapshot dev hanya pada epochs4 dan8, mencakup raw argmax dan legal BIO exact-K. K berasal dari exposed slot count, bukan gold span positions/labels. Decoder memaksimalkan penjumlahan logits dengan constraint BIO/K dan zero transition weights; bukan CRF layer yang dilatih.

Checkpoint dipilih menurut known-K exact-slot dev. Ini development selection. Reserved holdout tidak dibuka dan Kaggle target belum terbukti sampai submission dinilai.

CSV manual harus lulus semua row IDs/order, count, label, bounds, sorting, non-overlap checks. Tidak ada placeholder span atau upload otomatis.
