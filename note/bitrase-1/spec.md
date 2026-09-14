# bitrase-1 — Specification

## Identitas dan ruang lingkup

| Item | Ketetapan |
|---|---|
| Kode baseline | `bitrase-1` |
| Implementasi/run asal | M1 / `output/m1/v1` |
| Task | Exact character-span NER dengan slot yang jumlahnya diketahui |
| Metode | Train-only candidate generation, logistic contextual scoring, fixed-K interval decoding |
| Pretrained / external data | Tidak digunakan |
| Metrik utama | Exact-slot accuracy pada dev |
| Submission | CSV manual; file lokal sudah tervalidasi |

Baseline mendefinisikan model dan hasil yang sudah dihasilkan, bukan konfigurasi baru untuk ditrain. [Report](report.md) memuat hasil; [technicalities](technicalities.md) memuat implementasi.

## Data contract

- Gunakan prepared v1 melalui `PreparedData`, yang memeriksa schema, jumlah baris, dan hash terhadap manifest.
- Train: 55.582 dokumen untuk dictionary dan classifier fitting. Dev: 6.943 dokumen untuk evaluasi dan pemilihan antara M0/M1.
- Test: 23.156 dokumen untuk inference, dengan 285.318 entity slots dari sample submission.
- Reserved holdout tidak dipakai. Weak pool dan metadata audit tidak menjadi input fitting/scoring model ini.
- Input dokumen adalah `full_text` hasil rekonstruksi segmen sesuai `segment_index`, digabung dengan newline. Tidak mengubah teks atau menghitung offset pada teks yang dinormalisasi.
- Offset adalah indeks karakter Python Unicode, 0-based, end-exclusive: `full_text[start:end]`.
- Label yang sah: `NAME`, `DATE`, `EMAIL`, `PHONE`, `ADDRESS`, `USERNAME`, `JOB_TITLE`.

## Model contract

1. Usulkan kandidat `(start, end, label)` dari frozen M0 dictionary/regex dan fixed USERNAME context regex.
2. Pada train, label target kandidat adalah 1 hanya jika triple persis ada dalam gold; kandidat lain diberi 0. Semua kandidat dipakai, tanpa negative subsampling.
3. Encode fitur lexical, konteks, bentuk span, batas identifier, honorific, posisi, dan prior dengan feature hashing. Normalisasi fitur tidak mengubah offset sumber.
4. Fit binary logistic classifier dari nol. Tidak menggunakan pretrained tokenizer/model/embedding atau layanan inference eksternal.
5. Score kandidat dengan probabilitas kelas positif. Pilih span tanpa overlap, tepat K sesuai exposed slot count, dengan memaksimalkan jumlah probabilitas.
6. Sort hasil berdasarkan start lalu end; map ke slot dokumen yang bersesuaian.

## Frozen training configuration

| Parameter | Nilai |
|---|---|
| Epochs / seed | 2 / 2026 |
| Batch | 256 dokumen |
| FeatureHasher | 2^20 fitur, input dict, alternate sign |
| Classifier | SGDClassifier, logistic loss, L2 |
| Alpha | 0,000001 |
| Learning rate | Constant, eta0 = 0,02 |
| Averaged weights / shuffle | True / True |
| Tuning | Tidak ada sweep atau early stopping berdasarkan dev |

Parameter otoritatif dan source hashes: [config.json](../../output/m1/v1/config.json). Protokol pra-evaluasi: [M1_PROTOCOL.md](../M1_PROTOCOL.md).

## Output dan acceptance criteria

- CSV harus memiliki tepat kolom `row_id,Predicted`; seluruh row IDs dan urutan persis sample submission.
- `Predicted` berformat `LABEL:start:end`, dengan label sah dan `0 <= start < end <= len(full_text)`.
- Setiap dokumen memiliki K span, tersortir dan tanpa overlap. Tidak mengisi kekurangan kandidat dengan placeholder.
- Jika decoder tidak dapat mencapai K, generator submission harus gagal sebelum menulis CSV final.
- Independent dev metrics harus cocok dengan report; validasi submission harus menunjukkan `ready: true` dan daftar error kosong.
- Upload manual dan public score merupakan tahap terpisah dari keberhasilan validasi lokal.

## Versioning

Kode bitrase-1 menunjuk frozen M1 v1 beserta hashes di technicalities. Perubahan training data, candidate rules, features, hyperparameters, decoder, atau learned weights harus diberi identitas eksperimen/run baru. Catatan public score dapat ditambahkan dengan tanggal dan receipt tanpa mengganti hasil dev yang sudah tercatat.
