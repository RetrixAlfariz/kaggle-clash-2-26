# bitrase-1 — Batas exact-slot dari kandidat yang sudah tersedia

## Kesimpulan

**Prioritaskan diagnosis pemilihan kandidat sebelum memperluas pencarian USERNAME.** Pada 6.943 dokumen dev, candidate pool bitrase-1 memungkinkan maksimum **95,7369% exact-slot**, sedangkan model menghasilkan **71,4528%**. Ada selisih **24,2840 poin persentase** yang secara matematis dapat diperbaiki dengan memilih kombinasi kandidat yang berbeda dari pool yang sama.

Ini merupakan oracle diagnostic dengan bantuan gold dev. Angka tersebut bukan skor model yang dapat dijalankan pada test, bukan target kenaikan yang dijanjikan, dan tidak digunakan untuk membuat submission.

## Pertanyaan dan metode

Pertanyaan: dari kandidat asli bitrase-1, berapa jumlah slot tepat maksimum jika pilihan harus berjumlah tepat K, non-overlapping, dan diurutkan berdasarkan posisi?

Candidate pool direkonstruksi dengan fungsi `propose` M1 dan frozen dictionary M0. Tidak menambahkan kandidat dari gold. Gold hanya menjadi fungsi penilaian untuk memilih kombinasi ideal. Jumlah K adalah jumlah gold dev yang juga dipakai protokol known-count baseline.

Dynamic programming menggunakan state `D[i,k]`: jumlah slot benar maksimum dari i kandidat pertama yang diurutkan menurut end, dengan tepat k kandidat terpilih. Untuk kandidat i, pilih nilai maksimum antara:

- Melewati kandidat: `D[i-1,k]`.
- Mengambil kandidat: `D[p(i),k-1] + 1(candidate_i == gold[k-1])`, dengan p(i) prefix kandidat yang berakhir paling lambat pada start kandidat i.

State yang tidak feasible tidak dapat digunakan. Hasil wajib state tepat K, tidak melonggarkan jumlah. Pada kumpulan span positif tanpa overlap, end order dan start order menghasilkan urutan yang sama. Reward membandingkan label, start, end, dan posisi slot sekaligus; memaksimalkan jumlah gold spans secara unordered saja tidak cukup.

## Hasil lengkap

| Ukuran | Nilai |
|---|---:|
| Dokumen / gold slots | 6.943 / 85.336 |
| Slot benar aktual | 60.975 / 71,4528% |
| Slot benar maksimum dalam pool | 81.698 / 95,7369% |
| Selisih pemilihan aktual terhadap oracle | **20.723 slot / 24,2840 pp** |
| Slot tetap salah bahkan pada oracle | **3.638 slot / 4,2631 pp** |
| Gold spans tidak tersedia | 3.506 |
| Penalti tambahan akibat constraint slot/K/non-overlap | 132 |
| Dokumen yang bisa diperbaiki dalam pool yang sama | 4.156 |
| Dokumen sempurna aktual → maksimum | 2.397 → 4.572 |

Dekomposisi tepat dari 24.361 slot salah aktual:

```text
24.361 = 20.723 + 3.638
3.638  =  3.506 +   132
```

20.723 adalah gap terhadap pemilihan ideal, bukan 20.723 classifier errors yang sudah terdiagnosis satu per satu. Kesalahan scoring, tradeoff decoder, serta posisi slot dapat berinteraksi. Candidate recall 95,8915% sedikit lebih tinggi daripada slot oracle 95,7369%, karena kandidat benar yang tersedia belum tentu dapat ditempatkan sekaligus pada posisi yang benar sambil memenuhi K.

### Pemisahan dokumen berdasarkan kelengkapan kandidat

| Kelompok | Dokumen | Gold slots | Benar aktual | Benar oracle | Gap pemilihan |
|---|---:|---:|---:|---:|---:|
| Semua gold tersedia | 4.572 | 54.200 | 44.265 | 54.200 | **9.935** |
| Ada gold yang hilang | 2.371 | 31.136 | 16.710 | 27.498 | **10.788** |

Kelompok pertama adalah bukti paling jelas: pada 2.175 dokumen yang seluruh jawaban benarnya sudah tersedia, model masih membuat kesalahan. Menambahkan jawaban yang hilang tidak diperlukan agar dokumen-dokumen itu bisa sempurna secara ideal.

Gold yang belum tersedia menurut label: USERNAME 1.783, ADDRESS 693, NAME 503, JOB_TITLE 244, DATE 128, PHONE 102, EMAIL 53. USERNAME tetap masalah coverage terbesar, tetapi hasil ini tidak mendukung menjadikannya perubahan pertama sebelum masalah pemilihan diperiksa.

## Keputusan dan konsiderasi berikutnya

1. **Baseline tetap bitrase-1.** Ini audit, bukan `.data-1` atau `.algo-1`; model, fitur, kandidat, dan submission tidak berubah.
2. **Langkah berikutnya: diagnosis scoring/decoder.** Mulai dari dokumen yang semua gold-nya tersedia, untuk membedakan kandidat salah yang diberi skor terlalu tinggi, kandidat benar yang terlalu rendah, serta pilihan kombinasi yang menggeser slot.
3. Classifier sekarang mempelajari benar/salah per kandidat; decoder memaksimalkan jumlah probabilitas. Keduanya tidak langsung mengoptimalkan exact-slot. Ini alasan masuk akal untuk diperiksa, tetapi oracle belum membuktikan bahwa decoder merupakan akar masalah atau menentukan solusi terbaik.
4. Perubahan scorer/objective/decoder masuk keluarga `.algo-n` sesuai konvensi penamaan. Jika diagnosis mengarah ke representasi fitur, perubahan dicatat menurut konvensi `.data-n` yang disepakati. Belum ada implementasi eksperimen baru dalam audit ini.
5. Tetap ukur kandidat baru pada eksperimen tersendiri bila diperlukan. Gap oracle bukan prediksi kausal manfaat suatu perubahan: kandidat tambahan juga dapat mengubah persaingan dan hasil pemilihan.

## Validasi dan batasan

- Empat unit tests lulus, termasuk perbandingan DP terhadap exhaustive enumeration untuk 300 kasus kecil acak deterministik, missing-first-span, filler yang memulihkan slot berikutnya, dan K yang tidak feasible.
- Semua 6.943 dokumen memenuhi `actual_correct <= oracle_correct <= candidate_gold_found <= K`.
- Saved baseline predictions berada dalam pool yang direkonstruksi, berjumlah K, dan tanpa overlap.
- Actual slot count 60.975 dan candidate gold count 81.830 cocok dengan frozen run report.
- Source hashes frozen dan diagnostic diperiksa sebelum/sesudah run. Tidak membaca train rows, reserved holdout, atau test; tidak fitting atau memuat checkpoint untuk inference.
- Tidak melakukan eksperimen generalisasi baru atau verifikasi public score. Dev sudah merupakan development data yang dianalisis; oracle menggunakan gold secara sengaja dan tidak boleh disajikan sebagai hasil prediksi.

## Artefak dan menjalankan ulang

- [Script diagnostik](../../src/audit_slot_oracle.py)
- [Tests](../../tests/test_slot_oracle.py)
- [Hasil agregat dan source hashes](../../output/bitrase-1/slot_oracle/report.json)
- [Ringkasan per dokumen tanpa teks mentah](../../output/bitrase-1/slot_oracle/documents.json)

```powershell
uv run --with scikit-learn==1.9.1 python -m unittest discover -s tests -p test_slot_oracle.py
uv run --with scikit-learn==1.9.1 python src/audit_slot_oracle.py
```

Script menolak overwrite direktori hasil yang sudah ada. Perintah kedua adalah perintah run awal; untuk menjalankan ulang, gunakan salinan workspace dengan output diagnostic belum tersedia. Environment menggunakan project uv beserta scikit-learn versi eksplisit.
