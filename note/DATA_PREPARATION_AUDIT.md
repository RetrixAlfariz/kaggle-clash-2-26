# Audit ulang data preparation v1

> Catatan historis sebelum perbaikan builder. Status terkini dan bukti perbaikannya ada di [PREPARATION_HARDENING.md](PREPARATION_HARDENING.md). Temuan di bawah menjelaskan kode pada saat audit, bukan seluruhnya masalah yang masih terbuka.

Tanggal pemeriksaan: 14 September 2026 UTC / WIB.

**Keputusan: artefak v1 lolos pemeriksaan integritas yang dilakukan; builder perlu diperkuat sebelum menjadi pipeline rutin.** Tidak ditemukan alasan berbasis hasil audit ini untuk membersihkan ulang teks, mengganti label, atau membuang split v1. Validitas evaluasi terhadap dokumen dengan template serupa masih belum dibuktikan.

Status: **Share with caveats** untuk data v1; **Needs revision** untuk kontrol validasi builder. Cakupan audit implementasi/readback selesai; pembuktian kesiapan modelling masih **Partial**, karena near-duplicate/template leakage dan token alignment belum diperiksa. Audit ini menambahkan skrip pemeriksaan dan laporan, tanpa mengubah data mentah, builder, atau artefak v1. Tidak ada model yang dilatih.

## Hasil yang terverifikasi

Pemeriksaan independen membaca CSV/Parquet/JSONL mentah langsung, tanpa mengimpor helper dari `prepare_data.py`. Karena itu, pemeriksaan tidak hanya mengulang asumsi dari builder yang sedang diaudit.

| Objek | Cakupan | Hasil |
| --- | ---: | --- |
| Gold train + dev + holdout | 69.468 dokumen; 855.761 span | Cakupan ID lengkap, tidak ada ID ganda lintas split; teks dan entitas sama dengan sumber |
| Competition test | 23.156 dokumen | Teks dan expected entity count sesuai sumber |
| Submission slots | 285.318 baris | Isi pemetaan dan urutan global sama dengan sample submission |
| Metadata audit | 92.624 baris | Semua kolom asli dan partition sesuai sumber |
| Weak pool | 20.000 dokumen | Isi dan flag overlap sesuai sumber; tetap terpisah |
| Provenance | 13 sumber/code hashes; 8 output hashes | Semuanya sesuai manifest v1 |
| Fresh rebuild | 8 Parquet | Seluruh hash byte identik dengan v1 pada environment yang sama |
| Unit tests | 7 kasus | Lolos; tidak mencakup semua kegagalan guard yang ditemukan di bawah |

Rekonstruksi ulang memeriksa urutan segmen, cakupan metadata, jumlah karakter/kata/segmen/nonempty segments. Readback memeriksa teks utuh, span character-offset end-exclusive, label, ordering, dan batas span. Pemeriksaan ID dilakukan terhadap himpunan yang diharapkan serta jumlah baris, sehingga baris hilang atau ganda tidak tertutup oleh kesamaan total agregat.

Hasil terperinci ada di [data_checks.json](../output/preparation_audit/data_checks.json). Angka pemeriksaan di file tersebut adalah jumlah pemeriksaan teknis, **bukan persentase akurasi atau bukti seluruh risiko data telah teratasi**.

### Split dan exposure

| Partition | Dokumen | Entitas | Rata-rata entitas/dokumen |
| --- | ---: | ---: | ---: |
| Train | 55.582 | 685.026 | 12,3246 |
| Dev | 6.943 | 85.336 | 12,2909 |
| Reserved holdout | 6.943 | 85.399 | 12,3000 |

- Semua 10.000 ID training probe sebelumnya berada di train; semua 2.000 ID validation probe berada di dev. Ini dibandingkan langsung dengan report probe, bukan hanya mempercayai flag di assignments.
- Flag 80 contoh yang pernah direview sesuai sumber contoh tersimpan dan dua ID inspeksi awal; semuanya di luar holdout. Catatan ini tidak dapat merekonstruksi exposure manusia yang tidak pernah disimpan.
- Tidak ditemukan normalized duplicate group yang terpecah antara train/dev/holdout, ataupun normalized text overlap dengan competition test. Normalisasi hanya untuk grouping, bukan perubahan input.
- Seluruh 18 kombinasi channel/domain muncul di setiap supervised split. Marginal channel dan domain dev/holdout identik; stratifikasi memang menargetkan keseimbangan ini.
- Pergeseran terbesar proporsi label dev/holdout terhadap train sekitar **0,161 percentage point**. Denominator label adalah jumlah entitas; denominator channel/domain adalah jumlah dokumen. Ini statistik deskriptif, bukan bukti independensi atau kemampuan generalisasi.
- Satu weak document mempunyai normalized overlap dengan train; tidak ada overlap jenis ini dengan dev, holdout, atau competition test. Ini tidak mencakup near duplicates.

Bukti tambahan: [split_review.json](../output/preparation_audit/split_review.json).

## Temuan dan rekomendasi berprioritas

### 1. Prioritas tinggi — lengkapi readback sebelum builder dipakai rutin

**Lokasi:** [prepare_data.py](../src/prepare_data.py), blok readback sekitar baris 291–302.

Builder mengecek isi setiap baris gold yang terbaca, tetapi tidak membandingkan jumlah baris dan himpunan ID hasil baca dengan yang seharusnya ditulis. Kalau hasil baca kosong, loop tidak memeriksa apa pun. Total di akhir berasal dari objek sebelum serialization, sehingga tidak menutup celah tersebut. Selain itu, `split_assignments.parquet` dan `metadata_audit_only.parquet` belum memiliki perbandingan readback di builder.

**Bukti:** fixture yang mengeksekusi blok readback aktual dengan pembaca Parquet sintetis menerima hasil gold kosong pada mode Python normal. Ini reproduksi kekurangan guard, bukan kejadian hilangnya baris pada v1. Audit independen saat ini membuktikan baris v1 lengkap dan kedua artefak pendukung benar.

**Perbaikan yang diusulkan:** periksa expected schema, jumlah baris, uniqueness, himpunan ID, dan isi seluruh delapan file sesudah serialization. Untuk gold, bandingkan dengan ID partition yang diharapkan; untuk metadata, pastikan cakupan train+test; untuk slots, pertahankan kesamaan urutan penuh. Jadikan verifier independen sebagai gate yang keluar nonzero bila gagal.

**Mengapa:** hash membuktikan berkas tidak berubah sejak dicatat; hash sendiri tidak membuktikan berkas yang pertama kali dicatat sudah lengkap. Pemeriksaan builder harus menangkap kegagalan sebelum manifest dan final directory diterbitkan.

**Kriteria selesai:** fault injection baris hilang, baris ganda, partition salah, metadata berubah, dan slot reorder semuanya ditolak sebelum finalisasi. Pertahankan split IDs saat hanya memperbaiki guard; tidak perlu membuat pembagian baru.

### 2. Prioritas tinggi — ganti `assert` untuk kontrak integritas runtime

**Lokasi:** [prepare_data.py](../src/prepare_data.py), baris 220–251 dan 296–302; ada 12 statement `assert`.

Pemeriksaan historical roles, reviewed holdout exclusion, group isolation, slot continuity, dan sebagian readback hilang saat Python memakai optimization (`-O` atau pengaturan sejenis).

**Bukti:** fixture salah teks ditolak saat normal, tetapi diterima ketika blok yang sama dikompilasi dengan `optimize=1`. Pemanggilan normal yang digunakan untuk v1 tetap menjalankan assert; tidak ada bukti v1 dibangun dengan validasi nonaktif.

**Perbaikan yang diusulkan:** gunakan `if not condition: raise ValueError(...)` dengan pesan file/document/kontrak yang jelas. Tambahkan negative tests untuk mode normal dan `-O` pada jalur guard aktual.

**Mengapa:** integritas dataset merupakan kontrak runtime. Perilakunya tidak seharusnya berubah karena pilihan optimization interpreter. Dua prioritas pertama dapat dikerjakan dalam satu perubahan builder.

**Kriteria selesai:** malformed data ditolak pada kedua mode dengan penyebab yang sama. Bukti reproduksi: [guard_review.json](../output/preparation_audit/guard_review.json), [fixture](../src/review_prepare_guards.py).

### 3. Sebelum mempercayai skor — audit near duplicates dan keluarga template

**Status:** risiko metodologis yang belum diukur, bukan leakage yang telah terbukti.

Grouping saat ini menangkap exact/whitespace-normalized duplicates. Dua surat dengan nama/tanggal berbeda tetapi kerangka kalimat hampir sama dapat berada di split berbeda. Keseimbangan distribusi label/channel/domain tidak menguji hal ini.

**Tindakan yang disarankan:** definisikan kemiripan sebelum melihat skor model; ukur overlap word-shingles/kemiripan template, kandidat lintas split, lalu tinjau contoh train/dev. Bedakan frasa umum dari dokumen yang hampir seluruhnya sama. Untuk holdout, keluarkan ringkasan agregat sebisa mungkin agar tidak memperluas review manualnya. Jangan otomatis masking entitas atau memilih threshold berdasarkan label holdout untuk memperbaiki skor.

**Mengapa:** duplikasi template dapat mengukur hafalan format alih-alih generalisasi. Sebaliknya, pengelompokan berlebihan dapat menghasilkan split yang jauh dari distribusi competition test. Karena itu jangan langsung mengganti v1. Jika overlap material terbukti, bekukan family-grouped split sebagai evaluasi tambahan dan dokumentasikan perubahan distribusi serta exposure historis.

**Kriteria selesai:** ada definisi similarity, cakupan pencarian dan keterbatasannya, jumlah pasangan/group lintas split, contoh yang ditinjau, dan keputusan mempertahankan atau menambah split berdasarkan bukti. Ketiadaan kandidat dari pencarian perkiraan tidak boleh disebut bukti tidak ada near duplicates.

### 4. Sebelum eksperimen berulang — kunci environment dan pemisahan input

**Lokasi:** dependency builder `pyarrow>=18` di baris 3; manifest mencatat Python 3.13.2 dan PyArrow 25.0.1.

Fresh rebuild sudah identik pada environment saat ini. Namun range dependency terbuka dapat memilih versi lain di masa depan. Manifest mencatat versi yang dipakai, tetapi bukan mekanisme untuk memaksa instalasi versi tersebut.

**Tindakan yang disarankan:** pin/lock environment script yang benar-benar dipakai (`uv run src/prepare_data.py`), bukan menganggap lockfile proyek otomatis mengunci dependency PEP 723. Bedakan reproducibility isi/membership dengan kesamaan byte Parquet antarversi. Catat source/config/code/runtime pada setiap versi.

Tambahkan loader dengan daftar field eksplisit: train untuk fitting, dev untuk keputusan eksperimen, holdout untuk evaluasi setelah keputusan dibekukan. Hindari glob semua Parquet karena ada metadata audit dan weak pool. Jika builder diperbaiki, simpan provenance versi baru tanpa menimpa v1.

**Mengapa:** metadata operasional sudah dikeluarkan dari empat file model-facing, tetapi konsumen tetap dapat tanpa sengaja membaca file audit atau holdout. Pencegahan pada loader lebih dapat diandalkan daripada instruksi README saja.

### 5. Saat model/tokenizer dipilih — validasi alignment sebelum training

**Status:** belum dilakukan karena representasi model belum dipilih; bukan kekurangan data character-offset v1.

**Tindakan yang disarankan:** buat mapping character↔token, audit span yang tidak dapat direpresentasikan tepat, dan rancang windowing berdasarkan panjang token. Uji blank line, Unicode, whitespace awal span, honorific, dan alamat multiline; pastikan semua gold span mendapat coverage dan hasil decode kembali ke teks asli. Jangan mengklaim tidak ada truncation loss sebelum ini diukur.

**Mengapa:** text preservation yang benar belum menjamin token alignment benar. Kesalahan satu karakter dapat menggagalkan exact-match slot. Evaluasi utama harus memakai urutan slot kompetisi, dengan span/label diagnostics sebagai pelengkap. Expected count tersedia dari slot test untuk kompetisi ini; laporkan penggunaan count secara eksplisit, bukan sebagai NER unknown-count.

## Keputusan yang sebaiknya tetap dipertahankan

- **Pertahankan gold apa adanya.** Variasi honorific, batas alamat, dan cross-label strings pada audit sebelumnya bukan izin untuk relabel massal. Jika perlu adjudication, simpan daftar terpisah berisi bukti dan keputusan; jangan menimpa label asli.
- **Weak data tetap terpisah.** Tidak adanya ADDRESS/JOB_TITLE pada weak labels adalah anotasi parsial, bukan bukti semua token lain negatif. Penggunaannya memerlukan objective/masking dan duplicate policy tersendiri.
- **Holdout tetap reserved dengan caveat exposure.** Tidak ada fitting/scoring probe terdahulu pada holdout, tetapi corpus-wide profiling sudah melihat agregat seluruh gold. Jangan menyebutnya fully blind; jangan menjadikannya dev kedua.
- **Metadata tetap audit-only sebagai default.** Hubungan deskriptif yang kecil bukan bukti metadata tidak berguna; masukkan fitur tambahan hanya melalui eksperimen terkontrol train/dev dengan ketersediaan test yang jelas.

## Scorecard dan batas cakupan

Penyebut berikut adalah inventaris komponen dalam cakupan, bukan jumlah baris atau persentase kelulusan. `0 / N` berarti tidak ada defect teramati, bukan sertifikasi seluruh kemungkinan kegagalan. Tidak ada perubahan perbaikan pada builder dalam audit ini.

### Kegunaan dan kejelasan artefak

| Category | Observed defects | Assessment |
| --- | --- | --- |
| Usefulness/completeness | 0 / 3 | Tiga pertanyaan—integritas data, ketahanan builder, dan batas evaluasi—memiliki jawaban; near-duplicate dan token alignment dinyatakan belum terverifikasi |
| Analytical clarity | 0 / 3 | Tiga interpretasi utama dibedakan: data v1 benar pada cakupan cek, guard belum lengkap, generalisasi belum terbukti |
| Visual/interaction consistency | N/A | Dataset dan laporan Markdown; tidak ada dashboard interaktif yang diaudit |

### Kebenaran analitis dan ketahanan

| Category | Observed defects | Assessment |
| --- | --- | --- |
| Source authority/confidence | 0 / 3 | Tiga kelompok sumber: raw competition files, catatan exposure tersimpan, kode/manifest; exposure tak tercatat tidak dapat dipastikan |
| Value accuracy | 0 / 8 | Delapan Parquet diperiksa terhadap sumber dan kontrak; tidak ditemukan mismatch aktual |
| Within-chart agreement | N/A | Tidak ada grafik baru |
| Complete source details | 0 / 3 | Tiga bukti audit tersedia: data_checks, guard_review, split_review, beserta skripnya |
| Cross-artifact consistency | 0 / 3 | Raw↔prepared, prepared↔manifest, dan v1↔fresh rebuild sesuai pada environment saat ini |
| Data-quality controls | 3 / 5 | Lima keluarga kontrol: input reconstruction, split/slot guards, serialized readback, hash/staging, environment. Defect pada guards, readback completeness, dan version pinning; dampak korupsi v1 tidak ditemukan |
| Conclusion support | 0 / 3 | Integritas v1 didukung full readback; celah guard didukung fixture; ketidakpastian evaluasi diungkap, tanpa klaim leakage terbukti |

Tidak dilakukan: semantic near-duplicate search, evaluasi model baru, tokenizer/windowing, penggunaan weak data untuk training, adjudication label, atau pemeriksaan ulang aturan Kaggle secara live. Kontrak kompetisi mengikuti README dan catatan lokal yang sudah tersedia.

## Bukti dan reproduksi

- [Audit readback independen](../src/audit_prepared_data.py) → [data_checks.json](../output/preparation_audit/data_checks.json).
- [Fixture guard](../src/review_prepare_guards.py) → [guard_review.json](../output/preparation_audit/guard_review.json).
- [Review split independen](../src/review_prepared_splits.py) → [split_review.json](../output/preparation_audit/split_review.json).
- Fresh build pembanding disimpan di `output/preparation_audit/rebuild/`; v1 tetap pada `output/prepared/v1/`.
- Konteks: [data preparation](DATA_PREPARATION.md) dan [data understanding](REPORT.md).

```powershell
uv run src/prepare_data.py --output output/preparation_audit/rebuild
uv run src/audit_prepared_data.py
uv run python src/review_prepare_guards.py
uv run src/review_prepared_splits.py
uv run --with pyarrow python -m unittest discover -s tests -v
```

Untuk audit ulang setelah source/build berubah, perhatikan bahwa verifier ini sengaja menunjuk snapshot `v1`. Jangan menganggap hasil audit lama tetap berlaku untuk versi baru; pindahkan target verifier secara eksplisit dan buat bukti baru.

**Urutan kerja yang disarankan:** perbaiki guard dan readback → bekukan environment serta akses split → audit template similarity → pilih tokenizer dan validasi alignment → mulai baseline train/dev. Buka reserved holdout setelah keputusan pengembangan dibekukan.
