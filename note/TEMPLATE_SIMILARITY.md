# Template similarity — pemeriksaan pertama

Tanggal: 15 September 2026 WIB. Target: mengukur kemiripan teks lintas split tanpa gold labels, tanpa mengubah v1, dan tanpa menambah review contoh holdout.

**Tidak ditemukan pasangan pada ambang literal 5-word-shingle Jaccard 0,8 atau 0,9 dalam kandidat yang diperiksa. Namun review pasangan terkuat memperlihatkan kerangka surat yang berulang.** Jadi hasil ini tidak membuktikan bebas template leakage dan tidak menutup risiko kemiripan semantik/template yang diparafrasekan.

Keputusan: pertahankan split v1. **Belum ada dasar dari threshold audit ini untuk membuat `template_grouped_dev` secara otomatis.** Untuk klaim generalisasi lintas template, evaluasi family-separated tetap layak dipertimbangkan setelah definisi family dan cakupannya diukur. Jangan menurunkan threshold sesudah melihat hasil lalu menganggap kelompok baru sebagai ground truth.

## Metode yang digunakan

- Semua 55.582 dokumen train sebagai reference; semua 6.943 dev dan 6.943 holdout sebagai query. Tidak ada sampling atau candidate cap. Pasangan di dalam split dan competition test tidak dicari.
- Normalisasi hanya untuk audit: lowercase dan whitespace word splitting, kemudian himpunan 5-word shingles. Teks model asli tidak berubah. Tidak menggunakan anotasi, entity masking, atau skor model.
- Kandidat diperoleh dari delapan deterministic affine MinHash permutations atas BLAKE2b 64-bit shingle hashes, dibagi empat band berisi dua nilai. Ini pencarian perkiraan, bukan exhaustive all-pairs atau jaminan recall.
- Ambang **0,8 dan 0,9 ditetapkan sebelum scoring**. Kandidat dihitung dengan hashed-shingle Jaccard; setiap pasangan yang mencapai 0,8 diperiksa ulang dengan string shingles asli. Delapan pasangan dev terkuat juga diperiksa dengan string shingles meskipun berada di bawah ambang.
- Tidak ada dokumen kurang dari lima kata pada ketiga split. Implementasi mengecualikan empty shingle sets agar dua dokumen pendek tidak menjadi false match.
- Holdout hanya menghasilkan counts/distribution. Tidak ada ID, pasangan, excerpt, atau family assignment holdout disimpan. Akses ini adalah audit teks agregat, bukan membuka holdout untuk tuning model.

## Hasil

| Ukuran | Dev → train | Holdout → train |
| --- | ---: | ---: |
| Query documents | 6.943 | 6.943 |
| Unique candidate pairs scored | 76.011 | 75.634 |
| Queries tanpa kandidat | 4.680 | 4.618 |
| Pasangan terverifikasi Jaccard ≥ 0,8 | 0 | 0 |
| Pasangan terverifikasi Jaccard ≥ 0,9 | 0 | 0 |
| Maximum retrieved score | 0,303738 | 0,330508 |
| P95 best retrieved score | 0,112000 | 0,116981 |

Semua kandidat berada di bawah 0,5. Zero-candidate query tidak berarti tidak punya tetangga mirip; pencarian bisa tidak menemukannya. Karena itu mean/quantile di JSON adalah **best retrieved scores**, bukan true nearest-neighbor distribution. Angka maksimum juga terbatas pada kandidat yang ditemukan.

Connected components dari **cross-split train/dev edges yang terverifikasi** berjumlah nol pada kedua ambang. Ini bukan jumlah semua template family di dataset. Transitive component pada umumnya juga tidak berarti setiap pasangan anggotanya memenuhi ambang; implementation tests mencakup perbedaan ini.

## Review contoh train/dev

Delapan pasangan dengan skor tertinggi, dipilih setelah scoring untuk pemeriksaan kualitatif, diperiksa melalui excerpt maksimal 1.000 karakter per dokumen. Seluruh skor dihitung lagi dari teks penuh. Ini sampel terpilih yang bias ke dokumen paling mirip, bukan sampel acak untuk memperkirakan prevalensi template.

| Pasangan dev / train | Exact Jaccard | Pengamatan |
| --- | ---: | --- |
| DOC_047225 / DOC_039124 | 0,303738 | Konfirmasi janji klinik; pembuka, jadwal, instruksi kedatangan, alamat dan telepon berulang; rincian dan susunan paragraf berbeda |
| DOC_014374 / DOC_074555 | 0,293878 | Email/surat konfirmasi janji; instruksi kedatangan dan reschedule serupa, dengan variasi nama/waktu/kontak |
| DOC_103301 / DOC_008710 | 0,283422 | Jadwal konsultasi dan instruksi paperwork serupa; bagian kontak dan administrasi berbeda |
| DOC_115952 / DOC_009008 | 0,277512 | Appointment reminder dengan insurance/photo ID dan reschedule; beberapa field dan kalimat berubah |
| DOC_071229 / DOC_026272 | 0,275000 | Pembuka appointment confirmation dan instruksi administrasi berulang; tambahan informasi berbeda |
| DOC_071294 / DOC_051215 | 0,264463 | Jadwal/alamat klinik serupa; variasi keterangan dokter dan prosedur administrasi |
| DOC_128625 / DOC_009008 | 0,252336 | Instruksi appointment berulang; variasi pasien, dokter dan jalur kontak |
| DOC_105200 / DOC_027021 | 0,252252 | Pembuka, jadwal dan alamat serupa; tambahan instruksi dan sign-off berbeda |

**Interpretasi:** kerangka appointment confirmation berulang lintas split pada contoh ini. Karena pergantian satu kata memengaruhi sampai lima shingles, Jaccard 5-gram dapat rendah walaupun pembaca melihat pola surat serupa. Nilai di bawah 0,8 tidak meniadakan shared-template risk. Kita belum mengetahui generator asal, prevalensi family ini pada seluruh korpus, atau pengaruhnya terhadap skor model.

## Rekomendasi dan konsiderasi

1. **Gunakan v1 untuk baseline train/dev yang dinyatakan mengukur distribusi kompetisi.** High-literal-overlap audit tidak memberi bukti yang memerlukan resplit. Jangan mengklaim hasil baseline nanti sebagai generalisasi ke template yang belum pernah dilihat.
2. **Jangan membuat grouped split dari nol positive edges ini.** Split tersebut tidak memberi pemisahan tambahan yang bermakna. Jika robustness terhadap template menjadi target, definisikan family memakai fitur struktur/semantik yang label-independent, validasi pada train/dev, dan laporkan perubahan ukuran serta distribusi kelompok sebelum membuat evaluasi tambahan.
3. **Pertahankan ambang audit ini sebagai catatan tetap.** Uji sensitivitas atau metode lain boleh menjadi eksperimen terpisah; jangan menyamakan hasil metode baru dengan hasil awal atau memilih threshold berdasarkan holdout/model score.
4. **Holdout tetap reserved.** Hasil agregat ini tidak menjadi izin untuk membukanya per contoh. Tidak dilakukan model selection, threshold tuning, ataupun training pada holdout.

Pemeriksaan word-shingle ini selesai dalam scope-nya. Structural-line patterns, semantic template clustering, retrieval recall pada pasangan nyata yang belum diketahui, dan dampak pada model **belum diverifikasi**. Batas ini tetap relevan setelah builder dan loader selesai diperbaiki.

## Bukti dan reproduksi

- [Report JSON utama](../output/template_similarity/template_similarity_report.json): definisi, sumber/code hashes, runtime, candidate denominators, distributions, family counts, dan contoh train/dev.
- [Train/dev positive pairs](../output/template_similarity/train_dev_pairs.json): kosong karena tidak ada kandidat terverifikasi ≥0,8.
- [Validasi independen](../output/template_similarity/validation.json): recomputation delapan skor contoh, arithmetic, hashes, dan pemeriksaan tidak ada ID holdout dalam output.
- [Script audit](../src/audit_template_similarity.py), [script verifier](../src/verify_template_similarity.py), [tests](../tests/test_template_similarity.py).

```powershell
uv run --frozen python src/audit_template_similarity.py
uv run --frozen python src/verify_template_similarity.py
uv run --frozen python -m unittest discover -s tests -v
```

Kembali ke [status preparation](PREPARATION_HARDENING.md).
