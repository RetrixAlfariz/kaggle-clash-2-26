# M0 baseline — hasil dan diagnosis dev

Tanggal: 15 September 2026 WIB. Protokol: [M0_EVALUATION_PROTOCOL.md](M0_EVALUATION_PROTOCOL.md). Run lokal `output/m0/v1`; satu konfigurasi tetap, tanpa tuning setelah melihat skor.

**Known-count exact slot accuracy: 64.00%.** Entity micro F1 adalah 85.82% dan candidate recall 94.70%. Gap besar terhadap slot metric terutama tampak sebagai span exact yang berada di rank berbeda. Ini bukan bukti bug sorting: output sudah terurut; pilihan span sebelumnya menggeser posisi entitas berikutnya.

Fitting memakai seluruh 55,582 train documents dan menghasilkan 71,494 dictionary phrases. Evaluasi memakai seluruh 6,943 dev documents / 85,336 gold entities, dengan 472,903 kandidat span-label. Holdout, competition test, weak pool, dan metadata audit tidak dibaca oleh run M0. Gold dev hanya dipakai untuk evaluasi sesudah prediction; K tersedia sebagai input count sesuai protokol kompetisi.

## Hasil dua decoder

| Metrik | Known count — utama | Unconstrained — diagnostik |
| --- | ---: | ---: |
| Exact slot accuracy | 64.00% | 43.99% |
| Entity micro precision | 85.82% | 89.15% |
| Entity micro recall | 85.82% | 82.00% |
| Entity micro F1 | 85.82% | 85.43% |
| Boundary-only F1 | 86.24% | 85.46% |
| Candidate recall | 94.70% | 94.70% |
| Exact-document rate | 24.08% | 13.19% |
| Count-correct-document rate | 100.00% | 29.08% |

Known-count menghasilkan 85,336 prediksi dan jumlah benar pada seluruh dokumen. Tetapi hanya 54,617 slot exact; memenuhi jumlah slot tidak berarti entitasnya benar. Unconstrained menghasilkan 78,492 prediksi, dengan 3,912 dokumen under-count dan 1,012 over-count.

**Perbandingan ini bukan ablation murni efek K.** Known-count memaksimalkan sum-confidence dengan fixed K; unconstrained memakai log-odds dan tanpa K. Selisih skor mencakup constraint dan objective yang berbeda. Angka known-count juga bukan hasil unknown-count NER.

## Per label — known count

| Label | Gold | Candidate recall | Precision | Recall | Entity F1 | Slot accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NAME | 26,009 | 98.07% | 79.61% | 81.32% | 80.46% | 62.32% |
| DATE | 16,239 | 99.21% | 89.59% | 95.54% | 92.47% | 69.88% |
| EMAIL | 9,723 | 99.45% | 98.41% | 98.26% | 98.34% | 68.18% |
| PHONE | 9,300 | 98.90% | 96.89% | 95.51% | 96.19% | 65.89% |
| ADDRESS | 8,226 | 91.58% | 89.11% | 81.30% | 85.03% | 58.49% |
| USERNAME | 7,642 | 63.40% | 73.01% | 55.68% | 63.18% | 44.88% |
| JOB_TITLE | 8,197 | 97.02% | 77.83% | 87.75% | 82.49% | 73.95% |

## Diagnosis full-dev

### Candidate generation dan selection

- 80,816 gold triples tersedia sebagai kandidat exact; 4,520 tidak tersedia.
- 7,579 kandidat gold exact tersedia tetapi tidak dipilih. Ini batas scoring/selection pada baseline, bukan kegagalan candidate generation.
- USERNAME menyumbang 2,797 dari 4,520 gold triples yang tidak tersedia (61.88%). Candidate recall USERNAME hanya 63.40%.

| Candidate-stage category | Gold entities |
| --- | ---: |
| `exact_candidate` | 80,816 |
| `overlap_same_label` | 2,217 |
| `overlap_wrong_label` | 294 |
| `no_overlapping_candidate` | 1,448 |
| `same_boundary_wrong_label` | 561 |

Candidate recall merupakan batas coverage candidate pool untuk entity recall; bukan jaminan slot accuracy sebesar itu dapat dicapai oleh decoder.

### Output, boundary, dan rank

| Output-stage category | Gold entities |
| --- | ---: |
| `correct_slot` | 54,617 |
| `overlap_same_label` | 5,979 |
| `no_overlapping_prediction` | 5,589 |
| `exact_span_wrong_slot` | 18,620 |
| `overlap_wrong_label` | 173 |
| `same_boundary_wrong_label` | 358 |

Sebanyak 18,620 dari 30,719 slot salah (60.61%) memiliki span-label exact di posisi lain. Ini gejala propagasi rank akibat komposisi prediksi. Tidak sah menambahkan angka ini ke skor kompetisi atau menggeser slot memakai gold.

Pada NAME ada 3,826 kasus overlap dengan label benar tetapi batas berbeda. Sebanyak 3,572 di antaranya (93.36%) berbeda hanya pada honorific awal menurut regex diagnostik Mr/Mrs/Ms/Miss/Dr/Prof/Mx. Pemeriksaan ini many-to-one dengan satu representatif overlap terbesar, bukan matching tambahan untuk F1.

## Contoh konkret yang diperiksa

| Dokumen | Temuan | Implikasi |
| --- | --- | --- |
| DOC_000003 | Gold `Mr. Robert Johnson`, prediksi `Robert Johnson`; kandidat keduanya tersedia | Scorer phrase-global memilih batas tanpa honorific pada konteks yang gold-nya memasukkan honorific |
| DOC_000003 | Gold `rjohnson_prime`, prediksi `rjohnson` | Cakupan username baru dan aturan batas underscore perlu diuji; matcher alphanumeric saat ini mengizinkan match berhenti sebelum underscore |
| DOC_000044 | Kandidat `Mr. Chen` tersedia tetapi tidak dipilih; DATE setelahnya exact namun salah slot | Pemilihan kandidat dapat menggeser banyak posisi; sorting ulang saja tidak memperbaikinya |
| DOC_000678 | Gold alamat penuh, output memilih `456` sebagai PHONE | Fixed K dapat memilih kandidat confidence rendah yang berasal dari potongan teks; jumlah slot benar bukan bukti coverage entitas benar |
| DOC_002447 | Gold USERNAME `Sterling & Associates`, output NAME | Satu label global per phrase tidak bisa menangkap variasi anotasi/konteks; jangan otomatis mengganti gold |

Contoh disimpan deterministik, maksimum empat per kategori error, bukan sampel representatif. Frekuensi di atas berasal dari seluruh dev. Teks asli dan labels tidak dibersihkan atau direlabel.

## Rekomendasi berikutnya dan alasannya

1. **Eksperimen scoring/boundary NAME yang memakai konteks lokal.** Candidate coverage NAME sudah tinggi, tetapi ribuan kasus salah batas hanya berbeda honorific. Uji fitur konteks/salutation atau scorer kandidat pada train, lalu ukur di dev. Jangan membuat aturan selalu-strip/selalu-include; audit anotasi sebelumnya menunjukkan kedua convention ada.
2. **Perluas candidate generation USERNAME secara context-aware.** Ini penyumbang terbesar gold yang tidak masuk candidate pool. Uji kandidat setelah penanda username/portal/account dan batas underscore secara terpisah; ukur tambahan recall beserta false positives. Hindari regex identifier generik tanpa konteks.
3. **Lakukan ablation objective decoder yang terkontrol.** Pertahankan candidate pool, ubah satu komponen scoring/constraint per eksperimen, dan ukur exact slot bersama entity F1 serta rank shifts. Jangan menyimpulkan decoder DP salah: ia mengoptimalkan objective heuristik yang diberikan.
4. **Pertahankan EMAIL/PHONE sebagai bagian baseline yang relatif kuat.** Keduanya mempunyai entity F1 tinggi; dampak slot masih dipengaruhi entitas sebelumnya. Memperluas regex keduanya bukan prioritas utama dari hasil ini.

Ini rekomendasi untuk eksperimen berikutnya; perubahan tersebut belum dijalankan. Tidak memilih tokenizer atau melatih neural model pada langkah 1–3 ini.

## Verifikasi dan reproducibility

- Lima contract tests evaluator/baseline lolos: rank shift, exact boundary/label, non-overlap/fixed K, invalid spans, dan fitting tanpa dev labels.
- Verifier terpisah menghitung ulang coverage ID, slot/entity/boundary counts dari seluruh prediksi tersimpan. Candidate recall dihitung melalui direct gold-substring dictionary lookup dan regex, tanpa memakai trie matcher. Tidak ditemukan mismatch.
- Hash data train/dev, source code, protokol, environment, dictionary, config, predictions dan examples dicatat/diperiksa. Existing output ditolak agar run lama tidak tertimpa.
- Skor ini development score; dev telah digunakan dalam audit sebelumnya. Tidak ada confidence interval atau klaim generalisasi ke template yang belum pernah dilihat.

Bukti: [report.json](../output/m0/v1/report.json), [frozen config](../output/m0/v1/frozen_config.json), [diagnostics dan verification](../output/m0/v1/validation_and_diagnostics.json), [error examples](../output/m0/v1/error_examples.json), [predictions](../output/m0/v1/dev_predictions.parquet), [dictionary](../output/m0/v1/dictionary.parquet).

```powershell
uv run --frozen python -m unittest discover -s tests -p test_m0.py -v
uv run --frozen python src/analyze_m0_errors.py
# Untuk rerun baru, jangan menimpa snapshot v1:
uv run --frozen python src/run_m0.py --output output/m0/reproduction
```

Rerun penuh selesai di `output/m0/reproduction`. Config, dictionary Parquet, prediction Parquet, dan error examples identik byte demi byte; seluruh metrics dan source hashes juga identik. `elapsed_seconds` dapat berbeda. Bukti: [reproducibility.json](../output/m0/v1/reproducibility.json).
