# Honorific style — hasil dengan text-only inference

**Hasil valid adalah text_only_v2**, bukan report awal. [Koreksi evaluasi](VALIDATION_CORRECTION.md) menjelaskan gold-dependent gate pada implementasi awal yang ditemukan root review. Hasil awal tidak dipakai untuk pemilihan.

TF-IDF word1–2gram + SGD logistic style model dilatih pada39.334 dokumen train dengan minimal dua honorific observations, memakai majority included/excluded sebagai target. Saat inference diperbaiki, policy diprediksi dari teks pada **semua** dev docs tanpa eligibility dari gold. Threshold tetap0.5, hanya boundary NAME yang disesuaikan. K tetap sama; adjustment invalid kembali ke prediksi base.

| Versi | Exact-slot dev | Unordered entity F1 |
|---|---:|---:|
| CRF12k known-K | 76,9968% | 90,5796% |
| **Text-only style adjustment** | **77,6366%** | **91,4468%** |

Hasil naik **0,6398 pp /546 slot benar**. Total66.252 slot benar,78.037 entitas benar,3.340 dokumen sempurna. [Independent check](../../output/bitrase-1.algo-5/text_only_v2/independent_check.json) cocok dan semua output contracts valid.

Style-classification accuracy88,3441% pada subset dev dengan observasi gold merupakan diagnostic tersendiri. **Itu bukan exact-slot kompetisi dan bukan pencapaian target88%.** Mixed annotation policy dan kesalahan classifier masih membatasi metode.

Sumber: [train audit](../../output/experiments88/honorific_train_audit.json), [corrected evaluator](../../src/evaluate_document_style.py), [corrected report](../../output/bitrase-1.algo-5/text_only_v2/report.json). Tidak ada pretrained atau holdout access. Model style tidak dilatih ulang setelah koreksi evaluasi.
