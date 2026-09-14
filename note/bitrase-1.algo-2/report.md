# bitrase-1.algo-2 — Boosted candidate scorer

## Hasil sah: v2

LightGBM dari nol pada semua 55.582 train documents, memakai frozen M1 candidate pool. Features mencakup categorical phrase/context/label dan numeric prior/shape/boundary. Negatives dibatasi40 per dokumen dengan inverse inclusion probability weights; semua positives dipakai.

2.882.647 kandidat training. 500 trees, learning_rate .06,63leaves,8threads. Encoding, training, dan dev evaluation sekitar115 detik pada run v2. Tidak ada pretrained atau dev-based early stopping.

| Decoder | Exact-slot dev | Unordered entity F1 |
|---|---:|---:|
| Sum probabilities | 72,1278% | 89,7136% |
| Slot MBR | **72,5473%** | **89,5074%** |

Hasil Slot MBR hanya +0,2449 pp terhadap bitrase-1.algo-1 (72,3024%). Ini improvement kecil, belum mencapai88% dan kalah dari CRF12k. Tidak menghasilkan submission baru dari metode ini.

Hasil [independent check](../../output/bitrase-1.algo-2/v2/independent_slot_mbr.json) cocok dengan report:61.909 slot benar /85.336, valid coverage, offsets, K, labels, non-overlap. Root tidak menafsirkan pemakaian sample weights sebagai bukti calibration sempurna.

## Run awal yang tidak sah

[Aborted runs](aborted_runs.md) mencatat upaya v1 yang tidak menghasilkan model/report sah. Ada penggunaan hard labels sebagai probability sebelum diperbaiki, serta negative sampling tanpa correction. Hasil tersebut tidak menjadi pembanding atau dasar pemilihan. V2 merupakan implementasi baru dengan fitur kategori nyata, context tokenization sekali per dokumen, predict_proba, correction weights, dan hash guards.

Sumber: [protocol v2](protocol_v2.md), [script](../../src/run_boosted_v2.py), [config](../../output/bitrase-1.algo-2/v2/config.json), [report](../../output/bitrase-1.algo-2/v2/report.json). Kandidat gold tidak ditambahkan pada inference dan reserved holdout tidak dibuka.
