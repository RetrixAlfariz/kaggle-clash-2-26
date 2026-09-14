# bitrase-1.algo-3 — CRF12k results

CRF sequence labelling dari nol, train12.000 dokumen /2.834.474 tokens. 60 L-BFGS iterations, features lexical/shape/context tanpa dictionary candidate features. Training sekitar275 detik, keseluruhan train+dev sekitar396 detik.

| Decoder | Exact-slot dev | Unordered entity F1 | Dokumen sempurna |
|---|---:|---:|---:|
| Raw Viterbi | 56,5014% | 90,1794% | 1.968 |
| **Legal BIO, tepat exposed K** | **76,9968%** | **90,5796%** | **2.947** |

Known-K:65.706/85.336 slot benar pada6.943 dev documents. [Independent check](../../output/bitrase-1.algo-3/crf12k/independent_known_k.json) cocok; seluruh dokumen memenuhi count, sorting, bounds, labels, dan non-overlap.

Ada57 gold entities train subset dan41 dev entities yang tidak align token; tidak ada gold dev yang dikeluarkan dari penilaian. In-sample probe256 training docs mendapat82,9136% exact-slot dan91,9031% entity F1; ini diagnosis fit pada train, bukan bukti generalisasi.

Compared dengan bitrase-1.algo-1: naik4,6944pp exact-slot. Masih belum mencapai88%. Salah satu error besar adalah batas NAME, termasuk honorific inclusion/exclusion. Tidak mengubah gold atau menerapkan strip rule global.

Metode ini mengganti representasi dan model secara bersamaan. Tidak ada pretrained, external data, atau holdout access. [Spec](spec.md), [technicalities](technicalities.md), [protocol](protocol.md), [machine report](../../output/bitrase-1.algo-3/crf12k/report.json).
