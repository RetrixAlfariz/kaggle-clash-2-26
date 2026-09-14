# bitrase-1.algo-1 — Technicalities

## Implementasi

- [slot_decoder.py](../../src/slot_decoder.py): validasi candidate probabilities, log-odds, forward/backward log-partition, positional marginals, dan constrained decoding.
- [run_algo1.py](../../src/run_algo1.py): parent hash checks, frozen checkpoint loading, replay baseline, dev ablation, paired metrics, pemilihan challenger, dan test CSV inference.
- [test_slot_decoder.py](../../tests/test_slot_decoder.py): exhaustive comparison terhadap enumerasi seluruh feasible combinations pada kasus kecil.
- Parent classifier/generator: [train_m1.py](../../src/train_m1.py); hasil learned weights tidak berubah.

## Algoritma

Dengan n kandidat dan K slots, hitung prefix partition menurut end-sorted intervals dan suffix partition menurut start-sorted intervals. Perhitungan dilakukan dalam log-space dengan `logaddexp`.

Untuk candidate i pada posisi k, seluruh pilihan sebelah kiri harus berakhir sebelum atau pada start i; seluruh pilihan sebelah kanan harus mulai setelah atau pada end i. Marginal dihitung dari:

```text
exp(prefix_log_partition[i,k-1] + logit(p_i)
    + suffix_log_partition[i,K-k] - total_log_partition[K])
```

Lalu weighted interval DP memilih tepat K kandidat dengan reward yang bergantung pada slot. Backtracking memulihkan kombinasi terpilih; sorting final mengikuti posisi. Strict `>` mempertahankan pilihan sebelumnya pada tie. Touching intervals compatible; overlap tidak compatible.

Waktu DP/marginals O(nK), sorting O(n log n), memory O(nK). Tidak ada beam pruning atau enumerasi semua kombinasi pada data penuh. Semua kombinasi hanya dienumerasi dalam small synthetic tests untuk independent mathematical check.

## Menjalankan

Gunakan root repository dan environment project Python 3.13.2 / PyArrow 25.0.1 ditambah scikit-learn 1.9.1. Parent checkpoint hanya dimuat setelah hash sesuai frozen report.

```powershell
uv run --with scikit-learn==1.9.1 python -m unittest discover -s tests -p test_slot_decoder.py
uv run --with scikit-learn==1.9.1 python src/run_algo1.py
uv run --with scikit-learn==1.9.1 python src/run_algo1.py --submission-only
uv run src/verify_submission.py --submission output/bitrase-1.algo-1/v1/submission.csv --report output/bitrase-1.algo-1/v1/submission_validation.json
```

Ini adalah urutan run awal. Evaluasi menolak existing output directory dan inference menolak overwrite CSV. Untuk reproduksi ulang, gunakan workspace copy dengan output eksperimen belum tersedia. Exact byte-identical reproduction challenger belum diuji; replay parent di run ini cocok persis.

## Provenance

`output/bitrase-1.algo-1/v1/config.json` menyimpan source hashes, parent hashes, protocol, dan probability clipping. Hash checks sebelum/sesudah evaluasi dan inference menolak source changes. Hashing train artifact hanya memeriksa identitas sumber parent; tidak membaca baris train untuk fitting. Tidak membuka reserved holdout.

Artefak hasil: `report.json`, `paired_documents.json`, `slot_mbr_dev_predictions.parquet`, `map_dev_predictions.parquet`. Selected decoder tercatat dalam report. Setelah inference, `submission_manifest.json` menyimpan CSV hash, parent model hash, test/slot hashes, dan status upload lokal. `submission_validation.json` memeriksa semua row IDs/order, bounds, labels, counts, dan non-overlap.

Tidak ada model pickle baru karena classifier sama. CSV eksperimen berada di output terpisah; CSV bitrase-1 tetap menjadi pembanding asli.
