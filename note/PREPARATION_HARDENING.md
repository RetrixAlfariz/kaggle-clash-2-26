# Penutupan tindak lanjut data preparation

Tanggal: 15 September 2026 WIB. Implementasi lokal mengikuti tiga paket kerja dari review; tidak membuat atau memublikasikan PR remote.

**Artefak `output/prepared/v1` tetap digunakan dan tidak diubah.** Perubahan menyasar builder dan cara mengakses data. Build terpisah untuk menguji builder menghasilkan delapan Parquet yang byte-identical dengan v1, termasuk split assignments. Manifest v1 juga identik dengan hash pada audit sebelumnya.

## Paket 1 — Builder guards dan serialized readback

[prepare_data.py](../src/prepare_data.py) sudah mengganti seluruh 12 integrity `assert` dengan pengecekan eksplisit yang melempar `ValueError`. Kontrak ini tetap aktif pada Python `-O`.

Validasi sebelum finalisasi sekarang memeriksa:

- Inventaris tepat delapan Parquet.
- Schema, jumlah baris, uniqueness dan cakupan ID yang diharapkan.
- Isi serta urutan setiap baris, termasuk gold, test slots, split assignments, metadata, dan weak pool.
- Keutuhan sumber sepanjang build, sebelum manifest ditulis dan staging dipindahkan ke tujuan final.

Uji negatif meliputi hasil kosong/baris hilang, duplicate ID, ID salah, partition salah, metadata berubah, teks berubah, slot reorder, schema tambahan/salah, dan artifact hilang. Kasus ini dijalankan pada mode normal dan optimized. Nama internal list Arrow `item` dapat berubah menjadi `element` setelah Parquet roundtrip; perbandingan schema memakai semantic equality Arrow, tetap memeriksa field/type/nullability. Ada regression test nested gold untuk kasus ini.

**Mengapa dilakukan:** validasi sebelumnya hanya memeriksa baris yang berhasil terbaca; berkas gold kosong bisa melewati loop. Total sebelum serialization dan hash file tidak membuktikan kelengkapan hasil serialization. Readback lengkap menutup celah itu tanpa mengubah isi dataset atau split.

Builder sekarang mewajibkan `--output` eksplisit. Menunjuk v1 dengan kode baru akan ditolak oleh provenance guard, karena v1 milik build historis. Default loader tetap menunjuk v1; build di `output/hardening_validation/build` hanya bukti validasi, bukan dataset pengganti.

## Paket 2 — Environment dan model-facing loader

Environment preparation dikunci ke **Python 3.13.2 / PyArrow 25.0.1** melalui `.python-version`, `pyproject.toml`, `uv.lock`, serta metadata PEP 723 dan [script lockfile](../src/prepare_data.py.lock). Builder juga memeriksa versi runtime. Lockfile script tercatat dalam provenance build baru.

```powershell
uv sync --locked
uv run --locked src/prepare_data.py --output output/hardening_validation/build
```

Gunakan [prepared_loader.py](../src/prepared_loader.py):

```python
from src.prepared_loader import PreparedData

data = PreparedData()  # output/prepared/v1
train = data.load_train()       # train_fitting
dev = data.load_dev()           # tuning
test = data.load_test()         # inference
slots = data.load_test_slots()  # inference
```

| Akses | Tujuan yang diterima | Perilaku |
| --- | --- | --- |
| `load_train()` | `train_fitting` | Sumber fitting |
| `load_dev()` | `tuning`, `evaluation` | Tidak menerima fitting |
| `load_holdout()` | Hanya final evaluation eksplisit | Ditolak secara default |
| `load_test()`, `load_test_slots()` | `inference` | Tidak menerima fitting/tuning |

Saat keputusan eksperimen sudah dibekukan, akses holdout harus disengaja:

```python
holdout = data.load_holdout("final_evaluation", allow_final_evaluation=True)
```

Loader mengembalikan `pyarrow.Table`, memeriksa hash, jumlah baris, dan schema manifest, serta schema model yang didefinisikan independen. Metadata/weak fields ditolak bahkan bila seseorang mengganti file beserta hash/schema dalam manifest. Tidak ada glob untuk memilih input model.

**Mengapa dilakukan:** dependency range terbuka tidak menjamin versi yang sama saat rerun. Pemisahan file juga belum cukup bila konsumen membaca semua Parquet. Loader menutup jalur kesalahan tidak sengaja; ini **workflow safeguard**, bukan kontrol keamanan terhadap orang yang sengaja membaca Parquet langsung. Flag final evaluation tidak membuktikan keputusan eksperimen benar-benar sudah dibekukan; disiplin proses itu tetap diperlukan.

## Bukti verifikasi

- [verification.json](../output/hardening_validation/verification.json): hasil unit tests kedua mode, actual loader reads, hash v1, dan perbandingan bytes build baru.
- [Audit independen v1](../output/hardening_validation/audit_v1/data_checks.json): 46 checks, tidak ada failed check.
- [Audit independen build baru](../output/hardening_validation/audit_build/data_checks.json): 46 checks, tidak ada failed check.
- Seluruh delapan Parquet v1 dan manifest-nya tetap sama; delapan Parquet hasil builder baru identik dengan v1. Karena itu split ID, teks, labels, dan urutan slot tidak berubah.
- Rerun builder pada path validasi yang sama memverifikasi hash dan menggunakan build yang sudah ada tanpa menulis ulang.
- Actual loader smoke test membaca train/dev/test/test_slots; akses holdout default ditolak. Akses holdout final diuji pada fixture sintetis.

Source builder historis disimpan di [snapshot](../output/preparation_audit/source_snapshots/prepare_data.py), dengan hash sesuai manifest v1. Audit independen membedakan kode historis yang sudah diverifikasi dengan kode kerja yang telah berubah; tidak menyebut hash kode saat ini sebagai hash pembuat v1.

```powershell
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python -O -m unittest discover -s tests -v
uv run src/audit_prepared_data.py --build output/prepared/v1 --output output/hardening_validation/audit_v1
uv run src/audit_prepared_data.py --build output/hardening_validation/build --output output/hardening_validation/audit_build
uv run --frozen python src/verify_preparation_hardening.py
```

## Paket 3 dan batas modelling

Hasil template similarity dan keputusan evaluasi tambahan dicatat terpisah di [TEMPLATE_SIMILARITY.md](TEMPLATE_SIMILARITY.md). Analisis itu memakai teks tanpa gold labels; holdout hanya dilaporkan secara agregat. Pencarian perkiraan tidak boleh ditafsirkan sebagai bukti tidak ada pasangan mirip yang terlewat.

Tidak ada positive pair pada threshold Jaccard 0,8/0,9 dalam kandidat yang ditemukan. Namun delapan contoh train/dev terkuat menunjukkan kerangka appointment confirmation yang berulang. Karena itu high-threshold literal audit selesai, sedangkan independensi terhadap semantic/generator families belum terbukti. Tidak dibuat grouped split otomatis dari graph tanpa positive edges.

| Checklist | Status saat ini |
| --- | --- |
| Artifact v1 dan split IDs | Dipertahankan; integritas dan byte identity diverifikasi |
| Builder hardening | Selesai pada kontrak yang diuji |
| Environment pinning dan safe loader | Selesai; holdout development ditolak |
| Literal template/near-duplicate audit | Selesai sebagai pencarian perkiraan dengan batas cakupan eksplisit |
| Generalisasi lintas semantic/generator family | Belum terbukti; kerangka berulang terlihat pada review train/dev |
| Tokenizer/offset alignment | Menunggu pilihan tokenizer/model |
| M0 dictionary + regex baseline | Selesai; [hasil dan diagnosis dev](M0_BASELINE_RESULTS.md). Neural training belum dilakukan |

Tidak dilakukan training, pemilihan neural model, atau token alignment dalam perubahan ini. Token alignment tetap bergantung pada tokenizer yang dipilih. Artifact character-offset v1 yang valid tidak menjamin semua batas gold dapat direpresentasikan oleh token; alignment, windowing, dan batas chunk harus diuji sebelum training model tersebut.

Status preparation dilaporkan berdasarkan checklist dan bukti, bukan persentase “90%” tanpa denominator. [Audit awal](DATA_PREPARATION_AUDIT.md) tetap disimpan sebagai catatan kondisi sebelum perbaikan.
