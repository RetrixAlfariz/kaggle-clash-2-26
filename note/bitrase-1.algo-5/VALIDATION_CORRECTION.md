# Koreksi evaluasi honorific style

Root review menemukan bahwa run awal menggunakan jumlah honorific observations dari **gold dev** untuk memilih apakah classifier style dipanggil atau default excluded diterapkan. Ini merupakan gold-dependent eligibility gate yang tidak tersedia saat inference.

Karena itu adjusted exact-slot 77,3823% pada `output/bitrase-1.algo-5/report.json` **tidak sah sebagai hasil model deployable dan tidak boleh dipilih sebagai improvement**. Artifact awal dipertahankan untuk audit, bukan dihapus. Dev style accuracy88,3441% adalah classification diagnostic terpisah, bukan exact-slot score kompetisi.

Koreksi memakai fitted style model yang sama, memprediksi policy dari teks untuk **setiap** dokumen dev, kemudian mengubah batas NAME. Gold hanya dibaca pada tahap evaluasi sesudah predictions terbentuk. Tidak refit, tidak tune threshold, dan tidak mengganti label.

Implementasi koreksi: `src/evaluate_document_style.py`. Hasil koreksi: `output/bitrase-1.algo-5/text_only_v2/`. Pengujian ke base CRF lain, bila dilakukan, harus memakai output terpisah dan dicatat sebagai kombinasi baru.
