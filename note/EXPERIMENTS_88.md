# Pencarian metode menuju 88% exact-slot

Target kerja: mencapai sekurangnya 88% exact-slot pada dev, lalu menyediakan file untuk validasi Kaggle manual oleh Vian. Target Kaggle tidak bisa dinyatakan tercapai dari dev saja. User diminta memperjelas apakah target utamanya Kaggle atau dev; sambil menunggu, eksperimen berjalan pada dev.

Semua metode tanpa pretrained, external datasets, atau test labels. Prepared v1 train/dev dipertahankan. Reserved holdout tidak digunakan untuk pencarian. Dev telah dipakai berulang untuk diagnosis dan pemilihan, sehingga hasil merupakan development selection dan bukan estimasi final yang bebas selection bias.

## Metode dan status awal

| Metode | Kode | Status saat protokol dibuat |
|---|---|---|
| Logistic span classifier + probability-sum decoder | bitrase-1 | Dev 71,4528%; Kaggle 0,705 dilaporkan Vian |
| Frozen classifier + MAP/Slot-MBR | bitrase-1.algo-1 | Slot-MBR dev 72,3024%; MAP 71,4575% |
| Boosted trees untuk candidate correctness | bitrase-1.algo-2 | Implementasi dan training terpisah |
| Linear-chain BIO CRF dari nol, raw/count-constrained | bitrase-1.algo-3 | Training awal 12.000 dokumen train |

## Hasil terverifikasi sementara

| Run | Exact-slot dev | Keputusan |
|---|---:|---|
| Boosted trees v2, sum-probability | 72,1278% | Tidak dipilih |
| Boosted trees v2, Slot-MBR | 72,5473% | Improvement kecil; kalah dari CRF |
| CRF12k raw | 56,5014% | K tidak dijamin |
| CRF12k known-K | 76,9968% | Pembanding sequence model |
| CRF+dictionary24k known-K | 71,5571% | Regresi; ditolak |
| CRF12k + text-only style adjustment | 77,6366% | Lebih baik dari CRF dasar |
| Byte-CNN BiLSTM epoch4, exact-K | 79,4225% | Checkpoint tunggal terpilih |
| Byte-CNN BiLSTM epoch8, exact-K | 79,4085% | Tidak mengungguli epoch4 |
| Neural epoch4 + honorific adjustment | 78,8811% | Regresi; ditolak |
| Neural ensemble epoch4/8, exact-K | **80,4057%** | Terbaik sementara; belum88% |
| Neural epoch4 + trained CRF transitions | 80,0670% | Membantu single checkpoint; kalah dari ensemble |

## Hasil putaran ini

Terbaik80,4057% exact-slot dev (68.615/85.336), naik8,9528 percentage points dibanding baseline71,4528%. Belum memenuhi88%; masih perlu6.481 slot benar tambahan untuk setidaknya88%. Entity F1 ensemble91,8428% tidak boleh dilaporkan sebagai keberhasilan target88%.

Submission manual yang dipilih: `output/bitrase-1.algo-6/ensemble_4_8/submission.csv`,285.318rows/23.156dokumen, validator ready=true/errors kosong. SHA256:`5bd13ceb311db06cc779ce574c7be5b0bde99ff9b1e61c034df9e0430d5a6206`. Belum ada skor Kaggle baru; upload dilakukan Vian.

Pertimbangan berikutnya: ensemble masih mempunyai9.760 entitas tepat tetapi salah slot dan3.940 overlap berlabel sama. Memperpanjang training yang sama tidak membantu pada epoch8. Joint sequence training atau keputusan yang mempertimbangkan posisi slot perlu diuji sebagai eksperimen baru; dari hasil ini belum ada bukti keduanya akan mencapai88%. Hindari memperlakukan dev yang berulang kali dipakai memilih metode sebagai validasi independen.

Hasil style adjustment awal77,3823% dibatalkan karena gold-dependent eligibility gate. Hasil terkoreksi77,6366% memakai teks saja untuk menentukan policy semua dokumen. Style-classification accuracy88,3441% bukan metrik kompetisi. Aborted boosted-tree v1 juga tidak dipakai sebagai hasil sah.

Keluarga boosted trees tetap memakai candidate pool frozen agar mengukur manfaat nonlinearity dalam scorer. Keluarga CRF mengganti pendekatan menjadi sequence labelling; fitur/representasinya ikut berubah sehingga tidak ditafsirkan sebagai ablation satu variabel. Eksperimen dipisahkan dari baseline dan tidak otomatis dipromosikan menjadi bitrase-2.

Pemilihan berdasarkan exact-slot, disertai entity F1, count correctness, dokumen sempurna, validasi independen, dan biaya run. Scaling data train atau training iterations dilakukan sebagai run terpisah bila hasil awal memberi alasan. Tidak akan mengklaim 88% jika percobaan belum mencapainya, atau menganggap batas oracle sebagai skor model.
