# Project-ABD: Traffic Volume Prediction Pipeline

# Kelompok 3

## Identitas Mahasiswa

| No | Nama | Peran | NIM |
|----|------|-------|-----|
| 1 | Nydia Manda Putri | Ketua | 123450018 |
| 2 | Najla Juwairia | Anggota 1 | 122450037 |
| 3 | Adil Aulia Rahma Nurhidayan | Anggota 2 | 122450058 |
| 4 | Kharisma Mustika Sari | Anggota 3 | 123450034 |
| 5 | Iqfina Haula Halika | Anggota 4 | 123450076 |

## Judul Tugas Besar

**Analisis dan Prediksi Kemacetan Lalu Lintas Kota Menggunakan Apache Spark dan Arsitektur Medallion**

---

## 1. Deskripsi Proyek

Proyek ini bertujuan memprediksi **Traffic Volume per jam** menggunakan pipeline ETL multi-layer **Bronze → Silver → Gold** dan model Machine Learning **Random Forest Regressor**.  

Pipeline ini memproses data mentah traffic, membersihkan data, menyiapkan fitur, dan membangun model prediksi agar hasil prediksi traffic per jam menjadi lebih akurat.

**Tujuan Utama:**
- Menyediakan prediksi traffic yang membantu analisis kemacetan lalu lintas kota.
- Memvisualisasikan trend prediksi agar lebih mudah dipahami.
- Menyusun pipeline yang modular menggunakan arsitektur Bronze → Silver → Gold untuk proses ETL dan Machine Learning.

---

## 2. Pipeline Data

### 2.1 Bronze Layer

Bronze Layer digunakan untuk menyimpan data mentah traffic dari file `Metro_Interstate_Traffic_Volume.csv`.

Tahapan pada Bronze Layer:
- Load data mentah traffic.
- Kolom waktu `date_time` diubah ke format timestamp.
- Data mentah disimpan dalam format Parquet.
- Tujuan layer ini adalah menyimpan data asli agar tetap bisa diakses kembali jika dibutuhkan.

### 2.2 Silver Layer

Silver Layer digunakan untuk membersihkan dan menyiapkan data agar siap digunakan untuk analisis dan modeling.

Tahapan pada Silver Layer:
- Menghapus nilai null atau data duplikat.
- Melakukan konversi tipe data numerik dan kategorikal.
- Melakukan feature engineering, seperti:
  - Ekstraksi jam.
  - Ekstraksi hari.
  - Penentuan weekday atau weekend.
  - Pengolahan fitur holiday.
- Melakukan agregasi traffic per jam untuk kebutuhan modeling.
- Menyimpan data hasil preprocessing ke Silver Layer dalam format Parquet.

### 2.3 Gold Layer

Gold Layer digunakan untuk proses Machine Learning, evaluasi model, prediksi, dan visualisasi hasil.

Tahapan pada Gold Layer:
- Melatih model **Random Forest Regressor** menggunakan PySpark ML.
- Input model berasal dari fitur yang sudah diproses pada Silver Layer.
- Target prediksi adalah traffic volume per jam.
- Melakukan evaluasi model menggunakan beberapa metrik.
- Menyimpan hasil prediksi ke Gold Layer pada folder `predictions_rf`.

---

## 3. Hasil Evaluasi Model

| Model | RMSE | MAE | R² | MAPE |
|------|------|-----|----|------|
| Random Forest | 817.70 | 718.61 | 0.8077 | 61.41% |

**Interpretasi:**  
Model Random Forest mampu menjelaskan sekitar **81% variasi traffic per jam** berdasarkan nilai **R² = 0.8077**. Nilai RMSE sebesar **817.70** dan MAE sebesar **718.61** menunjukkan rata-rata kesalahan prediksi masih dalam batas yang dapat dianalisis untuk data traffic yang memiliki fluktuasi tinggi. Nilai MAPE sebesar **61.41%** menunjukkan bahwa data traffic cukup dinamis, sehingga error relatif masih cukup besar, tetapi model tetap dapat digunakan untuk melihat pola dan trend prediksi.

---

## 4. Visualisasi Prediksi

![LOESS Smoothed Plot](loess_plot.png)

**Keterangan Plot:**
- Titik biru menunjukkan traffic aktual per jam.
- Garis merah menunjukkan hasil prediksi Random Forest setelah dilakukan LOESS smoothing.
- LOESS smoothing digunakan untuk memperjelas pola trend prediksi agar lebih mudah dibandingkan dengan data aktual.

---

## 5. Kesimpulan

Berdasarkan hasil pipeline yang telah dibuat, arsitektur **Bronze → Silver → Gold** berhasil digunakan untuk memproses data traffic dari data mentah hingga menjadi data siap modeling. Model **Random Forest Regressor** mampu memberikan hasil prediksi dengan nilai **R² sebesar 0.8077**, yang menunjukkan bahwa model cukup baik dalam menjelaskan pola traffic volume per jam.

Visualisasi menggunakan LOESS smoothing juga membantu menampilkan trend prediksi dengan lebih jelas. Pipeline ini masih dapat dikembangkan lebih lanjut dengan menambahkan fitur baru, menggunakan data real-time, atau membandingkan model Machine Learning lain agar hasil prediksi menjadi lebih optimal.

---
