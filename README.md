# Project-ABD: Traffic Volume Prediction Pipeline

## 1. Deskripsi Proyek
Proyek ini bertujuan memprediksi **Traffic Volume per jam** menggunakan pipeline ETL multi-layer (Bronze → Silver → Gold) dan model Machine Learning (Random Forest Regressor).  
Pipeline ini memproses data mentah traffic, membersihkan, menyiapkan fitur, dan memodelkan agar prediksi traffic per jam akurat.  

**Tujuan Utama:**
- Menyediakan prediksi traffic yang membantu manajemen jalan tol dan transportasi.
- Memvisualisasikan trend prediksi agar lebih mudah dipahami.
- Menyusun pipeline yang modular: Bronze → Silver → Gold untuk proses ETL & ML.

---

## 2. Pipeline Data

### 2.1 Bronze Layer
- Load data mentah traffic (`Metro_Interstate_Traffic_Volume.csv`)
- Kolom waktu (`date_time`) diubah ke format timestamp
- Simpan data mentah sebagai Parquet untuk efisiensi dan konsistensi
- Tujuan: menyimpan data asli apa adanya agar bisa diakses ulang

### 2.2 Silver Layer
- Preprocessing:
  - Menghapus nilai null atau duplikat
  - Konversi tipe data numerik dan kategorikal
- Feature Engineering:
  - Ekstraksi jam, hari, weekday/weekend, holiday
  - Normalisasi fitur numeric jika perlu
- Agregasi traffic per jam untuk modeling ML
- Simpan data processed sebagai Parquet di Silver Layer

### 2.3 Gold Layer
- Pelatihan Model ML:
  - Random Forest Regressor (PySpark ML)
  - Input: fitur dari Silver Layer
  - Target: traffic volume per jam
- Evaluasi Model:
  - Root Mean Square Error (RMSE)
  - Mean Absolute Error (MAE)
  - R² (Koefisien Determinasi)
  - Mean Absolute Percentage Error (MAPE)
- Prediksi & Visualisasi:
  - LOESS smoothing untuk menampilkan trend prediksi
  - Hasil disimpan di Gold Layer (`predictions_rf`)

---

## 3. Hasil Evaluasi Model

| Model          | RMSE    | MAE    | R²      | MAPE    |
|----------------|---------|--------|---------|---------|
| Random Forest  | 817.70  | 718.61 | 0.8077  | 61.41%  |

**Interpretasi:**  
- Model menjelaskan **~81% variasi traffic per jam (R²=0.8077)**  
- **MAPE 61.41%** menunjukkan error relatif yang wajar untuk traffic dengan fluktuasi tinggi  
- Model konsisten dan siap digunakan untuk prediksi real-time atau batch

---

## 4. Visualisasi Prediksi

![LOESS Smoothed Plot](loess_plot.png)

**Keterangan Plot:**  
- Titik biru: traffic aktual per jam  
- Garis merah: prediksi Random Forest setelah LOESS smoothing  
- LOESS membantu menampilkan pola trend prediksi yang mengikuti data aktual

---

## 5. Kesimpulan
- **Random Forest** adalah model terbaik untuk pipeline ini.  
- Pipeline **Bronze → Silver → Gold** berhasil menyiapkan data bersih, terstruktur, dan siap ML.  
- LOESS smoothing memudahkan visualisasi trend dan evaluasi model.  
- Pipeline ini bisa dikembangkan untuk data real-time, fitur tambahan, atau model lain.

---
