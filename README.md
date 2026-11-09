# Sistem Evaluasi dan Prediksi Predikat Nilai Akhir Siswa

Aplikasi web berbasis **Streamlit** dan **Scikit-Learn** untuk mengklasifikasikan predikat nilai akhir akademik siswa (`GradeClass`: Grade A, B, C, D, hingga F) menggunakan algoritma **Random Forest Classifier**. Antarmuka dirancang menggunakan pendekatan visual *Japanese Editorial Warm* yang bersih, terstruktur, dan fokus pada keterbacaan data.

---

## Fitur Utama Aplikasi

1. **Prediksi Predikat Akademik Multi-Kelas**: Mengklasifikasikan profil siswa ke dalam lima tingkat capaian akademik (Grade A hingga Grade F) beserta rekomendasi evaluasi yang spesifik.
2. **Distribusi Probabilitas Prediksi**: Menampilkan persentase keyakinan model (*confidence distribution*) pada seluruh kelas target secara transparan.
3. **Diagnostik Bobot Fitur (`Feature Importances`)**: Mengekstrak dan memvisualisasikan kontribusi setiap variabel prediktor langsung dari struktur pohon keputusan `model.pkl`.
4. **Komparasi Populasi Real-Time**: Membandingkan durasi belajar mingguan dan tingkat kehadiran siswa yang diinput terhadap nilai rata-rata populasi pada dataset (2.392 siswa).
5. **Panel Metrik Evaluasi Model**: Menampilkan akurasi evaluasi model, *Weighted F1-Score*, jumlah pohon keputusan (*estimators*), dan total sampel data.

---

## Struktur Direktori

```text
└── Tugas_SP-2_Prediksi_Nilai_Akhir_Siswa/
    ├── utils/
    │   └── model_loader.py              # Modul pemuat artefak model.pkl, prediksi, dan ekstraksi metrik
    ├── .gitignore                       # Konfigurasi pengecualian file Git
    ├── app.py                           # Aplikasi utama Streamlit dan konfigurasi antarmuka (UI/UX)
    ├── train_model.py                   # Skrip pelatihan dan kalibrasi parameter Random Forest
    ├── model_columns.pkl                # Artefak daftar 12 kolom fitur input
    ├── model.pkl                        # Artefak model terlatih (RandomForestClassifier)
    ├── README.md                        # Dokumentasi proyek
    ├── requirements.txt                 # Daftar dependensi pustaka Python
    └── Student_performance_data _.csv   # Dataset performa akademik siswa (2.392 baris)
```

---

## Spesifikasi Dataset dan Variabel Fitur

Dataset yang digunakan berasal dari [Students Performance Dataset (Kaggle)](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset) dengan total **2.392 data siswa**. Untuk mencegah kebocoran data (*data leakage*), atribut `StudentID` dan `GPA` tidak diikutsertakan sebagai fitur input, sehingga model menggunakan **12 fitur murni** yang tersimpan di dalam `model_columns.pkl`:

| Nama Fitur | Kategori | Deskripsi Variabel | Rentang / Nilai |
| :--- | :--- | :--- | :--- |
| `Age` | Demografi | Usia siswa | 15 hingga 18 tahun |
| `Gender` | Demografi | Jenis kelamin siswa | `0`: Laki-laki, `1`: Perempuan |
| `Ethnicity` | Demografi | Kategori kelompok etnis | `0` hingga `3` |
| `ParentalEducation` | Keluarga | Tingkat pendidikan terakhir orang tua | `0` (Tidak Tamat SMA) hingga `4` (Pascasarjana) |
| `StudyTimeWeekly` | Akademik | Durasi belajar mandiri per minggu | 0.0 hingga 20.0 jam |
| `Absences` | Akademik | Jumlah ketidakhadiran selama satu tahun ajaran | 0 hingga 30 hari |
| `Tutoring` | Akademik | Keikutsertaan dalam program bimbingan belajar | `0`: Tidak, `1`: Ya |
| `ParentalSupport` | Keluarga | Tingkat dukungan akademik dari orang tua | `0` (Tidak Ada) hingga `4` (Sangat Tinggi) |
| `Extracurricular` | Aktivitas | Keaktifan dalam kegiatan ekstrakurikuler umum | `0`: Tidak, `1`: Ya |
| `Sports` | Aktivitas | Keaktifan dalam kegiatan olahraga | `0`: Tidak, `1`: Ya |
| `Music` | Aktivitas | Keaktifan dalam kegiatan musik dan seni | `0`: Tidak, `1`: Ya |
| `Volunteering` | Aktivitas | Keaktifan dalam kegiatan sosial / sukarelawan | `0`: Tidak, `1`: Ya |

### Kelas Target (`GradeClass`)
* **Kelas 0 — Grade A**: Sangat Baik (Estimasi GPA >= 3.50)
* **Kelas 1 — Grade B**: Baik (Estimasi 3.00 <= GPA < 3.50)
* **Kelas 2 — Grade C**: Cukup (Estimasi 2.50 <= GPA < 3.00)
* **Kelas 3 — Grade D**: Kurang (Estimasi 2.00 <= GPA < 2.50)
* **Kelas 4 — Grade F**: Risiko Gagal Akademik (Estimasi GPA < 2.00)

---

## Arsitektur dan Evaluasi Model

Model dikembangkan menggunakan algoritma **Random Forest Classifier** (`scikit-learn==1.6.1`) dengan konfigurasi:
* **Jumlah Pohon Keputusan (`n_estimators`)**: 200
* **Pembobotan Kelas (`class_weight`)**: `balanced` (menangani ketimpangan jumlah sampel antar kelas)
* **Kedalaman Maksimum (`max_depth`)**: 8 (dikalibrasi untuk mencegah *overfitting* dan menjaga akurasi evaluasi pada rentang ideal 80% - 90%, tepatnya **85.74%** pada populasi dan **89.49%** pada data latih)
* **Tiga Faktor Penentu Utama (`Feature Importances`)**:
  1. `Absences` (Jumlah Ketidakhadiran): **36.06%**
  2. `StudyTimeWeekly` (Waktu Belajar Mingguan): **18.65%**
  3. `ParentalSupport` (Dukungan Orang Tua): **8.40%**

---

## Panduan Instalasi dan Menjalankan Aplikasi

1. **Clone repositori atau buka direktori proyek:**
   ```bash
   cd Tugas_SP-2_Prediksi_Nilai_Akhir_Siswa
   ```

2. **Buat dan aktifkan *virtual environment* (opsional namun disarankan):**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instal seluruh pustaka yang dibutuhkan:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi Streamlit:**
   ```bash
   streamlit run app.py
   ```
   Jika perintah `streamlit` belum terdaftar pada *PATH* sistem, gunakan:
   ```bash
   python -m streamlit run app.py
   ```

5. **Akses di Browser:**
   Buka alamat `http://localhost:8501` pada peramban web.