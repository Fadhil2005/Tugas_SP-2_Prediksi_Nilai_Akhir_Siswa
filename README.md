# 🎓 Student Performance Classification

Aplikasi web berbasis **Machine Learning** untuk melakukan klasifikasi performa siswa berdasarkan data yang tersedia. Aplikasi ini dibangun menggunakan Python dan Streamlit dengan model Machine Learning yang telah dilatih sebelumnya.

## 📌 Deskripsi Proyek

Proyek ini bertujuan untuk mengimplementasikan Machine Learning dalam proses klasifikasi data siswa melalui antarmuka web yang interaktif.

Pengguna dapat memasukkan data siswa melalui aplikasi, kemudian sistem akan memproses data tersebut menggunakan model yang telah dilatih untuk menghasilkan prediksi klasifikasi.

## 🚀 Teknologi yang Digunakan

- **Python** – Bahasa pemrograman utama.
- **Streamlit** – Framework untuk membangun antarmuka aplikasi web.
- **Pandas** – Pengolahan dan manipulasi data.
- **Scikit-learn** – Pengembangan dan implementasi model Machine Learning.
- **Pickle** – Penyimpanan dan pemuatan model Machine Learning.

## 📂 Struktur Proyek

```text
klasifikasi_siswa/
│
├── utils/
│   └── model_loader.py
│
├── Student_performance_data_.csv
├── train_model.py
├── model.pkl
├── model_columns.pkl
├── app.py
└── requirements.txt
```

### Keterangan File

| File / Folder | Deskripsi |
|---|---|
| `utils/model_loader.py` | Modul untuk memuat model dan melakukan prediksi. |
| `Student_performance_data_.csv` | Dataset utama yang digunakan dalam proyek. |
| `train_model.py` | Script untuk melatih ulang model Machine Learning. |
| `model.pkl` | File model Machine Learning yang telah dilatih. |
| `model_columns.pkl` | File yang menyimpan informasi kolom fitur model. |
| `app.py` | File utama aplikasi Streamlit. |
| `requirements.txt` | Daftar library Python yang dibutuhkan. |

## ⚙️ Instalasi dan Persiapan

### 1. Clone Repository

Clone repository ini ke komputer lokal:

```bash
git clone https://github.com/USERNAME/REPOSITORY.git
```

Masuk ke direktori proyek:

```bash
cd klasifikasi_siswa
```

### 2. Buka Proyek di VS Code

Buka folder proyek menggunakan Visual Studio Code melalui menu **File → Open Folder**.

### 3. Buat dan Aktifkan Virtual Environment

Buat virtual environment untuk mengisolasi dependensi proyek.

**Windows (Command Prompt / PowerShell):**

```bash
python -m venv venv
.\venv\Scripts\activate
```

**Windows (Git Bash):**

```bash
python -m venv venv
source venv/Scripts/activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependensi

Pastikan virtual environment sudah aktif, kemudian jalankan:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Jalankan Aplikasi

Jalankan aplikasi menggunakan perintah:

```bash
python -m streamlit run app.py
```

Aplikasi akan tersedia melalui browser pada alamat:

```text
http://localhost:8501
```

## 🧠 Model Machine Learning

Proyek ini menggunakan model Machine Learning yang telah dilatih dan disimpan dalam format `.pkl`.

File model yang digunakan:

- `model.pkl`
- `model_columns.pkl`

Kedua file tersebut sudah tersedia di repository sehingga pengguna tidak perlu melakukan proses training ulang untuk menjalankan aplikasi.

Apabila ingin melatih ulang model, jalankan:

```bash
python train_model.py
```

## 📋 Requirements

Dependensi utama proyek:

```text
streamlit>=1.35
pandas>=2.0
scikit-learn==1.6.1
```
