import streamlit as st
import pandas as pd
from utils.model_loader import predict_student_grade

# Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Kinerja Akademik Siswa",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Sistem Klasifikasi Predikat Nilai Siswa (Student Performance)")
st.markdown(
    "Aplikasi ini memprediksi kategori nilai akhir siswa (**Grade A, B, C, D, atau F**) "
    "berdasarkan data demografi, kebiasaan belajar, kehadiran, dan aktivitas ekstrakurikuler."
)
st.divider()

# Pemetaan opsi kategori sesuai dokumentasi dataset Student_performance_data _.csv
gender_map = {"Laki-laki (Male)": 0, "Perempuan (Female)": 1}
ethnicity_map = {
    "Caucasian": 0,
    "African American": 1,
    "Asian": 2,
    "Lainnya (Other)": 3
}
parental_edu_map = {
    "Tidak Ada (None)": 0,
    "SMA / Sederajat (High School)": 1,
    "Perguruan Tinggi Sebagian (Some College)": 2,
    "Sarjana / S1 (Bachelor's)": 3,
    "Pascasarjana / S2-S3 (Higher)": 4
}
parental_support_map = {
    "Tidak Ada (None)": 0,
    "Rendah (Low)": 1,
    "Sedang (Moderate)": 2,
    "Tinggi (High)": 3,
    "Sangat Tinggi (Very High)": 4
}
binary_map = {"Tidak": 0, "Ya": 1}

grade_info = {
    0: ("Grade A (Sangat Baik / GPA ≥ 3.5)", "success", "Pertahankan kebiasaan belajar dan tingkat kehadiran yang sangat baik ini!"),
    1: ("Grade B (Baik / 3.0 ≤ GPA < 3.5)", "info", "Performa sudah baik. Sedikit tambahan jam belajar mingguan dapat mendorong nilai ke Grade A."),
    2: ("Grade C (Cukup / 2.5 ≤ GPA < 3.0)", "warning", "Performa cukup, namun perlu peningkatan jam belajar mandiri dan pengurangan jumlah absen."),
    3: ("Grade D (Kurang / 2.0 ≤ GPA < 2.5)", "warning", "Perhatian khusus diperlukan. Disarankan mengikuti bimbingan belajar (tutoring) dan meningkatkan kehadiran."),
    4: ("Grade F (Gagal / GPA < 2.0)", "error", "Risiko tinggi gagal akademik! Perlu evaluasi menyeluruh terhadap kehadiran, jam belajar, dan dukungan orang tua.")
}

# Form Input Pengguna
with st.form("student_prediction_form"):
    st.subheader("📋 Masukkan Data Siswa")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**1. Data Demografi & Keluarga**")
        age = st.slider("Usia Siswa (Tahun)", min_value=15, max_value=18, value=16, step=1)
        gender_label = st.selectbox("Jenis Kelamin", options=list(gender_map.keys()))
        ethnicity_label = st.selectbox("Kategori Etnis", options=list(ethnicity_map.keys()))
        parental_edu_label = st.selectbox("Pendidikan Orang Tua", options=list(parental_edu_map.keys()), index=2)

    with col2:
        st.markdown("**2. Kebiasaan Belajar & Dukungan**")
        study_time = st.number_input(
            "Waktu Belajar per Minggu (Jam)",
            min_value=0.0,
            max_value=20.0,
            value=10.0,
            step=0.5
        )
        absences = st.slider("Jumlah Ketidakhadiran / Absen (Hari)", min_value=0, max_value=30, value=3, step=1)
        tutoring_label = st.selectbox("Mengikuti Bimbingan Belajar (Tutoring)?", options=list(binary_map.keys()))
        parental_support_label = st.selectbox("Tingkat Dukungan Orang Tua", options=list(parental_support_map.keys()), index=2)

    with col3:
        st.markdown("**3. Kegiatan Ekstrakurikuler**")
        extracurricular_label = st.selectbox("Aktif Ekstrakurikuler Umum?", options=list(binary_map.keys()))
        sports_label = st.selectbox("Aktif Kegiatan Olahraga (Sports)?", options=list(binary_map.keys()))
        music_label = st.selectbox("Aktif Kegiatan Musik (Music)?", options=list(binary_map.keys()))
        volunteering_label = st.selectbox("Aktif Kegiatan Sukarelawan (Volunteering)?", options=list(binary_map.keys()))

    submit_btn = st.form_submit_button("🔍 Prediksi Predikat Nilai (GradeClass)", use_container_width=True)

# Proses Prediksi Ketika Tombol Ditekan
if submit_btn:
    input_payload = {
        "Age": age,
        "Gender": gender_map[gender_label],
        "Ethnicity": ethnicity_map[ethnicity_label],
        "ParentalEducation": parental_edu_map[parental_edu_label],
        "StudyTimeWeekly": study_time,
        "Absences": absences,
        "Tutoring": binary_map[tutoring_label],
        "ParentalSupport": parental_support_map[parental_support_label],
        "Extracurricular": binary_map[extracurricular_label],
        "Sports": binary_map[sports_label],
        "Music": binary_map[music_label],
        "Volunteering": binary_map[volunteering_label]
    }

    try:
        pred_class, probs = predict_student_grade(input_payload)
        label_teks, status_color, rekomendasi = grade_info[pred_class]

        st.divider()
        st.subheader("📊 Hasil Prediksi Klasifikasi")

        if status_color == "success":
            st.success(f"**Hasil Prediksi:** {label_teks}")
        elif status_color == "info":
            st.info(f"**Hasil Prediksi:** {label_teks}")
        elif status_color == "warning":
            st.warning(f"**Hasil Prediksi:** {label_teks}")
        else:
            st.error(f"**Hasil Prediksi:** {label_teks}")

        st.write(f"**Rekomendasi Evaluasi:** {rekomendasi}")

        # Tampilkan Grafik Probabilitas Tiap Kelas
        st.markdown("#### Distribusi Probabilitas Prediksi:")
        prob_df = pd.DataFrame({
            "Kategori Nilai": ["Grade A", "Grade B", "Grade C", "Grade D", "Grade F"],
            "Probabilitas (%)": [round(p * 100, 2) for p in probs]
        })
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")