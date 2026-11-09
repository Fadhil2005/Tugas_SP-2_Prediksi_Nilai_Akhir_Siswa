import streamlit as st
import pandas as pd
from utils.model_loader import predict_student_grade, get_model_diagnostics

st.set_page_config(
    page_title="Sistem Evaluasi & Prediksi Akademik Siswa",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS: Adaptasi Desain Editorial Jepang (Quick Teacher Style)
# Tanpa warna biru/ungu, tanpa pill berlebih, tanpa dot glow, tanpa emoji
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700;900&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg-ivory: #FDFBF7;
    --bg-sand: #F5EFE6;
    --bg-card: #FFFFFF;
    --ink-dark: #1C1917;
    --ink-muted: #57534E;
    --ink-soft: #78716C;
    --amber-main: #F59E0B;
    --amber-deep: #D97706;
    --amber-soft: #FEF3C7;
    --border-warm: #E5DDD0;
    --olive-ok: #3F6212;
    --terracotta-err: #991B1B;
}

.stApp {
    background-color: var(--bg-ivory);
    color: var(--ink-dark);
    font-family: 'Plus Jakarta Sans', 'Noto Sans JP', sans-serif;
}

/* Sembunyikan elemen dekoratif bawaan Streamlit */
header[data-testid="stHeader"] {
    background: transparent;
}
footer {
    visibility: hidden;
}
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1220px !important;
}

/* Banner Editorial Atas */
.editorial-hero {
    background-color: var(--bg-card);
    border: 1px solid var(--border-warm);
    border-top: 5px solid var(--amber-main);
    padding: 28px 32px;
    margin-bottom: 24px;
}
.hero-kicker {
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--amber-deep);
    margin-bottom: 8px;
}
.hero-title {
    font-size: 2.05rem;
    font-weight: 800;
    color: var(--ink-dark);
    line-height: 1.25;
    margin: 0 0 10px 0;
}
.hero-highlight {
    background: linear-gradient(transparent 62%, #FDE68A 62%);
    padding: 0 4px;
}
.hero-subtitle {
    font-size: 0.96rem;
    color: var(--ink-muted);
    line-height: 1.6;
    max-width: 820px;
    margin: 0;
}

/* Baris Metrik Evaluasi Model (Gaya Medali/Statistik Quick Teacher) */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 28px;
}
.stat-box {
    background-color: var(--bg-card);
    border: 1px solid var(--border-warm);
    padding: 18px 20px;
    border-left: 3px solid var(--amber-main);
}
.stat-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink-soft);
    margin-bottom: 6px;
}
.stat-value {
    font-size: 1.65rem;
    font-weight: 800;
    color: var(--ink-dark);
    line-height: 1.1;
}
.stat-sub {
    font-size: 0.78rem;
    color: var(--ink-muted);
    margin-top: 6px;
}

/* Penanda Seksi POINT 01, POINT 02, POINT 03 */
.point-header {
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-warm);
}
.point-tag {
    font-size: 0.76rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    color: var(--amber-deep);
    text-transform: uppercase;
    display: block;
    margin-bottom: 2px;
}
.point-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: var(--ink-dark);
    margin: 0;
}

/* Styling Form & Tombol Submit */
div[data-testid="stForm"] {
    background-color: var(--bg-card);
    border: 1px solid var(--border-warm);
    padding: 24px 26px !important;
    border-radius: 4px !important;
}
div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background-color: var(--ink-dark) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--ink-dark) !important;
    border-radius: 4px !important;
    padding: 0.7rem 1.4rem !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.04em !important;
    transition: all 0.15s ease-in-out !important;
}
div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: var(--amber-deep) !important;
    border-color: var(--amber-deep) !important;
    color: #FFFFFF !important;
}

/* Kartu Hasil Klasifikasi */
.result-banner {
    background-color: var(--bg-card);
    border: 1px solid var(--border-warm);
    border-left: 6px solid var(--amber-main);
    padding: 24px 28px;
    margin-top: 8px;
    margin-bottom: 20px;
}
.result-kicker {
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--amber-deep);
    margin-bottom: 4px;
}
.result-grade {
    font-size: 1.85rem;
    font-weight: 800;
    color: var(--ink-dark);
    margin-bottom: 8px;
}
.result-desc {
    font-size: 0.95rem;
    color: var(--ink-muted);
    line-height: 1.55;
    margin: 0;
}

/* Custom Horizontal Bar Tanpa Warna Biru */
.bar-row {
    margin-bottom: 12px;
}
.bar-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--ink-dark);
    margin-bottom: 4px;
}
.bar-track {
    width: 100%;
    height: 10px;
    background-color: var(--bg-sand);
    border: 1px solid var(--border-warm);
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    background-color: var(--amber-main);
}
.bar-fill-active {
    height: 100%;
    background-color: var(--ink-dark);
}

/* Panel Informasi Samping */
.side-panel {
    background-color: var(--bg-sand);
    border: 1px solid var(--border-warm);
    padding: 20px 22px;
    margin-top: 16px;
}
.side-title {
    font-size: 0.92rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--ink-dark);
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# Muat diagnostik dan metrik dari model.pkl & CSV
model_meta, feat_df, metrics = get_model_diagnostics()

# 1. HERO HEADER
st.markdown(f"""
<div class="editorial-hero">
    <div class="hero-kicker">SISTEM PENDUKUNG KEPUTUSAN AKADEMIK — RANDOM FOREST CLASSIFIER</div>
    <h1 class="hero-title">Evaluasi Terukur & <span class="hero-highlight">Prediksi Predikat Nilai Akhir</span> Siswa</h1>
    <p class="hero-subtitle">
        Sistem ini menganalisis 12 variabel demografi, kebiasaan belajar mingguan, tingkat kehadiran, 
        serta keaktifan ekstrakurikuler dari model terlatih untuk mengklasifikasikan capaian akademik 
        siswa ke dalam lima tingkat predikat nilai (Grade A hingga Grade F).
    </p>
</div>
""", unsafe_allow_html=True)

# 2. PANEL METRIK & AKURASI MODEL
st.markdown(f"""
<div class="stat-grid">
    <div class="stat-box">
        <div class="stat-label">Akurasi Evaluasi Model</div>
        <div class="stat-value">{metrics['eval_accuracy']}%</div>
        <div class="stat-sub">Rentang kalibrasi ideal (80% - 90%)</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Weighted F1-Score</div>
        <div class="stat-value">{metrics['f1_weighted']}%</div>
        <div class="stat-sub">Keseimbangan presisi & sensitivitas</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Arsitektur Ensemble</div>
        <div class="stat-value">{model_meta['n_estimators']} Pohon</div>
        <div class="stat-sub">Random Forest (Max Depth: {model_meta['max_depth']})</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Basis Data Populasi</div>
        <div class="stat-value">{metrics['total_rows']} Siswa</div>
        <div class="stat-sub">{model_meta['n_features']} variabel prediktor aktif</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Pemetaan Kategori Dataset
gender_map = {"Laki-laki": 0, "Perempuan": 1}
ethnicity_map = {
    "Kelompok Etnis A (Caucasian)": 0,
    "Kelompok Etnis B (African American)": 1,
    "Kelompok Etnis C (Asian)": 2,
    "Kelompok Etnis Lainnya": 3
}
parental_edu_map = {
    "Tidak Tamat SMA": 0,
    "SMA / Sederajat": 1,
    "Perguruan Tinggi (Diploma / Sebagian)": 2,
    "Sarjana (S1)": 3,
    "Pascasarjana (S2 / S3)": 4
}
parental_support_map = {
    "Sangat Rendah / Tidak Ada": 0,
    "Rendah": 1,
    "Sedang": 2,
    "Tinggi": 3,
    "Sangat Tinggi": 4
}
binary_map = {"Tidak Aktif / Tidak Mengikuti": 0, "Aktif / Mengikuti": 1}

grade_info = {
    0: (
        "GRADE A — SANGAT BAIK (ESTIMASI GPA 3.50 - 4.00)",
        "#3F6212",
        "Siswa berada pada kuartil performa tertinggi. Pertahankan konsistensi jam belajar mandiri dan disiplin kehadiran."
    ),
    1: (
        "GRADE B — BAIK (ESTIMASI GPA 3.00 - 3.49)",
        "#D97706",
        "Performa akademik tergolong solid. Penambahan durasi belajar 2 hingga 3 jam per minggu berpotensi mendorong capaian ke Grade A."
    ),
    2: (
        "GRADE C — CUKUP (ESTIMASI GPA 2.50 - 2.99)",
        "#B45309",
        "Performa berada di ambang rata-rata. Diperlukan pemantauan absensi serta penjadwalan ulang jam belajar mingguan."
    ),
    3: (
        "GRADE D — KURANG (ESTIMASI GPA 2.00 - 2.49)",
        "#9A3412",
        "Peringatan akademik tingkat awal. Disarankan mengikuti program bimbingan belajar (tutoring) dan menekan angka ketidakhadiran."
    ),
    4: (
        "GRADE F — RISIKO GAGAL AKADEMIK (ESTIMASI GPA DI BAWAH 2.00)",
        "#991B1B",
        "Prioritas intervensi tinggi. Siswa membutuhkan pendampingan intensif dari orang tua dan evaluasi menyeluruh atas tingkat kehadiran."
    )
}

# 3. FORMULIR INPUT PARAMETER SISWA (3 KOLOM POINT 01 - POINT 03)
with st.form("student_prediction_form"):
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("""
        <div class="point-header">
            <span class="point-tag">POINT 01</span>
            <h3 class="point-title">Profil Demografi & Keluarga</h3>
        </div>
        """, unsafe_allow_html=True)
        age = st.slider("Usia Siswa (Tahun)", min_value=15, max_value=18, value=16, step=1)
        gender_label = st.selectbox("Jenis Kelamin", options=list(gender_map.keys()))
        ethnicity_label = st.selectbox("Kategori Demografi Etnis", options=list(ethnicity_map.keys()))
        parental_edu_label = st.selectbox("Pendidikan Terakhir Orang Tua", options=list(parental_edu_map.keys()), index=2)

    with col2:
        st.markdown("""
        <div class="point-header">
            <span class="point-tag">POINT 02</span>
            <h3 class="point-title">Disiplin Belajar & Kehadiran</h3>
        </div>
        """, unsafe_allow_html=True)
        study_time = st.number_input(
            "Durasi Belajar Mandiri per Minggu (Jam)",
            min_value=0.0,
            max_value=20.0,
            value=12.0,
            step=0.5
        )
        absences = st.slider("Jumlah Ketidakhadiran / Absen (Hari)", min_value=0, max_value=30, value=4, step=1)
        tutoring_label = st.selectbox("Program Bimbingan Belajar (Tutoring)", options=list(binary_map.keys()), index=1)
        parental_support_label = st.selectbox("Intensitas Dukungan Orang Tua", options=list(parental_support_map.keys()), index=3)

    with col3:
        st.markdown("""
        <div class="point-header">
            <span class="point-tag">POINT 03</span>
            <h3 class="point-title">Aktivitas Ekstrakurikuler</h3>
        </div>
        """, unsafe_allow_html=True)
        extracurricular_label = st.selectbox("Organisasi / Ekstrakurikuler Umum", options=list(binary_map.keys()))
        sports_label = st.selectbox("Kegiatan Olahraga (Sports)", options=list(binary_map.keys()))
        music_label = st.selectbox("Kegiatan Musik & Seni", options=list(binary_map.keys()))
        volunteering_label = st.selectbox("Kegiatan Sosial (Volunteering)", options=list(binary_map.keys()))

    st.markdown("<br>", unsafe_allow_html=True)
    submit_btn = st.form_submit_button("JALANKAN EVALUASI & PREDIKSI PREDIKAT NILAI", use_container_width=True)

# 4. HASIL PREDIKSI & ANALISIS MENDALAM DARI MODEL.PKL
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
        label_teks, border_hex, rekomendasi = grade_info[pred_class]

        st.markdown(f"""
        <div class="result-banner" style="border-left-color: {border_hex};">
            <div class="result-kicker">HASIL KLASIFIKASI MODEL RANDOM FOREST</div>
            <div class="result-grade">{label_teks}</div>
            <p class="result-desc"><strong>Rekomendasi Evaluasi:</strong> {rekomendasi}</p>
        </div>
        """, unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1.15, 1], gap="large")

        with res_col1:
            st.markdown("""
            <div class="point-header">
                <span class="point-tag">PROBABILITAS KELAS</span>
                <h3 class="point-title">Distribusi Keyakinan Prediksi Model</h3>
            </div>
            """, unsafe_allow_html=True)

            grade_names = [
                "Grade A (Sangat Baik)",
                "Grade B (Baik)",
                "Grade C (Cukup)",
                "Grade D (Kurang)",
                "Grade F (Gagal)"
            ]

            bars_html = ""
            for idx, (g_name, p_val) in enumerate(zip(grade_names, probs)):
                pct = round(float(p_val) * 100, 2)
                fill_class = "bar-fill-active" if idx == pred_class else "bar-fill"
                bars_html += f"""
                <div class="bar-row">
                    <div class="bar-meta">
                        <span>{g_name}</span>
                        <span>{pct:.2f}%</span>
                    </div>
                    <div class="bar-track">
                        <div class="{fill_class}" style="width: {max(pct, 1.5)}%;"></div>
                    </div>
                </div>
                """
            st.markdown(bars_html, unsafe_allow_html=True)

            # Perbandingan Input Siswa dengan Rata-Rata Populasi Dataset
            selisih_belajar = round(study_time - metrics["avg_study_time"], 2)
            selisih_absen = round(absences - metrics["avg_absences"], 2)
            status_belajar = f"+{selisih_belajar} jam di atas rata-rata" if selisih_belajar >= 0 else f"{selisih_belajar} jam di bawah rata-rata"
            status_absen = f"{abs(selisih_absen)} hari lebih sedikit dari rata-rata" if selisih_absen <= 0 else f"+{selisih_absen} hari lebih banyak dari rata-rata"

            st.markdown(f"""
            <div class="side-panel">
                <div class="side-title">Perbandingan Terhadap Rata-Rata Dataset ({metrics['total_rows']} Siswa)</div>
                <p style="margin: 0 0 6px 0; font-size: 0.88rem; color: #1C1917;">
                    <strong>Jam Belajar Siswa:</strong> {study_time} jam/minggu 
                    <span style="color: #57534E;">(Rata-rata populasi: {metrics['avg_study_time']} jam — {status_belajar})</span>
                </p>
                <p style="margin: 0; font-size: 0.88rem; color: #1C1917;">
                    <strong>Ketidakhadiran Siswa:</strong> {absences} hari 
                    <span style="color: #57534E;">(Rata-rata populasi: {metrics['avg_absences']} hari — {status_absen})</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("""
            <div class="point-header">
                <span class="point-tag">BOBOT FITUR MODEL.PKL</span>
                <h3 class="point-title">Faktor Penentu Utama di Dalam Model</h3>
            </div>
            """, unsafe_allow_html=True)

            top_features = feat_df.head(6)
            feat_html = ""
            for _, row in top_features.iterrows():
                f_name = row["Nama Fitur"]
                f_pct = row["Kontribusi (%)"]
                # Skala visual relatif terhadap fitur tertinggi (36.06%)
                bar_w = min(round((f_pct / 40.0) * 100, 1), 100)
                feat_html += f"""
                <div class="bar-row">
                    <div class="bar-meta">
                        <span>{f_name}</span>
                        <span>{f_pct:.2f}%</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: {bar_w}%;"></div>
                    </div>
                </div>
                """
            st.markdown(feat_html, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="side-panel">
                <div class="side-title">Ringkasan Validasi Model</div>
                <p style="margin: 0; font-size: 0.86rem; color: #57534E; line-height: 1.55;">
                    Berdasarkan struktur pohon keputusan pada <code>model.pkl</code>, variabel 
                    <strong>Jumlah Ketidakhadiran (36.06%)</strong> dan <strong>Waktu Belajar Mingguan (18.65%)</strong> 
                    memegang lebih dari separuh bobot keputusan prediksi. Akurasi evaluasi model dijaga pada 
                    <strong>{metrics['eval_accuracy']}%</strong> untuk memastikan kemampuan generalisasi yang objektif.
                </p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")