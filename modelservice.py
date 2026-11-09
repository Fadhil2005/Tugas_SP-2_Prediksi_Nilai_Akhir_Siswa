"""Logika model (tanpa Streamlit) - bisa dipakai FastAPI / Flask / script apa pun."""
import pickle
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent
CSV_PATH = next(BASE.glob("Student_performance_data*.csv"))

LABELS = {
    "Age": "Usia siswa", "Gender": "Jenis kelamin", "Ethnicity": "Kelompok etnis",
    "ParentalEducation": "Pendidikan orang tua", "StudyTimeWeekly": "Waktu belajar mingguan",
    "Absences": "Jumlah absen", "Tutoring": "Bimbingan belajar", "ParentalSupport": "Dukungan orang tua",
    "Extracurricular": "Ekstrakurikuler", "Sports": "Olahraga", "Music": "Musik", "Volunteering": "Kegiatan sosial",
}


@lru_cache(maxsize=1)
def load_model():
    with open(BASE / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE / "model_columns.pkl", "rb") as f:
        cols = pickle.load(f)
    return model, cols


def predict(payload: dict):
    model, cols = load_model()
    df = pd.DataFrame([payload]).reindex(columns=cols, fill_value=0)
    return int(model.predict(df)[0]), [float(p) for p in model.predict_proba(df)[0]]


@lru_cache(maxsize=1)
def info():
    model, cols = load_model()
    df = pd.read_csv(CSV_PATH)
    X, y = df[cols], df["GradeClass"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pred_te = model.predict(X_te)
    imp = sorted(zip(cols, model.feature_importances_), key=lambda t: -t[1])
    return {
        "n_trees": int(model.get_params()["n_estimators"]),
        "total_rows": int(len(df)),
        "train_acc": round(accuracy_score(y_tr, model.predict(X_tr)) * 100, 1),
        "test_acc": round(accuracy_score(y_te, pred_te) * 100, 1),
        "test_f1": round(f1_score(y_te, pred_te, average="weighted") * 100, 1),
        "avg_study": round(float(df["StudyTimeWeekly"].mean()), 1),
        "avg_absences": round(float(df["Absences"].mean()), 1),
        "importances": [{"name": LABELS[c], "pct": round(float(v) * 100, 1)} for c, v in imp[:6]],
    }