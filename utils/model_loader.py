import pickle
import os
import pandas as pd

MODEL_PATH = "model.pkl"
COLUMNS_PATH = "model_columns.pkl"

def load_artifacts():
    """Memuat model.pkl dan model_columns.pkl"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
        raise FileNotFoundError(
            "File 'model.pkl' atau 'model_columns.pkl' belum ditemukan. "
            "Jalankan 'python train_model.py' terlebih dahulu di terminal!"
        )
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        
    with open(COLUMNS_PATH, "rb") as f:
        model_columns = pickle.load(f)
        
    return model, model_columns

def predict_student_grade(input_data: dict):
    """
    Melakukan prediksi GradeClass berdasarkan dictionary input dari form Streamlit.
    Mengembalikan kelas prediksi (int) dan probabilitas tiap kelas.
    """
    model, model_columns = load_artifacts()
    
    # Ubah dictionary input menjadi DataFrame 1 baris
    df_input = pd.DataFrame([input_data])
    
    # Samakan urutan dan keberadaan kolom sesuai model_columns.pkl
    df_input = df_input.reindex(columns=model_columns, fill_value=0)
    
    # Prediksi kelas dan probabilitasnya
    prediction = int(model.predict(df_input)[0])
    probabilities = model.predict_proba(df_input)[0]
    
    return prediction, probabilities