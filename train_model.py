import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    # 1. Baca dataset Student Performance (perhatikan ada spasi sebelum .csv sesuai nama file asli)
    file_path = "Student_performance_data _.csv"
    df = pd.read_csv(file_path)

    print("Ukuran Dataset:", df.shape)
    print("Cek Missing Values:\n", df.isnull().sum())

    # 2. Pisahkan fitur (X) dan target klasifikasi (y)
    # Target: GradeClass (0: A, 1: B, 2: C, 3: D, 4: F)
    # Drop StudentID (tidak relevan) dan GPA (karena GradeClass diturunkan langsung dari GPA)
    X = df.drop(columns=["StudentID", "GPA", "GradeClass"])
    y = df["GradeClass"].astype(int)

    # 3. Simpan daftar nama kolom fitur ke model_columns.pkl 
    model_columns = list(X.columns)
    with open("model_columns.pkl", "wb") as f:
        pickle.dump(model_columns, f)
    print(f"\nKolom fitur berhasil disimpan ke model_columns.pkl ({len(model_columns)} fitur):")
    print(model_columns)

    # 4. Bagi data menjadi Training (80%) dan Testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Latih Model Klasifikasi (Random Forest Classifier)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # 6. Evaluasi Model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAkurasi Model pada Data Uji: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Grade A", "Grade B", "Grade C", "Grade D", "Grade F"]))

    # 7. Simpan model terlatih ke model.pkl
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model berhasil disimpan ke model.pkl!")

if __name__ == "__main__":
    main()