import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("Student_performance_data _.csv")

feature_cols = [
    "Age", "Gender", "Ethnicity", "ParentalEducation",
    "StudyTimeWeekly", "Absences", "Tutoring", "ParentalSupport",
    "Extracurricular", "Sports", "Music", "Volunteering"
]

X = df[feature_cols]
y = df["GradeClass"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# max_depth=8 menjaga akurasi berada tepat di rentang 80% - 90% (tidak overfitting 100%)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

print(f"Akurasi Train Set : {accuracy_score(y_train, model.predict(X_train)) * 100:.2f}%")
print(f"Akurasi Test Set  : {accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%")
print(f"Akurasi Populasi  : {accuracy_score(y, model.predict(X)) * 100:.2f}%")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model_columns.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

print("Berhasil memperbarui model.pkl dan model_columns.pkl.")