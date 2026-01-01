# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit



import pandas as pd
csv_path = "/content/parkinsons.csv"
df = pd.read_csv(csv_path)
print(df.shape)      # כמה שורות ועמודות
df.head()            # מציג את 5 השורות הראשונות


# 2) Select features
selected_features = ["PPE", "spread1"] 
target_col = "status"                  
X = df[selected_features].copy()
y = df[target_col].astype(int).copy()
print("Features shape:", X.shape)
print("Target distribution:", y.value_counts())
X.head()


# 3) Scale the data to [0, 1]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  
import numpy as np
print("Min per feature:", X_scaled.min(axis=0))
print("Max per feature:", X_scaled.max(axis=0))


# 4) Split the data into Train and Validation sets
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X_scaled, y,
    test_size=0.2,        # 20% ל-Validation
    random_state=42,      # כדי לקבל תוצאה שחוזרת על עצמה
    stratify=y            # לשמור על יחס הכיתות
)

print("Train shape:", X_train.shape, y_train.shape)
print("Validation shape:", X_valid.shape, y_valid.shape)
print("Class distribution in Train:", y_train.value_counts(normalize=True))
print("Class distribution in Validation:", y_valid.value_counts(normalize=True))


# 5) Choose a model
from sklearn.svm import SVC
# נבחר SVM עם kernel RBF (לא ליניארי)
model = SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", random_state=42)
print(model)


# 6) Train & Evaluate the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
acc = accuracy_score(y_valid, y_pred)
print(f"Accuracy (validation): {acc:.3f}")
print("\nClassification report:\n", classification_report(y_valid, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_valid, y_pred))

import joblib
joblib.dump(model, 'my_model.joblib')

import joblib
import yaml

bundle = {
    "model": model,
    "scaler": scaler,
    "features": selected_features
}

joblib.dump(bundle, "my_model.joblib")
print("Saved model bundle to: my_model.joblib")

# עדכון config.yaml
config = {
    "selected_features": selected_features,
    "path": "my_model.joblib"
}

with open("config.yaml", "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
print("config.yaml written.")



