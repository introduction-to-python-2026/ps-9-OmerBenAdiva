import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("parkinsons.csv")

# 2. Select features and target
selected_features = ["PPE", "spread1"]
X = df[selected_features]
y = df["status"].astype(int)

# 3. Scale data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split data
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Choose model
model = SVC(
    kernel="rbf",
    C=2.0,
    gamma="scale",
    class_weight="balanced",
    random_state=42
)

# 6. Train and test accuracy
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

# דרישת המטלה: דיוק לפחות 0.8
assert accuracy >= 0.8

# 7. Save model
joblib.dump(model, "my_model.joblib")
