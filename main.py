import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load data
df = pd.read_csv("parkinsons.csv")

# Features must match config.yaml
features = ["PPE", "spread1"]
X = df[features]
y = df["status"].astype(int)

# Scale
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model
model = SVC(
    kernel="rbf",
    C=2.0,
    gamma="scale",
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Save model ONLY
joblib.dump(model, "my_model.joblib")
