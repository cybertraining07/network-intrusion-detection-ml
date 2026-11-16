
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# Load original Web Attack model
web_model = joblib.load("webattack_detection_model.joblib")

# Load dataset
df = pd.read_csv("crossfamily_test.csv")
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

# Keep SSH-Patator traffic
ssh = df[df["Label"] == "SSH-Patator"].copy()

# Prepare features
features = joblib.load("webattack_feature_names.joblib")

X = ssh[features].copy()
X = X.replace([np.inf, -np.inf], np.nan)

# Identify original model's false alerts
predictions = web_model.predict(X)

# Target: 0 = correctly classified, 1 = false alert
y = (predictions != "BENIGN").astype(int)

print("\nSSH FALSE-ALERT REDUCTION EXPERIMENT")
print("=" * 55)
print("Dataset size:", len(X))
print("Correctly classified:", (y == 0).sum())
print("False alerts:", (y == 1).sum())

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Train second-stage model
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=2
    ))
])

pipeline.fit(X_train, y_train)

# Evaluate
predicted = pipeline.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predicted), 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predicted,
    target_names=["Correctly Classified", "False Alert"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predicted))

# Save model
joblib.dump(
    pipeline,
    "ssh_false_alert_reduction_model.joblib"
)

print("\nModel saved successfully.")
