
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load Thursday dataset
df = pd.read_csv("thursday_test.csv")

df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

# Load seven-class model
model = joblib.load("seven_class_intrusion_model.joblib")
feature_names = joblib.load("seven_class_feature_names.joblib")

# Prepare features
X_test = df.reindex(columns=feature_names)

X_test = X_test.apply(
    pd.to_numeric,
    errors="coerce"
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)

# Actual labels:
# BENIGN = 0
# All Web Attacks = 1
y_test = (
    df["Label"] != "BENIGN"
).astype(int)

# Predict
predictions = model.predict(X_test)

# Convert model predictions to binary
predicted_binary = (
    predictions != "BENIGN"
).astype(int)

# Overall evaluation
print("Thursday Cross-Dataset Evaluation")
print("=================================")

print("\nAccuracy:")
print(
    f"{accuracy_score(y_test, predicted_binary) * 100:.2f}%"
)

print("\nClassification Report:")
print(classification_report(
    y_test,
    predicted_binary,
    labels=[0, 1],
    target_names=["BENIGN", "ATTACK"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    predicted_binary,
    labels=[0, 1]
))

# Original model predictions
print("\nOriginal Seven-Class Model Predictions:")
print(pd.Series(predictions).value_counts())

# Detection rate for each Web Attack family
print("\nDetection Rate by Attack Family:")

for label in df["Label"].unique():

    if label == "BENIGN":
        continue

    mask = df["Label"] == label

    detected = predicted_binary[mask].sum()
    total = mask.sum()

    rate = detected / total * 100

    print(
        f"{label}: {detected}/{total} "
        f"({rate:.2f}%)"
    )

print("\nThursday evaluation completed!")
