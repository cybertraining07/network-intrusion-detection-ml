
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load PortScan test dataset
df = pd.read_csv("portscan_test.csv")

df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

# Load the existing DDoS-trained model
model = joblib.load("ddos_detection_clean_model.joblib")
feature_names = joblib.load("clean_feature_names.joblib")

# Prepare test features using the training feature order
X_test = df.reindex(columns=feature_names)

X_test = X_test.apply(pd.to_numeric, errors="coerce")
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Convert actual labels:
# BENIGN = 0
# PortScan = 1
y_test = df["Label"].map({
    "BENIGN": 0,
    "PortScan": 1
})

if y_test.isna().any():
    raise ValueError("Unexpected labels found in the test dataset.")

y_test = y_test.astype(int)

# Generate predictions
predictions = model.predict(X_test)

# Display results
print("\n================================")
print("DDOS MODEL - PORTSCAN EVALUATION")
print("================================")

print("\nTest dataset shape:", X_test.shape)

print("\nAccuracy:")
print(f"{accuracy_score(y_test, predictions) * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["BENIGN", "PortScan"],
    zero_division=0
))

cm = confusion_matrix(y_test, predictions, labels=[0, 1])

tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix:")
print(cm)

print("\nDetailed Results:")
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

fpr = (fp / (fp + tn)) * 100 if (fp + tn) else 0

print(f"\nFalse Positive Rate: {fpr:.4f}%")
print("False Negatives:", fn)

print("\nCross-attack-family evaluation completed!")
