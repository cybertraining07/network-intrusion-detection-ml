
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report

# Load original Web Attack model
model = joblib.load("webattack_detection_model.joblib")
features = joblib.load("webattack_feature_names.joblib")

# Load FTP test dataset
df = pd.read_csv("ftp_crossfamily_test.csv")
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

X = df[features].copy()
X = X.replace([np.inf, -np.inf], np.nan)

# Original model predictions
predictions = model.predict(X)

# Convert into binary evaluation:
# BENIGN = 0
# FTP-Patator = 1
# Any Web Attack prediction = alert

actual = (df["Label"] != "BENIGN").astype(int)
predicted = (predictions != "BENIGN").astype(int)

print("\nFTP CROSS-FAMILY EVALUATION")
print("=" * 55)

print("Total records:", len(df))
print("\nActual labels:")
print(df["Label"].value_counts())

print("\nOriginal model prediction distribution:")
print(pd.Series(predictions).value_counts())

print("\nConfusion Matrix:")
print(confusion_matrix(actual, predicted))

print("\nClassification Report:")
print(classification_report(
    actual,
    predicted,
    target_names=["BENIGN", "FTP-Patator"],
    zero_division=0
))

# False alert rate on benign records
benign_mask = df["Label"] == "BENIGN"
false_alerts = (
    (predictions != "BENIGN") & benign_mask
).sum()

print("\nBENIGN FALSE ALERTS:", false_alerts)
print(
    "BENIGN FALSE ALERT RATE:",
    round(false_alerts / benign_mask.sum() * 100, 2),
    "%"
)

# Alert rate on FTP-Patator records
ftp_mask = df["Label"] == "FTP-Patator"
ftp_alerts = (
    (predictions != "BENIGN") & ftp_mask
).sum()

print("\nFTP-PATATOR ALERTS:", ftp_alerts)
print(
    "FTP-PATATOR ALERT RATE:",
    round(ftp_alerts / ftp_mask.sum() * 100, 2),
    "%"
)
