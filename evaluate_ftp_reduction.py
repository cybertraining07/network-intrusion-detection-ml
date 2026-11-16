
import joblib
import numpy as np
import pandas as pd

# Load models
original_model = joblib.load("webattack_detection_model.joblib")
reduction_model = joblib.load("ssh_false_alert_reduction_model.joblib")

features = joblib.load("webattack_feature_names.joblib")

# Load FTP dataset
df = pd.read_csv("ftp_crossfamily_test.csv")
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

X = df[features].copy()
X = X.replace([np.inf, -np.inf], np.nan)

# First-stage predictions
original_predictions = original_model.predict(X)

# Second-stage predictions
reduction_predictions = reduction_model.predict(X)

# Class 1 means False Alert
original_alerts = original_predictions != "BENIGN"
second_stage_alerts = reduction_predictions == 1

benign_mask = df["Label"] == "BENIGN"
ftp_mask = df["Label"] == "FTP-Patator"

print("\nCORRECTED FTP SECOND-STAGE EVALUATION")
print("=" * 55)

print("Total records:", len(df))

print("\nOriginal Web Attack Model")
print("Total alerts:", original_alerts.sum())
print("Benign alerts:", original_alerts[benign_mask].sum())
print("FTP alerts:", original_alerts[ftp_mask].sum())

print("\nSecond-Stage Model")
print("Total alerts:", second_stage_alerts.sum())
print("Benign alerts:", second_stage_alerts[benign_mask].sum())
print("FTP alerts:", second_stage_alerts[ftp_mask].sum())

print("\nAlert Rates")

print(
    "Original benign alert rate:",
    round(original_alerts[benign_mask].mean() * 100, 2), "%"
)

print(
    "Second-stage benign alert rate:",
    round(second_stage_alerts[benign_mask].mean() * 100, 2), "%"
)

print(
    "Original FTP alert rate:",
    round(original_alerts[ftp_mask].mean() * 100, 2), "%"
)

print(
    "Second-stage FTP alert rate:",
    round(second_stage_alerts[ftp_mask].mean() * 100, 2), "%"
)

print("\nSecond-stage prediction distribution:")
print(pd.Series(reduction_predictions).value_counts().sort_index())
