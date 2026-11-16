
import joblib
import numpy as np
import pandas as pd

# Load both trained models
web_model = joblib.load("webattack_detection_model.joblib")
reduction_model = joblib.load("ssh_false_alert_reduction_model.joblib")

# Load SSH-Patator test data
df = pd.read_csv("crossfamily_test.csv")
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

ssh = df[df["Label"] == "SSH-Patator"].copy()

# Prepare model features
features = joblib.load("webattack_feature_names.joblib")
X = ssh[features].copy()
X = X.replace([np.inf, -np.inf], np.nan)

# Generate predictions
original_predictions = web_model.predict(X)
reduction_predictions = reduction_model.predict(X)

# Class 1 means the second-stage model predicts an original alert.
# Class 0 means it predicts an original BENIGN decision.
original_alerts = original_predictions != "BENIGN"
second_stage_alerts = reduction_predictions == 1

# Keep original alerts confirmed by the second-stage model.
# Suppress original alerts predicted as class 0.
retained_alerts = original_alerts & second_stage_alerts
suppressed_alerts = original_alerts & ~second_stage_alerts

# Count results
total = len(ssh)
original_count = int(original_alerts.sum())
retained_count = int(retained_alerts.sum())
suppressed_count = int(suppressed_alerts.sum())
benign_flagged = int(((~original_alerts) & second_stage_alerts).sum())

print("\nCORRECTED SSH FALSE-ALERT FILTERING EVALUATION")
print("=" * 50)
print("Total SSH records:", total)
print("Original Web Attack alerts:", original_count)
print("Original BENIGN predictions:", int((~original_alerts).sum()))
print("Alerts retained:", retained_count)
print("Alerts suppressed:", suppressed_count)
print("Original BENIGN records flagged by second stage:", benign_flagged)

if original_count:
    print("Alert suppression rate: {:.2f}%".format(
        suppressed_count / original_count * 100
    ))

print("\nSECOND-STAGE PREDICTION DISTRIBUTION")
print(pd.Series(reduction_predictions).value_counts().sort_index())

print("\nINTERPRETATION")
print("Class 1 = retain/confirm original alert")
print("Class 0 = candidate for suppressing original alert")
print("Suppression does not prove an alert was a false positive.")

# Save corrected results
results = pd.DataFrame([{
    "Total SSH Records": total,
    "Original Alerts": original_count,
    "Alerts Retained": retained_count,
    "Alerts Suppressed": suppressed_count,
    "Original BENIGN Records Flagged": benign_flagged,
    "Alert Suppression Rate (%)": (
        round(suppressed_count / original_count * 100, 2)
        if original_count else 0
    )
}])

results.to_csv("ssh_false_alert_reduction_results.csv", index=False)

print("\nCorrected results saved to ssh_false_alert_reduction_results.csv")
