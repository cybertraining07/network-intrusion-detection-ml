
import joblib
import numpy as np
import pandas as pd

# Load models
web_model = joblib.load("webattack_detection_model.joblib")
reduction_model = joblib.load(
    "ssh_false_alert_reduction_model.joblib"
)

# Load dataset
df = pd.read_csv("crossfamily_test.csv")
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

# Select SSH-Patator traffic
ssh = df[df["Label"] == "SSH-Patator"].copy()

# Prepare features
features = joblib.load("webattack_feature_names.joblib")
X = ssh[features].copy()
X = X.replace([np.inf, -np.inf], np.nan)

# Original model predictions
original_predictions = web_model.predict(X)

# Second-stage model predictions
reduction_predictions = reduction_model.predict(X)

# Original alerts
original_alerts = original_predictions != "BENIGN"

# Second-stage identifies alerts to suppress
suppressed = reduction_predictions == 1

# Alerts remaining after filtering
remaining_alerts = original_alerts & ~suppressed

# Summary
print("\nSSH FALSE-ALERT REDUCTION EVALUATION")
print("=" * 55)

print("Total SSH records:", len(ssh))
print("Original alerts:", original_alerts.sum())
print("Original BENIGN predictions:", (~original_alerts).sum())
print("Alerts suppressed:", (original_alerts & suppressed).sum())
print("Alerts remaining:", remaining_alerts.sum())

# Alert suppression rate
if original_alerts.sum() > 0:
    rate = (
        (original_alerts & suppressed).sum()
        / original_alerts.sum()
    )
    print(f"Alert suppression rate: {rate:.2%}")

# Check reduction model behavior
print("\nSECOND-STAGE PREDICTIONS")
print(pd.Series(reduction_predictions).value_counts())
print("\nDETAILED FILTERING CHECK")
print("-" * 45)

print("Original alerts flagged by reduction model:",
      (original_alerts & suppressed).sum())

print("Original BENIGN records flagged by reduction model:",
      ((~original_alerts) & suppressed).sum())

print("Original alerts not flagged by reduction model:",
      (original_alerts & ~suppressed).sum())

print("Total records flagged by reduction model:",
      suppressed.sum())
# Save evaluation results
results = pd.DataFrame([{
    "Total SSH Records": len(ssh),
    "Original Alerts": int(original_alerts.sum()),
    "Original BENIGN Predictions": int((~original_alerts).sum()),
    "Alerts Suppressed": int((original_alerts & suppressed).sum()),
    "Alerts Remaining": int(remaining_alerts.sum()),
    "Original BENIGN Records Flagged": int(((~original_alerts) & suppressed).sum()),
    "Total Second-Stage Alerts": int(suppressed.sum())
}])

results.to_csv(
    "ssh_false_alert_reduction_results.csv",
    index=False
)

print("\nResults saved: ssh_false_alert_reduction_results.csv")
print("\nEvaluation completed.")
