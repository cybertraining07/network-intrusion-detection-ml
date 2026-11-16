
import joblib
import numpy as np
import pandas as pd

model = joblib.load("webattack_detection_model.joblib")
features = joblib.load("webattack_feature_names.joblib")

df = pd.read_csv("crossfamily_test.csv")
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

ssh = df[df["Label"] == "SSH-Patator"].copy()

X = ssh[features].copy()
X = X.replace([np.inf, -np.inf], np.nan)

predictions = model.predict(X)

ssh["Group"] = np.where(
    predictions == "BENIGN",
    "Correctly Classified",
    "False Alert"
)

# Compare numeric feature averages
comparison = ssh.groupby("Group")[features].mean().T

comparison["Absolute_Difference"] = (
    comparison["False Alert"] -
    comparison["Correctly Classified"]
).abs()

comparison = comparison.sort_values(
    "Absolute_Difference",
    ascending=False
)

print("\nSSH-PATATOR GROUP COMPARISON")
print("=" * 70)
print("Correctly classified:", (ssh["Group"] == "Correctly Classified").sum())
print("False alerts:", (ssh["Group"] == "False Alert").sum())

print("\nTop 20 features with largest mean differences:")
print(comparison.head(20).to_string())

comparison.to_csv("ssh_group_comparison.csv")

print("\nComparison saved to ssh_group_comparison.csv")
