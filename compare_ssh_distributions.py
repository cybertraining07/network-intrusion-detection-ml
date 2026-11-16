
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

selected_features = [
    "Flow Duration",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Init_Win_bytes_forward",
    "Flow IAT Mean",
    "Flow IAT Min",
    "Flow IAT Max",
]

print("\nSSH-PATATOR DISTRIBUTION ANALYSIS")
print("=" * 75)

for feature in selected_features:

    print(f"\nFEATURE: {feature}")
    print("-" * 75)

    result = ssh.groupby("Group")[feature].agg(
        ["count", "mean", "median",
         lambda x: x.quantile(0.25),
         lambda x: x.quantile(0.75)]
    )

    result.columns = [
        "Count",
        "Mean",
        "Median",
        "25th Percentile",
        "75th Percentile"
    ]

    print(result.to_string())

print("\nDistribution analysis completed.")
