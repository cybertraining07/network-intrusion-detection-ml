
import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("crossfamily_test.csv")
df.columns = df.columns.str.strip()

# Select SSH-Patator records
ssh = df[df["Label"] == "SSH-Patator"].copy()

X = ssh.drop(columns=["Label"])
X = X.replace([np.inf, -np.inf], np.nan)
X = X.apply(pd.to_numeric, errors="coerce")

# Load Original Random Forest
model = joblib.load("webattack_detection_model.joblib")

predictions = model.predict(X)

ssh["Prediction"] = predictions
ssh["Web_Attack_Alert"] = predictions != "BENIGN"

print("SSH-PATATOR INVESTIGATION")
print("=" * 45)

print("\nTotal SSH-Patator records:", len(ssh))

print("\nPrediction distribution:")
print(ssh["Prediction"].value_counts())

alerts = ssh[ssh["Web_Attack_Alert"]]

print("\nTotal Web Attack alerts:", len(alerts))

if len(alerts) > 0:

    print("\nAverage feature values: alerts vs non-alerts")

    feature_columns = [
        col for col in X.columns
        if X[col].notna().any()
    ]

    comparison = pd.DataFrame({
        "Alert Average": X.loc[alerts.index, feature_columns].mean(),
        "Non-Alert Average": X.loc[
            ssh.index.difference(alerts.index),
            feature_columns
        ].mean()
    })

    comparison["Absolute Difference"] = (
        comparison["Alert Average"]
        - comparison["Non-Alert Average"]
    ).abs()

    print(
        comparison.sort_values(
            "Absolute Difference",
            ascending=False
        ).head(15).to_string()
    )

print("\nInvestigation completed.")
