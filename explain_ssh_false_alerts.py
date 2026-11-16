
import joblib
import numpy as np
import pandas as pd
import shap

model = joblib.load("webattack_detection_model.joblib")
features = joblib.load("webattack_feature_names.joblib")

df = pd.read_csv("crossfamily_test.csv")
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

# Select SSH-Patator traffic
ssh = df[df["Label"] == "SSH-Patator"].copy()

X = ssh[features].copy()
X = X.replace([np.inf, -np.inf], np.nan)

# Get model predictions
predictions = model.predict(X)

# Identify records incorrectly classified as Web Attacks
false_alerts = ssh[predictions != "BENIGN"].copy()
false_alerts["Predicted_Class"] = predictions[predictions != "BENIGN"]

print("\nSSH-PATATOR FALSE ALERT INVESTIGATION")
print("=" * 60)
print("Total SSH records:", len(ssh))
print("False alerts:", len(false_alerts))

if false_alerts.empty:
    print("No false alerts found.")
    raise SystemExit

print("\nPredicted classes:")
print(false_alerts["Predicted_Class"].value_counts())

# Explain a maximum of 20 false alerts
sample = false_alerts.head(20)
X_sample = sample[features].copy()

imputer = model.named_steps["imputer"]
classifier = model.named_steps["classifier"]

X_imputed = imputer.transform(X_sample)

explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_imputed)

class_names = classifier.classes_

print("\nFALSE ALERT SHAP ANALYSIS")
print("=" * 60)

for class_index, class_name in enumerate(class_names):

    values = shap_values[:, :, class_index]

    importance = pd.DataFrame({
        "Feature": features,
        "Mean_SHAP": values.mean(axis=0),
        "Mean_Absolute_SHAP": np.abs(values).mean(axis=0)
    }).sort_values("Mean_Absolute_SHAP", ascending=False)

    print("\nCLASS:", class_name)
    print("-" * 60)
    print(importance.head(10).to_string(index=False))

print("\nFalse-alert SHAP investigation completed.")
