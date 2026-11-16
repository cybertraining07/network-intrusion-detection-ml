
import joblib
import numpy as np
import pandas as pd
import shap

# Load model and feature names
model = joblib.load("webattack_detection_model.joblib")
features = joblib.load("webattack_feature_names.joblib")

# Load cross-family dataset
df = pd.read_csv("crossfamily_test.csv")
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

# Select SSH-Patator records
ssh = df[df["Label"] == "SSH-Patator"].copy()
sample = ssh.sample(n=min(20, len(ssh)), random_state=42)

X = sample[features].copy()
X = X.replace([np.inf, -np.inf], np.nan)

# Apply model imputer
imputer = model.named_steps["imputer"]
classifier = model.named_steps["classifier"]
X_imputed = imputer.transform(X)

# Calculate SHAP values
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_imputed)

print("\nSSH-PATATOR SHAP FEATURE ANALYSIS")
print("=" * 60)

class_names = classifier.classes_

for class_index, class_name in enumerate(class_names):

    values = shap_values[:, :, class_index]

    importance = pd.DataFrame({
        "Feature": features,
        "Mean_Absolute_SHAP": np.abs(values).mean(axis=0)
    }).sort_values("Mean_Absolute_SHAP", ascending=False)

    print(f"\nCLASS: {class_name}")
    print("-" * 60)
    print(importance.head(10).to_string(index=False))

    filename = (
        "ssh_shap_"
        + str(class_index)
        + ".csv"
    )

    importance.to_csv(filename, index=False)

print("\nSHAP feature reports saved successfully.")
