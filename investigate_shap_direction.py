
import joblib
import numpy as np
import pandas as pd
import shap

model = joblib.load("webattack_detection_model.joblib")
features = joblib.load("webattack_feature_names.joblib")

df = pd.read_csv("crossfamily_test.csv")
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

ssh = df[df["Label"] == "SSH-Patator"].copy()
sample = ssh.sample(n=min(20, len(ssh)), random_state=42)

X = sample[features].copy()
X = X.replace([np.inf, -np.inf], np.nan)

imputer = model.named_steps["imputer"]
classifier = model.named_steps["classifier"]

X_imputed = imputer.transform(X)

explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_imputed)

class_names = classifier.classes_

for class_index, class_name in enumerate(class_names):

    values = shap_values[:, :, class_index]

    results = pd.DataFrame({
        "Feature": features,
        "Mean_SHAP": values.mean(axis=0),
        "Mean_Absolute_SHAP": np.abs(values).mean(axis=0)
    })

    results = results.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    print("\nCLASS:", class_name)
    print("=" * 65)

    print("\nFeatures pushing predictions toward this class:")
    print(
        results.sort_values("Mean_SHAP", ascending=False)
        .head(5).to_string(index=False)
    )

    print("\nFeatures pushing predictions away from this class:")
    print(
        results.sort_values("Mean_SHAP")
        .head(5).to_string(index=False)
    )

print("\nSHAP direction analysis completed.")
