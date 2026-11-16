
import joblib
import pandas as pd
import shap

# Load the trained Web Attack model
model = joblib.load("webattack_detection_model.joblib")
features = joblib.load("webattack_feature_names.joblib")

# Load the cross-family dataset
df = pd.read_csv("crossfamily_test.csv")
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

# Select SSH-Patator records only
ssh = df[df["Label"] == "SSH-Patator"].copy()

if ssh.empty:
    raise ValueError("No SSH-Patator records found.")

# Use a small sample to protect system memory
sample = ssh.sample(n=min(20, len(ssh)), random_state=42)

X = sample[features].copy()
X = X.replace([float("inf"), -float("inf")], float("nan"))

# Apply the model's imputer
imputer = model.named_steps["imputer"]
classifier = model.named_steps["classifier"]
X_imputed = imputer.transform(X)

# Generate SHAP explanations
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_imputed)

print("\nSHAP OUTPUT")
print("=" * 50)
print("Sample records:", len(sample))
print("SHAP version:", shap.__version__)
print("SHAP output type:", type(shap_values))

# Handle multiclass SHAP output
if isinstance(shap_values, list):
    values = shap_values
    print("Number of class explanations:", len(values))
else:
    print("SHAP output shape:", shap_values.shape)

print("\nSHAP analysis completed.")
