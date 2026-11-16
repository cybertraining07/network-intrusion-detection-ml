
import joblib
import pandas as pd

# Load original Web Attack model
model = joblib.load("webattack_detection_model.joblib")

# Extract Random Forest from pipeline
rf = model.named_steps["classifier"]

# Load feature names
features = joblib.load("webattack_feature_names.joblib")

# Create feature importance report
importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("WEB ATTACK MODEL FEATURE IMPORTANCE")
print("=" * 50)

print("\nTop 20 important features:")
print(importance.head(20).to_string(index=False))

# Save report
importance.to_csv(
    "webattack_model_feature_importance.csv",
    index=False
)

print("\nFeature importance report saved!")
