import joblib
import pandas as pd

# Load trained model and feature names
model = joblib.load("ddos_detection_clean_model.joblib")
feature_names = joblib.load("clean_feature_names.joblib")

# Extract Random Forest classifier
rf = model.named_steps["classifier"]

# Get feature importance scores
importance = rf.feature_importances_

# Create results table
results = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

# Sort from highest to lowest
results = results.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 15 IMPORTANT NETWORK FEATURES")
print("===================================")

print(results.head(15).to_string(index=False))

# Save results
results.to_csv("feature_importance_results.csv", index=False)

print("\nFeature importance saved successfully!")
