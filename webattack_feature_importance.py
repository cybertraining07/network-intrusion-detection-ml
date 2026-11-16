
import pandas as pd
import joblib

# Load saved model and feature names
model = joblib.load("webattack_detection_model.joblib")
feature_names = joblib.load("webattack_feature_names.joblib")

# Extract Random Forest classifier
rf = model.named_steps["classifier"]

# Get feature importance
importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf.feature_importances_
})

# Sort by importance
importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 20 WEB ATTACK FEATURES")
print("==========================")

print(importance.head(20).to_string(index=False))

# Save results
importance.to_csv(
    "webattack_feature_importance.csv",
    index=False
)

print("\nFeature importance saved successfully!")
