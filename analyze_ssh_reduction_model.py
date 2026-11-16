
import joblib
import pandas as pd

model = joblib.load("ssh_false_alert_reduction_model.joblib")

classifier = model.named_steps["classifier"]
features = joblib.load("webattack_feature_names.joblib")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": classifier.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nSSH FALSE-ALERT REDUCTION MODEL")
print("=" * 60)

print("\nTop 20 important features:")
print(importance.head(20).to_string(index=False))

importance.to_csv(
    "ssh_false_alert_model_importance.csv",
    index=False
)

print("\nFeature importance report saved.")
