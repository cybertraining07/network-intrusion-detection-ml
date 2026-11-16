import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("feature_importance_results.csv")

top_features = df.head(15).sort_values(
    "Importance",
    ascending=True
)

plt.figure(figsize=(11, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Network Traffic Features")
plt.title("Top 15 Features - Random Forest DDoS Detection")

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300
)

plt.show()

print("Feature importance chart saved!")
