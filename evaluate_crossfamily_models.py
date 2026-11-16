
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("crossfamily_test.csv")
df.columns = df.columns.str.strip()

X = df.drop(columns=["Label"])
y = df["Label"].astype(str).str.strip()

# Clean invalid numeric values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.apply(pd.to_numeric, errors="coerce")

models = {
    "Original Random Forest": "webattack_detection_model.joblib",
    "Improved Random Forest": "webattack_improved_model.joblib",
    "SMOTE Random Forest": "webattack_smote_model.joblib"
}

print("CROSS-FAMILY MODEL EVALUATION")
print("=" * 55)

all_results = []

for name, filename in models.items():

    print(f"\nMODEL: {name}")
    print("-" * 55)

    model = joblib.load(filename)

    predictions = model.predict(X)

    # 0 = BENIGN
    # 1 = Any non-BENIGN prediction (Web Attack alert)
    actual_binary = (y != "BENIGN").astype(int).to_numpy()
    predicted_binary = (predictions != "BENIGN").astype(int)

    accuracy = accuracy_score(
        actual_binary, predicted_binary
    )

    precision = precision_score(
        actual_binary, predicted_binary, zero_division=0
    )

    recall = recall_score(
        actual_binary, predicted_binary, zero_division=0
    )

    f1 = f1_score(
        actual_binary, predicted_binary, zero_division=0
    )

    matrix = confusion_matrix(
        actual_binary,
        predicted_binary,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Alert Precision: {precision:.4f}")
    print(f"Other Attack Alert Rate: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"False Positive Rate: {false_positive_rate:.4f}")

    print("\nCONFUSION MATRIX")
    print(matrix)

    print("\nPREDICTION DISTRIBUTION")
    print(pd.Series(predictions).value_counts())

    print("\nALERT RATE BY TRAFFIC CATEGORY")

    for label in sorted(y.unique()):

        mask = y == label

        alert_rate = (
            predicted_binary[mask].mean()
        )

        print(f"{label}: {alert_rate:.2%}")

    all_results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Alert Precision": precision,
        "Other Attack Alert Rate": recall,
        "F1 Score": f1,
        "False Positive Rate": false_positive_rate
    })

results = pd.DataFrame(all_results)

results.to_csv(
    "crossfamily_model_comparison.csv",
    index=False
)

print("\nFINAL MODEL COMPARISON")
print(results.to_string(index=False))

print("\nSaved: crossfamily_model_comparison.csv")
