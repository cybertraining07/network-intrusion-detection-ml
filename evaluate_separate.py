import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load separate datasets
train_df = pd.read_csv("chronological_train.csv")
test_df = pd.read_csv("chronological_test.csv")

# Prepare features and labels
X_train = train_df.drop("Label", axis=1)
y_train = train_df["Label"].map({
    "BENIGN": 0,
    "DDoS": 1
})

X_test = test_df.drop("Label", axis=1)
y_test = test_df["Label"].map({
    "BENIGN": 0,
    "DDoS": 1
})

# Convert features to numeric
X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_test = X_test.apply(pd.to_numeric, errors="coerce")

# Replace infinite values
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Keep the same features
X_test = X_test.reindex(columns=X_train.columns)

# Build model
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=2
    ))
])

print("Training on separate training dataset...")
model.fit(X_train, y_train)

# Predict unseen test records
predictions = model.predict(X_test)

# Evaluate
print("\nSEPARATE DATASET EVALUATION")
print("==============================")

print("Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["BENIGN", "DDoS"]
))

cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn) * 100

print(f"\nFalse Positive Rate: {fpr:.4f}%")
print(f"False Negatives: {fn}")

joblib.dump(model, "ddos_separate_model.joblib")

print("\nSeparate-dataset model saved!")
