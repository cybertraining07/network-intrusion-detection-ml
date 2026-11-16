import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# Load cleaned dataset
df = pd.read_csv("balanced_dataset_clean.csv")

X = df.drop("Label", axis=1)
y = df["Label"]

# Convert features to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Replace infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# Remove entirely empty columns
X = X.dropna(axis=1, how="all")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create pipeline
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

# Train
print("Training cleaned-data Random Forest...")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\nCLEAN MODEL EVALUATION")
print("==============================")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["BENIGN", "DDoS"]
))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculate false-positive rate
tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn) * 100

print(f"\nFalse Positive Rate: {fpr:.4f}%")
print(f"False Negatives: {fn}")

# Save cleaned-data model
joblib.dump(model, "ddos_detection_clean_model.joblib")
joblib.dump(X.columns.tolist(), "clean_feature_names.joblib")

print("\nClean model saved successfully!")
