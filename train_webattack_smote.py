
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("webattack_dataset.csv")

df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

X = df.drop(columns=["Label"])
y = df["Label"]

# Convert features to numeric
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

# Remove columns containing no usable values
X = X.dropna(axis=1, how="all")

print("Original class distribution:")
print(y.value_counts())

# Split before applying SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Build pipeline: imputation -> SMOTE -> Random Forest
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("smote", SMOTE(
        random_state=42,
        k_neighbors=3
    )),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        class_weight=None,
        random_state=42,
        n_jobs=2
    ))
])

print("\nTraining SMOTE Web Attack model...")

model.fit(X_train, y_train)

# Evaluate on untouched test data
predictions = model.predict(X_test)

print("\nSMOTE Model Accuracy:")
print(f"{accuracy_score(y_test, predictions) * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
))

# Save model and feature names
joblib.dump(model, "webattack_smote_model.joblib")
joblib.dump(X.columns.tolist(), "webattack_smote_features.joblib")

print("\nSMOTE Web Attack model saved successfully!")
