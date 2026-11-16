
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("webattack_dataset.csv")

df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

X = df.drop(columns=["Label"])
y = df["Label"]

# Clean features
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")

feature_names = X.columns.tolist()

# Same split for comparison
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Improved Random Forest
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=24,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=2
    ))
])

print("Training improved Web Attack model...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nImproved Model Accuracy:")
print(f"{accuracy_score(y_test, predictions) * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=sorted(y.unique()),
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=sorted(y.unique())
))

# Save improved model separately
joblib.dump(
    model,
    "webattack_improved_model.joblib"
)

joblib.dump(
    feature_names,
    "webattack_improved_features.joblib"
)

print("\nImproved Web Attack model saved successfully!")
