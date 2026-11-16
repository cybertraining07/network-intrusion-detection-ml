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

# Load prepared dataset
df = pd.read_csv("balanced_dataset.csv")

# Separate features and target
X = df.drop("Label", axis=1)
y = df["Label"]

# Convert features to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Replace infinite values with missing values
X = X.replace([np.inf, -np.inf], np.nan)

# Remove columns that contain no usable values
X = X.dropna(axis=1, how="all")

print("Dataset shape:", X.shape)
print("Missing values:", X.isna().sum().sum())

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Build machine learning pipeline
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

# Train model
print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("\nMODEL EVALUATION")
print("==============================")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["BENIGN", "DDoS"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and feature names
joblib.dump(model, "ddos_detection_model.joblib")
joblib.dump(X.columns.tolist(), "feature_names.joblib")

print("\nModel saved successfully!")
