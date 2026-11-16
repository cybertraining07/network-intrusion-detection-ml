
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
df = pd.read_csv("multiclass_dataset.csv")

df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

print("Dataset shape:", df.shape)

# Separate features and labels
X = df.drop(columns=["Label"])
y = df["Label"]

# Convert features to numeric
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

# Remove columns containing only missing values
X = X.dropna(axis=1, how="all")

feature_names = X.columns.tolist()

print("Number of features:", len(feature_names))
print("\nClass distribution:")
print(y.value_counts())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Build model pipeline
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
print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=["BENIGN", "DDoS", "PortScan"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["BENIGN", "DDoS", "PortScan"]
))

# Save model and feature names
joblib.dump(model, "multiclass_intrusion_model.joblib")
joblib.dump(feature_names, "multiclass_feature_names.joblib")

print("\nMulticlass model saved successfully!")
