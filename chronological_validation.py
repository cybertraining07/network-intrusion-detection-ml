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

FILE = (
    "MachineLearningCVE/"
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

CHUNK_SIZE = 50000
MAX_PER_CLASS = 10000
reservoirs = {
    "train": {"BENIGN": [], "DDoS": []},
    "test": {"BENIGN": [], "DDoS": []}
}

total_rows = sum(
    len(chunk)
    for chunk in pd.read_csv(FILE, chunksize=CHUNK_SIZE)
)

split_row = int(total_rows * 0.70)

print("Total original rows:", total_rows)
print("Training boundary:", split_row)

offset = 0

for chunk in pd.read_csv(FILE, chunksize=CHUNK_SIZE):

    chunk.columns = chunk.columns.str.strip()
    chunk["Label"] = chunk["Label"].astype(str).str.strip()

    # Preserve original row positions
    positions = np.arange(offset, offset + len(chunk))

    train_chunk = chunk[positions < split_row]
    test_chunk = chunk[positions >= split_row]

    for name, part in [
        ("train", train_chunk),
        ("test", test_chunk)
    ]:
        for label in ["BENIGN", "DDoS"]:

            rows = part[part["Label"] == label]

            if len(rows):
                reservoirs[name][label].append(
                    rows.sample(
                        n=min(len(rows), MAX_PER_CLASS),
                        random_state=42
                    )
                )

    offset += len(chunk)

def build_sample(data):
    frames = []

    for label, pieces in data.items():
        if pieces:
            combined = pd.concat(pieces, ignore_index=True)
            frames.append(
                combined.sample(
                    n=min(len(combined), MAX_PER_CLASS),
                    random_state=42
                )
            )

    return pd.concat(frames, ignore_index=True)

train_df = build_sample(reservoirs["train"])
test_df = build_sample(reservoirs["test"])

# Remove duplicates within each split
features = train_df.columns.drop("Label")

train_df = train_df.drop_duplicates(subset=features)
test_df = test_df.drop_duplicates(subset=features)

print("\nTraining distribution:")
print(train_df["Label"].value_counts())

print("\nTesting distribution:")
print(test_df["Label"].value_counts())

# Prepare features and labels
X_train = train_df.drop("Label", axis=1)
y_train = train_df["Label"].map({"BENIGN": 0, "DDoS": 1})

X_test = test_df.drop("Label", axis=1)
y_test = test_df["Label"].map({"BENIGN": 0, "DDoS": 1})

X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_test = X_test.apply(pd.to_numeric, errors="coerce")

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

X_test = X_test.reindex(columns=X_train.columns)

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

print("\nTraining chronological model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nCHRONOLOGICAL VALIDATION RESULTS")
print("================================")

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

print(f"\nFalse Positive Rate: {fp / (fp + tn) * 100:.4f}%")
print("False Negatives:", fn)

joblib.dump(model, "ddos_chronological_model.joblib")

print("\nChronological model saved!")
