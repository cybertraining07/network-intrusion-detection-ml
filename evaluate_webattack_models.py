
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("webattack_dataset.csv")
df.columns = df.columns.str.strip()
X = df.drop(columns=["Label"])

# Convert infinity values to missing values
X = X.replace([np.inf, -np.inf], np.nan)

# Ensure all features are numeric
X = X.apply(pd.to_numeric, errors="coerce")

y = df["Label"].astype(str).str.strip()

print("\nDATA QUALITY CHECK")
print("Infinite values remaining:", np.isinf(X.to_numpy()).sum())
print("Missing values:", X.isna().sum().sum())

classes = sorted(y.unique())

print("DATASET CLASS DISTRIBUTION")
print(y.value_counts())
print("\nFive-fold model comparison")

# Five-fold stratified validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Original Random Forest
original_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=2
    ))
])

# SMOTE Random Forest
smote_model = ImbPipeline([
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

models = {
    "Original Random Forest": original_model,
    "SMOTE Random Forest": smote_model
}

for model_name, model in models.items():

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    macro_f1_scores = []
    weighted_f1_scores = []

    print(f"\n{model_name}")
    print("-" * 45)

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y), start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy_scores.append(
            accuracy_score(y_test, predictions)
        )

        precision_scores.append(
            precision_score(
                y_test, predictions,
                average="macro",
                zero_division=0
            )
        )

        recall_scores.append(
            recall_score(
                y_test, predictions,
                average="macro",
                zero_division=0
            )
        )

        macro_f1_scores.append(
            f1_score(
                y_test, predictions,
                average="macro",
                zero_division=0
            )
        )

        weighted_f1_scores.append(
            f1_score(
                y_test, predictions,
                average="weighted",
                zero_division=0
            )
        )

        print(f"Fold {fold} completed")

    print("\nOVERALL RESULTS")
    print(
        f"Accuracy:       "
        f"{np.mean(accuracy_scores):.4f} "
        f"+/- {np.std(accuracy_scores):.4f}"
    )
    print(
        f"Macro Precision:{np.mean(precision_scores):.4f} "
        f"+/- {np.std(precision_scores):.4f}"
    )
    print(
        f"Macro Recall:   "
        f"{np.mean(recall_scores):.4f} "
        f"+/- {np.std(recall_scores):.4f}"
    )
    print(
        f"Macro F1:       "
        f"{np.mean(macro_f1_scores):.4f} "
        f"+/- {np.std(macro_f1_scores):.4f}"
    )
    print(
        f"Weighted F1:    "
        f"{np.mean(weighted_f1_scores):.4f} "
        f"+/- {np.std(weighted_f1_scores):.4f}"
    )

print("\nMODEL COMPARISON COMPLETED")
