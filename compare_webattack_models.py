
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("webattack_dataset.csv")

df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

X = df.drop(columns=["Label"])
y = df["Label"]

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")

# Five-fold stratified validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Original Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=2
    ),

    "Improved Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=24,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=2
    )
}

scoring = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted"
}

print("WEB ATTACK CROSS-VALIDATION")
print("===========================")

for name, classifier in models.items():

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", classifier)
    ])

    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1
    )

    print(f"\n{name}")
    print("---------------------------")

    for metric in scoring:

        scores = results[f"test_{metric}"]

        print(
            f"{metric}: "
            f"{scores.mean():.4f} "
            f"+/- {scores.std():.4f}"
        )

print("\nCross-validation completed!")
