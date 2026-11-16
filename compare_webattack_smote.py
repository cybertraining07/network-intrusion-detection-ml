
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

df = pd.read_csv("webattack_dataset.csv")

df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

X = df.drop(columns=["Label"])
y = df["Label"]

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Original Random Forest": SklearnPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=2
        ))
    ]),

    "Improved Random Forest": SklearnPipeline([
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
    ]),

    "SMOTE Random Forest": ImbPipeline([
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
}

scoring = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted"
}

print("THREE-MODEL CROSS-VALIDATION")
print("============================")

for name, model in models.items():

    print(f"\n{name}")
    print("----------------------------")

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1
    )

    for metric in scoring:
        scores = results[f"test_{metric}"]

        print(
            f"{metric}: "
            f"{scores.mean():.4f} "
            f"+/- {scores.std():.4f}"
        )

print("\nThree-model comparison completed!")
