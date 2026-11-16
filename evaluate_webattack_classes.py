
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline as SklearnPipeline

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Load dataset
df = pd.read_csv("webattack_dataset.csv")

df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

X = df.drop(columns=["Label"])
y = df["Label"]

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")

classes = sorted(y.unique())

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Original": SklearnPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=2
        ))
    ]),

    "SMOTE": ImbPipeline([
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

print("PER-CLASS FIVE-FOLD VALIDATION")
print("==============================")

for name, model in models.items():

    print(f"\n{name} Random Forest")
    print("------------------------------")

    fold_results = {
        label: {"precision": [], "recall": [], "f1": []}
        for label in classes
    }

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y), start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_test,
                predictions,
                labels=classes,
                zero_division=0
            )
        )

        for i, label in enumerate(classes):
            fold_results[label]["precision"].append(precision[i])
            fold_results[label]["recall"].append(recall[i])
            fold_results[label]["f1"].append(f1[i])

        print(f"Fold {fold} completed")

    for label in classes:

        print(f"\nClass: {label}")

        for metric in ["precision", "recall", "f1"]:

            scores = np.array(
                fold_results[label][metric]
            )

            print(
                f"{metric.capitalize()}: "
                f"{scores.mean():.4f} "
                f"+/- {scores.std():.4f}"
            )

print("\nPer-class evaluation completed!")
