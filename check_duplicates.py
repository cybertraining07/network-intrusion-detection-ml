import pandas as pd

df = pd.read_csv("balanced_dataset.csv")

X = df.drop("Label", axis=1)

# Exact duplicate rows, including labels
full_duplicates = df.duplicated().sum()

# Duplicate feature rows, ignoring labels
feature_duplicates = X.duplicated().sum()

# Check whether identical features have conflicting labels
hashes = pd.util.hash_pandas_object(X, index=False)

check = pd.DataFrame({
    "hash": hashes,
    "Label": df["Label"]
})

conflicting_labels = (
    check.groupby("hash")["Label"]
    .nunique()
    .gt(1)
    .sum()
)

print("TOTAL RECORDS:", len(df))
print("Exact duplicate rows:", full_duplicates)
print("Duplicate feature rows:", feature_duplicates)
print("Feature groups with conflicting labels:", conflicting_labels)
