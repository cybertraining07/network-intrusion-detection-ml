import pandas as pd

df = pd.read_csv("balanced_dataset.csv")

print("Original records:", len(df))

# Remove identical feature records
df = df.drop_duplicates(
    subset=df.columns.drop("Label"),
    keep="first"
)

print("Records after deduplication:", len(df))
print("Duplicates remaining:", df.drop(
    columns=["Label"]
).duplicated().sum())

print("\nClass distribution:")
print(df["Label"].value_counts())

df.to_csv("balanced_dataset_clean.csv", index=False)

print("\nClean dataset saved successfully!")
