
import pandas as pd
import numpy as np

FILE = "thursday_test.csv"

df = pd.read_csv(FILE)

df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

print("Original dataset:", df.shape)

# Remove duplicate feature rows
feature_columns = [
    col for col in df.columns if col != "Label"
]

df = df.drop_duplicates(
    subset=feature_columns
).reset_index(drop=True)

# Shuffle records
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

df.to_csv("webattack_dataset.csv", index=False)

print("\nWeb Attack dataset created!")
print("Dataset shape:", df.shape)

print("\nClass distribution:")
print(df["Label"].value_counts())
