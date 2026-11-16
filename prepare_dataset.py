import pandas as pd

file_path = (
    "MachineLearningCVE/"
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

MAX_PER_CLASS = 20000

reservoirs = {
    "BENIGN": pd.DataFrame(),
    "DDoS": pd.DataFrame()
}

for chunk in pd.read_csv(file_path, chunksize=50000):

    chunk.columns = chunk.columns.str.strip()
    chunk["Label"] = chunk["Label"].astype(str).str.strip()

    for label in reservoirs:

        new_rows = chunk[chunk["Label"] == label]

        if not new_rows.empty:

            combined = pd.concat(
                [reservoirs[label], new_rows],
                ignore_index=True
            )

            reservoirs[label] = combined.sample(
                n=min(len(combined), MAX_PER_CLASS),
                random_state=42
            )

# Combine both classes
df = pd.concat(reservoirs.values(), ignore_index=True)

# Shuffle records
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Convert labels into binary values
df["Label"] = df["Label"].map({
    "BENIGN": 0,
    "DDoS": 1
})

# Save prepared dataset
df.to_csv("balanced_dataset.csv", index=False)

print("Dataset preparation completed!")
print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["Label"].value_counts())
