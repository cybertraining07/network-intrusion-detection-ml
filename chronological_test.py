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

# Read original dataset in chunks
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

# Reconstruct dataset in chronological order
df = pd.concat(reservoirs.values(), ignore_index=True)

# Remove duplicate feature records
df = df.drop_duplicates(
    subset=df.columns.drop("Label")
)

# Split each class into earlier and later records
earlier = []
later = []

for label in ["BENIGN", "DDoS"]:

    class_data = df[df["Label"] == label].copy()

    split_point = int(len(class_data) * 0.70)

    earlier.append(class_data.iloc[:split_point])
    later.append(class_data.iloc[split_point:])

train_df = pd.concat(earlier, ignore_index=True)
test_df = pd.concat(later, ignore_index=True)

train_df.to_csv("chronological_train.csv", index=False)
test_df.to_csv("chronological_test.csv", index=False)

print("Training records:", len(train_df))
print("Testing records:", len(test_df))

print("\nTraining labels:")
print(train_df["Label"].value_counts())

print("\nTesting labels:")
print(test_df["Label"].value_counts())

print("\nChronological datasets saved!")
