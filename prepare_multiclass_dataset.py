
import pandas as pd
import numpy as np

FILES = {
    "MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": [
        "BENIGN", "DDoS"
    ],
    "MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": [
        "BENIGN", "PortScan"
    ]
}

MAX_PER_CLASS = 15000
rng = np.random.default_rng(42)

reservoirs = {
    "BENIGN": pd.DataFrame(),
    "DDoS": pd.DataFrame(),
    "PortScan": pd.DataFrame()
}

for file, labels in FILES.items():

    print(f"\nProcessing: {file}")

    for chunk in pd.read_csv(file, chunksize=50000):

        chunk.columns = chunk.columns.str.strip()
        chunk["Label"] = chunk["Label"].astype(str).str.strip()

        for label in labels:

            rows = chunk[chunk["Label"] == label].copy()

            if not rows.empty:

                rows["_random_key"] = rng.random(len(rows))

                combined = pd.concat(
                    [reservoirs[label], rows],
                    ignore_index=True
                )

                reservoirs[label] = combined.nsmallest(
                    MAX_PER_CLASS,
                    "_random_key"
                )

# Combine the three classes
df = pd.concat(reservoirs.values(), ignore_index=True)

# Remove sampling column
df = df.drop(columns=["_random_key"])

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

# Save dataset
df.to_csv("multiclass_dataset.csv", index=False)

print("\nMulticlass dataset created!")
print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["Label"].value_counts())
