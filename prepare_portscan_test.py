import pandas as pd
import numpy as np

FILE = (
    "MachineLearningCVE/"
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
)

MAX_PER_CLASS = 10000
rng = np.random.default_rng(42)

reservoirs = {
    "BENIGN": pd.DataFrame(),
    "PortScan": pd.DataFrame()
}

for chunk in pd.read_csv(FILE, chunksize=50000):

    chunk.columns = chunk.columns.str.strip()
    chunk["Label"] = chunk["Label"].astype(str).str.strip()

    for label in reservoirs:

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

# Combine samples
df = pd.concat(reservoirs.values(), ignore_index=True)

df = df.drop(columns=["_random_key"])

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df.to_csv("portscan_test.csv", index=False)

print("PortScan test dataset created!")
print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["Label"].value_counts())
