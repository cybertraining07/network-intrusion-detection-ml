
import pandas as pd
import numpy as np

FILE = (
    "MachineLearningCVE/"
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
)

LABELS = [
    "BENIGN",
    "Web Attack � Brute Force",
    "Web Attack � XSS",
    "Web Attack � Sql Injection"
]

MAX_PER_CLASS = 5000
rng = np.random.default_rng(42)

reservoirs = {
    label: pd.DataFrame()
    for label in LABELS
}

for chunk in pd.read_csv(FILE, chunksize=50000):

    chunk.columns = chunk.columns.str.strip()
    chunk["Label"] = chunk["Label"].astype(str).str.strip()

    for label in LABELS:

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

df = pd.concat(
    reservoirs.values(),
    ignore_index=True
)

df = df.drop(columns=["_random_key"])

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

df.to_csv("thursday_test.csv", index=False)

print("Thursday test dataset created!")
print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["Label"].value_counts())
