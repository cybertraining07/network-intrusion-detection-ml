
import pandas as pd

FILE = (
    "MachineLearningCVE/"
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
)

label_counts = {}

for chunk in pd.read_csv(FILE, chunksize=50000):

    chunk.columns = chunk.columns.str.strip()

    labels = chunk["Label"].astype(str).str.strip()

    counts = labels.value_counts()

    for label, count in counts.items():
        label_counts[label] = (
            label_counts.get(label, 0) + count
        )

print("\nEXACT LABELS FOUND IN THURSDAY DATASET")
print("=" * 50)

for label, count in label_counts.items():
    print(f"{repr(label)}: {count:,}")
