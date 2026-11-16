
import pandas as pd

file_path = (
    "MachineLearningCVE/"
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

label_counts = {}
total_rows = 0

# Read the dataset in smaller chunks to save RAM
for chunk in pd.read_csv(file_path, chunksize=50000):
    chunk.columns = chunk.columns.str.strip()

    counts = chunk["Label"].astype(str).str.strip().value_counts()

    for label, count in counts.items():
        label_counts[label] = label_counts.get(label, 0) + count

    total_rows += len(chunk)

print("\nTOTAL RECORDS:", total_rows)

print("\nATTACK LABEL DISTRIBUTION:")

for label, count in label_counts.items():
    percentage = (count / total_rows) * 100
    print(f"{label}: {count:,} ({percentage:.2f}%)")
