import pandas as pd
import glob
import os

files = glob.glob("MachineLearningCVE/*.csv")

for file in sorted(files):

    counts = {}
    total = 0

    for chunk in pd.read_csv(file, chunksize=50000):

        chunk.columns = chunk.columns.str.strip()

        labels = chunk["Label"].astype(str).str.strip()
        current_counts = labels.value_counts()

        for label, count in current_counts.items():
            counts[label] = counts.get(label, 0) + count

        total += len(chunk)

    print("\nFILE:", os.path.basename(file))
    print("TOTAL RECORDS:", total)

    for label, count in counts.items():
        print(f"{label}: {count:,} ({count / total * 100:.2f}%)")
