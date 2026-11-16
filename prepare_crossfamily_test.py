
import pandas as pd
from pathlib import Path

folder = Path("MachineLearningCVE")

files = {
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": [
        "BENIGN", "DDoS"
    ],
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": [
        "BENIGN", "PortScan"
    ],
    "Wednesday-workingHours.pcap_ISCX.csv": [
        "BENIGN", "DoS Hulk", "DoS GoldenEye",
        "DoS slowloris", "DoS Slowhttptest"
    ],
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": [
        "BENIGN", "Bot"
    ],
    "Tuesday-WorkingHours.pcap_ISCX.csv": [
        "BENIGN", "FTP-Patator", "SSH-Patator"
    ]
}

sample_size = 2000
samples = []

for filename, selected_labels in files.items():

    path = folder / filename
    print(f"\nProcessing: {filename}")

    # Collect a maximum of sample_size records per label
    collected = {label: [] for label in selected_labels}
    counts = {label: 0 for label in selected_labels}

    for chunk in pd.read_csv(path, chunksize=50000):

        chunk.columns = chunk.columns.str.strip()
        chunk["Label"] = chunk["Label"].astype(str).str.strip()

        for label in selected_labels:

            remaining = sample_size - counts[label]

            if remaining <= 0:
                continue

            matching = chunk[chunk["Label"] == label]

            if not matching.empty:
                selected = matching.sample(
                    n=min(len(matching), remaining),
                    random_state=42
                )

                collected[label].append(selected)
                counts[label] += len(selected)

    for label in selected_labels:
        if collected[label]:
            samples.extend(collected[label])

# Combine all selected records
dataset = pd.concat(samples, ignore_index=True)

# Remove duplicate feature records
dataset = dataset.drop_duplicates()

# Save evaluation dataset
dataset.to_csv("crossfamily_test.csv", index=False)

print("\nCROSS-FAMILY DATASET CREATED")
print("Dataset shape:", dataset.shape)

print("\nClass distribution:")
print(dataset["Label"].value_counts())

print("\nSaved: crossfamily_test.csv")
