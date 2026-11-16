
import pandas as pd

source = "MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv"

benign_samples = []
ftp_samples = []

for chunk in pd.read_csv(source, chunksize=50000, low_memory=False):

    chunk.columns = chunk.columns.str.strip()
    chunk["Label"] = chunk["Label"].astype(str).str.strip()

    benign = chunk[chunk["Label"] == "BENIGN"]
    ftp = chunk[chunk["Label"] == "FTP-Patator"]

    if len(benign_samples) < 2000:
        remaining = 2000 - len(benign_samples)
        benign_samples.extend(
            benign.head(remaining).to_dict("records")
        )

    if len(ftp_samples) < 2000:
        remaining = 2000 - len(ftp_samples)
        ftp_samples.extend(
            ftp.head(remaining).to_dict("records")
        )

    if len(benign_samples) >= 2000 and len(ftp_samples) >= 2000:
        break

dataset = pd.DataFrame(benign_samples + ftp_samples)

dataset = dataset.drop_duplicates(
    subset=dataset.columns.drop("Label").tolist()
)

dataset.to_csv("ftp_crossfamily_test.csv", index=False)

print("\nFTP CROSS-FAMILY TEST DATASET")
print("=" * 45)
print("Dataset shape:", dataset.shape)
print("\nClass distribution:")
print(dataset["Label"].value_counts())
print("\nSaved: ftp_crossfamily_test.csv")
