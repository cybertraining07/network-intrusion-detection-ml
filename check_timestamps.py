import pandas as pd

FILE = (
    "MachineLearningCVE/"
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

# Read only the timestamp column to save memory
df = pd.read_csv(
    FILE,
    usecols=lambda col: col.strip() == "Timestamp"
)

df.columns = df.columns.str.strip()

timestamps = pd.to_datetime(
    df["Timestamp"],
    errors="coerce"
)

boundary = int(len(timestamps) * 0.70)

print("Total timestamps:", len(timestamps))
print("Missing timestamps:", timestamps.isna().sum())

print("\nOriginal timestamp order:")
print("Chronologically sorted:", timestamps.is_monotonic_increasing)

print("\nTraining period:")
print("Start:", timestamps.iloc[0])
print("End:", timestamps.iloc[boundary - 1])

print("\nTesting period:")
print("Start:", timestamps.iloc[boundary])
print("End:", timestamps.iloc[-1])
