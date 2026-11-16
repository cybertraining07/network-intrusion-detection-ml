
import pandas as pd

file_path = (
    "MachineLearningCVE/"
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

# Read only five rows initially
df = pd.read_csv(file_path, nrows=5)

print("\nDATASET PREVIEW")
print(df.head())

print("\nDATASET COLUMNS")
print(df.columns.tolist())

print("\nNUMBER OF COLUMNS")
print(len(df.columns))

print("\nLABEL COLUMN")
label_column = [
    col for col in df.columns
    if col.strip().lower() == "label"
]

if label_column:
    print("Found:", repr(label_column[0]))
    print("Sample labels:")
    print(df[label_column[0]].value_counts())
else:
    print("Label column not found.")
