
import pandas as pd

results = [
    {
        "Dataset": "SSH-Patator",
        "Model": "Original Web Attack Model",
        "Benign Alert Rate (%)": 8.29,
        "Attack Alert Rate (%)": 8.29
    },
    {
        "Dataset": "FTP-Patator",
        "Model": "Original Web Attack Model",
        "Benign Alert Rate (%)": 0.25,
        "Attack Alert Rate (%)": 0.18
    },
    {
        "Dataset": "FTP-Patator",
        "Model": "SSH Second-Stage Model",
        "Benign Alert Rate (%)": 10.81,
        "Attack Alert Rate (%)": 0.30
    }
]

df = pd.DataFrame(results)

df.to_csv("crossfamily_experiment_summary.csv", index=False)

print("\nCROSS-FAMILY EXPERIMENT SUMMARY")
print("=" * 55)
print(df.to_string(index=False))
print("\nSaved: crossfamily_experiment_summary.csv")
