
import pandas as pd

results = {
    "Model": [
        "Original Random Forest",
        "Improved Random Forest"
    ],
    "Accuracy Mean": [0.8929, 0.8860],
    "Accuracy Std": [0.0044, 0.0026],
    "Macro F1 Mean": [0.7028, 0.6806],
    "Macro F1 Std": [0.0379, 0.0174],
    "Weighted F1 Mean": [0.8950, 0.8895]
}

df = pd.DataFrame(results)

df.to_csv(
    "webattack_model_comparison.csv",
    index=False
)

print("Model comparison saved successfully!")
print(df.to_string(index=False))

import pandas as pd

results = {
    "Metric": [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted F1"
    ],
    "Original Random Forest": [
        "89.29% +/- 0.44%",
        "80.72% +/- 0.73%",
        "67.13% +/- 4.05%",
        "70.28% +/- 3.79%",
        "89.50% +/- 0.44%"
    ],
    "SMOTE Random Forest": [
        "88.24% +/- 0.34%",
        "76.89% +/- 4.22%",
        "70.84% +/- 6.86%",
        "71.39% +/- 5.11%",
        "88.64% +/- 0.37%"
    ]
}

df = pd.DataFrame(results)

df.to_csv("webattack_model_comparison.csv", index=False)

print("Comparison report saved successfully!")
print(df.to_string(index=False))
