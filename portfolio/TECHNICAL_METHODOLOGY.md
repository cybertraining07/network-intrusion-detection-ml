
# Technical Methodology

## 1. Dataset

The project uses the CIC-IDS2017 intrusion detection dataset, which contains network traffic records representing benign activity and multiple attack categories.

The experiments use selected CSV files from the CIC-IDS2017 MachineLearningCSV collection.

## 2. Development Environment

| Component | Technology |
|---|---|
| Operating system | Kali Linux |
| Programming language | Python 3 |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Imbalanced data handling | Imbalanced-learn |
| Explainability | SHAP |
| Visualization | Matplotlib |
| Model persistence | Joblib |
| Version control | Git and GitHub |

## 3. Data Preparation

The data preparation workflow included:

1. Loading selected CIC-IDS2017 CSV files.
2. Inspecting dataset columns and traffic labels.
3. Preparing binary and multiclass datasets.
4. Separating target labels from model features.
5. Preparing training and testing datasets.
6. Checking data quality and class distributions.

The binary DDoS dataset contained 40,000 records, with 20,000 records in each class.

The three-class dataset contained 43,657 records:

- DDoS: 15,000
- BENIGN: 14,862
- PortScan: 13,795

## 4. Model Development

Random Forest was used as the primary classification algorithm.

### Binary Classification

The binary classifier was designed to distinguish:

- BENIGN
- DDoS

The dataset was divided into 32,000 training records and 8,000 testing records.

### Multiclass Classification

A second Random Forest model was developed to classify:

- BENIGN
- DDoS
- PortScan

The model was evaluated on 8,732 test records.

### Web Attack Model Experiments

Additional Random Forest experiments investigated Web Attack classification.

Three approaches were compared:

- Original Random Forest
- Improved Random Forest
- SMOTE Random Forest

The comparison examined accuracy, alert precision, F1 score, and false-positive rate.

## 5. Cross-Family Evaluation

Cross-family experiments were conducted to investigate how well models performed on attack categories different from their original training categories.

Experiments included:

- DDoS model against PortScan traffic.
- Web Attack model against other attack traffic.
- FTP-Patator evaluation.
- SSH-Patator investigation.

These tests exposed substantial limitations in generalization beyond the original training categories.

## 6. Model Explainability

Feature importance and SHAP analysis were used to investigate which network-flow characteristics contributed to classification predictions.

Examples of investigated features included:

- Flow Duration
- Flow Packets/s
- Flow IAT Mean
- Flow IAT Max
- Forward and backward inter-arrival time features
- Initial TCP window-size features

These analyses support interpretation of model behavior but do not establish causal relationships.

## 7. Error Investigation

The project included investigations into:

- False alerts on SSH-Patator traffic.
- Missed FTP-Patator records.
- Cross-family classification errors.
- Differences between alerted and non-alerted traffic.

A second-stage classifier was also trained using labels derived from the original model's predictions.

This experiment demonstrated why model-generated labels should not be treated as independent ground truth.

## 8. Model Evaluation

The following metrics and evaluation methods were used:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix
- False-positive rate
- Cross-family testing

The project emphasizes evaluating individual attack-class performance rather than relying on accuracy alone.

## 9. Reproducibility and Version Control

Python scripts were created for dataset preparation, model training, evaluation, feature analysis, and reporting.

Git was used to track project changes, while GitHub provides the remote source-code repository.

Repository:

https://github.com/cybertraining07/network-intrusion-detection-ml

The original dataset, generated CSV files, and trained model artifacts are maintained locally rather than being included in the public repository.

## 10. Methodological Limitations

- Results from random train-test splits may overestimate performance on unseen traffic.
- Cross-family evaluation showed weak generalization in several experiments.
- Some experiments use labels derived from model predictions rather than independently verified ground truth.
- Further independent testing is required before operational deployment.

## Conclusion

The methodology demonstrates a practical machine learning engineering workflow for cybersecurity: data preparation, model development, evaluation, explainability, error investigation, and reproducible project documentation.
