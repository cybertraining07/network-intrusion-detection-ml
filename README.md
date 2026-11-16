# Machine Learning-Based Network Intrusion Detection System

## Project Overview

This project develops and evaluates machine-learning models for network intrusion detection using the CIC-IDS2017 dataset.

The objective is to identify benign network traffic and several categories of malicious activity, evaluate model performance, investigate class imbalance, and examine whether models trained on particular attack families generalize to other attack types.

The project also explores model explainability using SHAP and a second-stage classification approach for investigating false alerts.

**Project status:** Experimental machine-learning research and portfolio project. Not validated for production deployment.

## Objectives

- Explore and preprocess CIC-IDS2017 network traffic data.
- Develop binary and multiclass intrusion detection models.
- Train and compare Random Forest classifiers.
- Investigate duplicate records and class imbalance.
- Evaluate SMOTE for minority-class handling.
- Analyze important network traffic features.
- Apply SHAP to investigate model predictions.
- Evaluate cross-family generalization.
- Investigate a second-stage false-alert classification approach.

## Dataset

The project uses the CIC-IDS2017 dataset.

The dataset contains benign traffic and multiple attack categories, including:

- DDoS
- PortScan
- DoS Hulk
- DoS GoldenEye
- DoS Slowloris
- DoS Slowhttptest
- FTP-Patator
- SSH-Patator
- Bot
- Web Attack Brute Force
- Web Attack XSS
- Web Attack SQL Injection
- Infiltration
- Heartbleed

The dataset contains network-flow features such as packet counts, flow duration, packet lengths, inter-arrival times, TCP window information, and packet rates.

## Technologies

- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Imbalanced-learn / SMOTE
- Matplotlib
- Seaborn
- SHAP
- Joblib

## Machine Learning Models

### 1. Binary DDoS Detection

A Random Forest classifier was trained to distinguish benign traffic from DDoS traffic.

The cleaned dataset contained 39,771 records after duplicate-feature removal.

Reported random test-split accuracy: approximately 99.96%.

### 2. Multiclass Intrusion Detection

A three-class model was developed to classify:

- BENIGN
- DDoS
- PortScan

Reported random test-split accuracy: approximately 99.94%.

### 3. Seven-Class Intrusion Detection

A seven-class Random Forest model was trained using selected DDoS, PortScan, and Wednesday DoS traffic categories.

Reported random test-split accuracy: approximately 99.78%.

### 4. Web Attack Detection

A dedicated model was developed for:

- BENIGN
- Web Attack Brute Force
- Web Attack XSS
- Web Attack SQL Injection

Three approaches were compared:

- Original Random Forest
- Improved Random Forest
- Random Forest with SMOTE

Five-fold cross-validation results:

| Metric | Original RF | SMOTE RF |
|---|---:|---:|
| Accuracy | 89.29% | 88.24% |
| Macro F1 | 70.28% | 71.39% |
| Weighted F1 | 89.50% | 88.64% |
| Macro Precision | 80.72% | 76.89% |
| Macro Recall | 67.13% | 70.84% |

SMOTE improved macro F1 and macro recall, while the original model achieved higher accuracy and weighted F1.

The SQL Injection class contained only 21 records, so its performance estimates are highly uncertain.

## Model Explainability

Feature importance and SHAP were used to investigate which network-flow characteristics influenced model predictions.

Important features included:

- Init_Win_bytes_backward
- Flow Packets/s
- Flow IAT Mean
- Fwd IAT Min
- Flow IAT Max
- Fwd Packets/s
- Flow Duration
- Destination Port

These features indicate model associations, not necessarily causal relationships or universal indicators of malicious activity.

## Cross-Family Generalization

Cross-family experiments evaluated models against attack categories that were not included in their original training classes.

### DDoS Model Against PortScan

The DDoS model achieved approximately 0.05% PortScan recall in the cross-family evaluation.

This demonstrated that a model trained to recognize DDoS traffic did not automatically generalize to PortScan.

### Seven-Class Model Against Wednesday DoS Traffic

The model achieved approximately 0.42% attack recall in the binary evaluation of the separate Wednesday dataset.

### Web Attack Model Against FTP-Patator

The original Web Attack model produced:

- 5 alerts among 1,989
