
# Model Performance and Evaluation Report

## 1. Binary DDoS Detection

| Metric | Result |
|---|---:|
| Dataset records | 40,000 |
| Training records | 32,000 |
| Testing records | 8,000 |
| Accuracy | 99.9375% |
| False positives | 0 |
| False negatives | 5 |

**Assessment:** Strong performance on the random test split. Five DDoS records were missed.

## 2. Multiclass Intrusion Detection

| Class | Test Records | Correct Predictions |
|---|---:|---:|
| BENIGN | 2,973 | 2,973 |
| DDoS | 3,000 | 2,997 |
| PortScan | 2,759 | 2,757 |
| **Total** | **8,732** | **8,727** |

**Accuracy:** 99.94%

## 3. DDoS Model Tested Against PortScan

| Metric | Result |
|---|---:|
| Test records | 20,000 |
| Accuracy | 50.02% |
| PortScan detected | 5 |
| PortScan missed | 9,995 |

**Assessment:** The binary DDoS model does not provide reliable detection of the unseen PortScan attack family.

## 4. Web Attack Cross-Family Model Comparison

| Model | Accuracy | Alert Precision | F1 Score |
|---|---:|---:|---:|
| Original Random Forest | 38.18% | 72.07% | 1.57% |
| Improved Random Forest | 38.34% | 68.39% | 2.71% |
| SMOTE Random Forest | 38.53% | 78.98% | 2.81% |

**Assessment:** SMOTE Random Forest performed best among these three experiments, but cross-family detection remained weak.

## 5. FTP-Patator Evaluation

| Metric | Result |
|---|---:|
| Total records | 3,639 |
| FTP-Patator records | 1,650 |
| FTP-Patator records flagged | 3 |
| FTP-Patator records missed | 1,647 |
| Detection rate | 0.18% |
| BENIGN false alerts | 5 |

## 6. SSH-Patator Investigation

| Metric | Result |
|---|---:|
| Total SSH-Patator records | 1,182 |
| Web Attack alerts | 98 |
| BENIGN predictions | 1,084 |
| Brute Force predictions | 85 |
| XSS predictions | 12 |
| SQL Injection predictions | 1 |

## 7. SSH Second-Stage Filtering

| Metric | Result |
|---|---:|
| Original alerts | 98 |
| Alerts retained | 98 |
| Alerts suppressed | 0 |
| Suppression rate | 0% |

**Assessment:** The second-stage model reproduced the original classifier's alert decisions. It did not demonstrate an actual reduction in false alerts.

## 8. Overall Technical Assessment

The project demonstrates:

- Binary and multiclass Random Forest development.
- Data preprocessing and feature preparation.
- Evaluation using confusion matrices and classification metrics.
- Cross-attack-family generalization testing.
- Feature-importance and SHAP analysis.
- Investigation of false alerts and missed attacks.
- Documentation and version control with GitHub.

## 9. Important Evaluation Limitations

- Random train-test splits may not represent performance on entirely unseen attack families.
- Cross-family results show that high accuracy on one dataset split can be misleading.
- Model-generated labels are not independent ground truth.
- Further testing on independent network traffic is necessary before operational deployment.

## Conclusion

The experiments show that machine learning can achieve strong results on selected intrusion detection tasks, while also revealing substantial generalization challenges.

The project emphasizes careful evaluation, explainability, and honest reporting of limitations.
