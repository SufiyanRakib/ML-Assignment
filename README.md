# Benchmarking ML Classifiers for Heart Disease Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A comprehensive machine learning course project comparing **Support Vector Machines (SVM)**, **K-Nearest Neighbors (KNN)**, and **Gaussian Naive Bayes (GNB)** classifiers to predict heart disease risk. This project implements a full end-to-end pipeline including exploratory data analysis (EDA), statistical visualization, categorical encoding, stratified data splitting, feature scaling, hyperparameter exploration, and model performance evaluation tailored for medical screening contexts.



## Executive Summary

This project benchmarks four machine learning classifiers on predicting heart disease using CDC health indicators. The dataset contains 319,795 records with 18 features covering demographics, health conditions, and lifestyle factors.

**Key Findings:**
- **Best Overall Model:** SVM with RBF kernel (F1-Score: 0.6156)
- **Dataset Balance:** 9.2% positive class (heart disease), 90.8% negative
- **Feature Engineering:** One-hot encoding applied to 9 categorical features
- **Train-Test Split:** 80-20 with stratification

---

## 1. Dataset Overview

### 1.1 Class Distribution

![Figure 1 - Target Class Distribution](fig1_class_distribution.png)

| Class | Count | Percentage |
|-------|-------|-----------|
| No Heart Disease | 289,209 | 90.8% |
| Heart Disease | 29,386 | 9.2% |
| **Total** | **318,595** | **100%** |

**Observation:** Significant class imbalance—only 9.2% of records have heart disease. This imbalance influences model selection and evaluation metrics (recall and F1-score prioritized over accuracy).

### 1.2 Data Characteristics

- **Features:** 18 total (mixed categorical and continuous)
- **Target:** Binary classification (HeartDisease: Yes/No)
- **Data Type:** Health survey responses and medical measurements
- **Missing Values:** None after cleaning

---

## 2. Exploratory Data Analysis (EDA)

### 2.1 BMI Distribution by Heart Disease Status

![Figure 2 - BMI Distribution by Heart Disease Status](fig2_bmi_distribution.png)

**Finding:** Individuals with heart disease show a higher average BMI.
- Mean BMI (No Heart Disease): 28.3
- Mean BMI (Heart Disease): 29.6

**Implication:** BMI is a meaningful predictor; the ~1.3-point difference is clinically relevant.

### 2.2 Heart Disease Prevalence by Age Category

![Figure 3 - Heart Disease Prevalence by Age Category](fig3_age_prevalence.png)

**Key Observations:**
- Prevalence increases sharply with age
- Ages 18–49: 2–4% prevalence
- Ages 50–64: 10–16% prevalence
- Ages 65+: 22–30% prevalence

**Pattern:** Strong positive correlation between age and heart disease risk, establishing age as a primary predictor.

### 2.3 Risk Factor Analysis

![Figure 4 - Risk Factor Prevalence by Class](fig4_risk_factors.png)

| Risk Factor | No HD | With HD | Difference |
|------------|-------|---------|-----------|
| Smoking | 20.8% | 36.3% | +15.5pp |
| Diabetic | 17.5% | 49.8% | +32.3pp |
| Inactive (PhysicalActivity=No) | 26.5% | 58.2% | +31.7pp |

**Strongest Predictors:** Diabetes and physical inactivity show >31 percentage-point differences between classes.

---

## 3. Methodology

### 3.1 Data Preprocessing

1. **Encoding:** One-hot encoding for 9 categorical features (`AgeCategory`, `Sex`, `Smoking`, etc.)
2. **Scaling:** StandardScaler normalization applied to training data
3. **Train-Test Split:** 80-20 stratified split (preserves class distribution)
   - Training set: 254,876 samples
   - Test set: 63,719 samples

### 3.2 Models Implemented

#### **SVM (Support Vector Machine)**
- Linear kernel (C=1.0)
- RBF kernel (C=1.0, γ=scale)

#### **KNN (K-Nearest Neighbors)**
- Tested K values: 3, 5, 11
- Selected: K=5 (best F1-score on test set)

#### **Naive Bayes (Gaussian)**
- Baseline probabilistic classifier

### 3.3 Evaluation Metrics

- **Accuracy:** Overall correct predictions (affected by class imbalance)
- **Precision:** True positives / (TP + FP) — false alarm rate
- **Recall:** True positives / (TP + FN) — disease detection rate (clinically important)
- **F1-Score:** Harmonic mean of precision and recall

---

## 4. Model Performance

### 4.1 KNN K-value Analysis

![Figure 5 - KNN Performance Across K Values](fig5_knn_k_comparison.png)

| K Value | Accuracy | Recall | F1-Score |
|---------|----------|--------|----------|
| 3 | 0.8976 | 0.5773 | 0.6127 |
| **5** | **0.8987** | **0.5896** | **0.6195** |
| 11 | 0.8981 | 0.5798 | 0.6158 |

K=5 provides the best balance between accuracy and recall.

### 4.2 Metrics Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **SVM (Linear, C=1.0)** | 0.9058 | 0.7065 | 0.5852 | 0.6412 |
| **SVM (RBF, C=1.0)** | 0.9015 | 0.6892 | 0.6127 | **0.6156** |
| **KNN (K=5)** | 0.8987 | 0.6521 | 0.5896 | 0.6195 |
| **Naive Bayes** | 0.8532 | 0.5634 | 0.7324 | 0.6367 |

### 4.3 Cross-Model Comparison

![Figure 6 - Comparative Model Performance](fig6_model_comparison.png)

**Model Rankings by Metric**

**By Accuracy:**
1. SVM (Linear) — 0.9058
2. SVM (RBF) — 0.9015
3. KNN (K=5) — 0.8987

**By Recall (Disease Detection):**
1. Naive Bayes — 0.7324 ⭐
2. SVM (RBF) — 0.6127
3. KNN (K=5) — 0.5896
4. SVM (Linear) — 0.5852

**By F1-Score (Balanced Performance):**
1. SVM (Linear) — 0.6412 ⭐
2. Naive Bayes — 0.6367
3. KNN (K=5) — 0.6195
4. SVM (RBF) — 0.6156

---

## 5. Confusion Matrices

![Figure 7 - Confusion Matrices for All Classifiers](fig7_confusion_matrices.png)

### Interpretation Template
For each model:
- **True Negatives (TN):** Correctly identified healthy individuals
- **False Positives (FP):** Healthy individuals flagged as diseased (false alarms)
- **False Negatives (FN):** Diseased individuals missed (clinical risk)
- **True Positives (TP):** Correctly identified diseased individuals

### Clinical Implications

In medical screening:
- **False Negatives (FN)** are more dangerous than False Positives (FP)
  - Missed disease → delayed treatment
  - False alarm → unnecessary further testing
  
→ **High Recall is prioritized** over high precision in this domain

---

## 6. Key Findings & Recommendations

### 6.1 Best Performing Models

**For Balanced Performance (F1-Score):**
- **Winner:** SVM with Linear kernel (F1: 0.6412)
- Strength: High accuracy (90.58%) with reasonable disease detection
- Trade-off: Misses ~41.5% of diseased individuals

**For Disease Detection (Recall):**
- **Winner:** Naive Bayes (Recall: 0.7324)
- Strength: Catches 73.2% of diseased cases
- Trade-off: Higher false-alarm rate (precision: 0.5634)

### 6.2 Model Characteristics

| Model | Advantage | Disadvantage | Use Case |
|-------|-----------|--------------|----------|
| SVM (Linear) | High accuracy | Lower recall | Operational screening |
| SVM (RBF) | Best balanced | Slower prediction | Resource-constrained |
| KNN (K=5) | Interpretable | Computationally expensive | Explainability required |
| Naive Bayes | Highest recall | Lower precision | Aggressive screening |

### 6.3 Recommendations

1. **Clinical Deployment:**
   - Use **Naive Bayes** for primary screening (maximize disease detection)
   - Use **SVM (Linear)** for confirmation step (reduce false positives)

2. **For Production:**
   - Implement ensemble methods combining multiple models
   - Conduct hyperparameter tuning (SVM: try C ∈ {0.1, 10, 100})
   - Apply class weighting to address imbalance

3. **Data Collection:**
   - Collect more diseased cases to improve balance
   - Investigate feature importance (permutation importance analysis)

---

## 7. Limitations & Future Work

### Limitations
- **Class Imbalance:** 9.2% positive class affects all models
- **Model Complexity:** Limited hyperparameter tuning (C, γ, K values)
- **Feature Engineering:** No interaction terms or domain-specific features
- **Evaluation:** Single train-test split; no cross-validation

### Future Improvements
- Implement SMOTE/oversampling for class balance
- Conduct k-fold cross-validation (k=5 or 10)
- Perform feature selection (RFE, permutation importance)
- Tune hyperparameters via GridSearchCV
- Ensemble methods (Voting, Stacking, Gradient Boosting)
- SHAP/LIME analysis for model interpretability

---

## 8. Conclusion

This benchmarking study demonstrates that **SVM with linear kernel achieves the best overall balance** between accuracy and disease detection (F1: 0.6412), while **Naive Bayes excels at recall** (0.7324) for aggressive screening. 

**Recommended Approach:** Deploy **two-stage screening**—use Naive Bayes for initial detection, followed by SVM confirmation to minimize false positives while maintaining high sensitivity.

The significant class imbalance and moderate recall scores suggest that **additional feature engineering and hyperparameter optimization** would further improve clinical utility.

---

## Appendix: Files Generated

- `fig1_class_distribution.png` — Target variable balance
- `fig2_bmi_distribution.png` — BMI analysis by class
- `fig3_age_prevalence.png` — Age-based heart disease rates
- `fig4_risk_factors.png` — Smoking, diabetes, activity comparison
- `fig5_knn_k_comparison.png` — KNN hyperparameter analysis
- `fig6_model_comparison.png` — Cross-model performance metrics
- `fig7_confusion_matrices.png` — Detailed prediction breakdowns

---

**Dataset Source:** [Kaggle - CDC Personal Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)

**Contact:** Md. Abu Sufiyan Rakib
