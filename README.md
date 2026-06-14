# 🩺 Heart Disease Prediction: Benchmarking ML Classifiers

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A comprehensive machine learning course project (PMIT_6121) comparing **Support Vector Machines (SVM)**, **K-Nearest Neighbors (KNN)**, and **Gaussian Naive Bayes (GNB)** classifiers to predict heart disease risk. This project implements a full end-to-end pipeline including exploratory data analysis (EDA), statistical visualization, categorical encoding, stratified data splitting, feature scaling, hyperparameter exploration, and model performance evaluation tailored for medical screening contexts.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset & Features](#-dataset--features)
3. [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
4. [Data Preprocessing Pipeline](#-data-preprocessing-pipeline)
5. [Model Training & Benchmarking](#-model-training--benchmarking)
6. [Key Findings & Performance Analysis](#-key-findings--performance-analysis)
7. [Clinical Recommendation](#-clinical-recommendation)
8. [Project Structure](#-project-structure)
9. [Installation & Usage](#-installation--usage)
10. [Future Directions](#-future-directions)

---

## 🔍 Project Overview

Predicting heart disease is one of the most critical applications of machine learning in healthcare. Early detection allows for preventative lifestyle changes and clinical interventions, which can dramatically lower mortality rates. 

This project benchmarks three distinct classification paradigms on the Kaggle *Personal Key Indicators of Heart Disease* dataset:
*   **Distance-based Classifiers:** K-Nearest Neighbors (KNN)
*   **Boundary-optimizing Classifiers:** Support Vector Machines (SVM) with Linear and RBF Kernels
*   **Probabilistic Classifiers:** Gaussian Naive Bayes (GNB)

A core focus of this project is addressing the severe **class imbalance** (only ~9.2% of patients have heart disease) and understanding the clinical tradeoffs between **Precision** and **Recall**.

---

## 📊 Dataset & Features

The project utilizes the **Personal Key Indicators of Heart Disease** dataset from Kaggle, containing **319,795 records** collected from the 2020 CDC BRFSS telephone survey.

### Dataset Overview

| Attribute | Value |
| :--- | :--- |
| **Total Records** | 319,795 |
| **Total Features** | 18 (mixed categorical & continuous) |
| **Target Variable** | HeartDisease (Binary: Yes/No) |
| **Positive Class Distribution** | 9.2% (29,386 cases) |
| **Negative Class Distribution** | 90.8% (289,209 cases) |
| **Missing Values** | None (pre-cleaned) |

### Key Features

- **Demographic:** AgeCategory, Sex, Race
- **Health Conditions:** Stroke, Diabetic, KidneyDisease, SkinCancer, Asthma
- **Lifestyle:** Smoking, AlcoholDrinking, PhysicalActivity
- **Vital Measurements:** BMI, SleepTime
- **Health Status:** GenHealth, PhysicalHealth, MentalHealth, DiffWalking

---

## 📈 Exploratory Data Analysis (EDA)

### 1. Target Class Distribution

| Visualization | Analysis |
| :---: | :--- |
| ![Figure 1 - Target Class Distribution](fig1_class_distribution.png) | **Significant Class Imbalance:** 91.44% negative class vs 8.56% positive class. This imbalance critically impacts classifier evaluation—accuracy alone is a deceptive metric. Recall and F1-score become primary evaluation metrics. |

**Key Insight:** Standard classifiers will naturally bias toward predicting "No Disease" due to class imbalance. Mitigation strategies include class weighting, SMOTE, or threshold tuning.

---

### 2. BMI Distribution Analysis

| Visualization | Analysis |
| :---: | :--- |
| ![Figure 2 - BMI Distribution by Heart Disease Status](fig2_bmi_distribution.png) | **BMI as a Predictor:** Individuals with heart disease show a mean BMI of 29.6 vs 28.3 for those without. The 1.3-point difference is clinically relevant. Both distributions are right-skewed, with extreme outliers reaching BMI ≥ 90. |

**Clinical Relevance:** BMI ≥ 25 (overweight) is a known cardiovascular risk factor.

---

### 3. Heart Disease Prevalence by Age Category

| Visualization | Analysis |
| :---: | :--- |
| ![Figure 3 - Heart Disease Prevalence by Age Category](fig3_age_prevalence.png) | **Strong Age Correlation:** Prevalence increases dramatically with age—from ~2% in ages 18–24 to ~27% in ages 80+. Age is the single strongest predictor in the dataset. |

**Observation:** Age categories form a natural segmentation for stratified analysis and model interpretation.

---

### 4. Risk Factor Analysis

| Visualization | Analysis |
| :---: | :--- |
| ![Figure 4 - Risk Factor Prevalence by Class](fig4_risk_factors.png) | **Comparative Risk Factors:** Smoking (36.3% vs 20.8%), Diabetes (49.8% vs 17.5%), and Physical Inactivity (58.2% vs 26.5%) all show pronounced differences between disease and non-disease groups. Diabetes and inactivity demonstrate >31 percentage-point differences. |

**Clinical Implication:** These categorical features are highly discriminative and should receive high feature importance weights.

---

## ⚙️ Data Preprocessing Pipeline

To prepare the data for distance-based (KNN) and boundary-based (SVM) classifiers, a robust preprocessing pipeline was implemented:

1.  **Target Encoding:** Mapped `HeartDisease` binary values: `Yes` → `1`, `No` → `0`
2.  **One-Hot Encoding (OHE):** Applied to 9 categorical features (`AgeCategory`, `Sex`, `Smoking`, `Diabetic`, etc.)
   - Used `drop_first=True` to prevent multicollinearity and the dummy variable trap
3.  **Stratified Train/Test Split:** 
   - **80% Training** (254,876 samples) / **20% Testing** (63,719 samples)
   - Used `stratify=y` to preserve 9.2% class balance in both subsets
4.  **Feature Scaling (StandardScaler):** 
   - **Applied to:** SVM and KNN (distance-based metrics require normalization)
   - **Not applied to:** Gaussian Naive Bayes (models probability densities directly)
   - **Data Leakage Prevention:** Scaler fitted *only* on training data

---

## 🏆 Model Training & Benchmarking

### 4.1 KNN Hyperparameter Analysis

| Visualization | Analysis |
| :---: | :--- |
| ![Figure 5 - KNN Performance Across K Values](fig5_knn_k_comparison.png) | **K-value Trade-off:** K=5 provides optimal balance between accuracy (89.87%) and recall (58.96%). As K increases, accuracy improves but recall decreases due to majority class domination in larger neighborhoods. |

**Selected Configuration:** K=5 (best F1-Score: 0.6195)

---

### 4.2 Consolidated Performance Table

| Model | Accuracy | Precision | Recall | F1-Score | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **SVM (Linear, C=1.0)** | 90.58% | 70.65% | 58.52% | **0.6412** ⭐ | Best F1-Score |
| **SVM (RBF, C=1.0)** | 90.15% | 68.92% | 61.27% | 0.6156 | Balanced kernel |
| **KNN (K=5)** | 89.87% | 65.21% | 58.96% | 0.6195 | Interpretable |
| **Naive Bayes** | 85.32% | 56.34% | **73.24%** | 0.6367 | Highest Recall ⭐ |

---

### 4.3 Cross-Model Performance Comparison

| Visualization | Analysis |
| :---: | :--- |
| ![Figure 6 - Comparative Model Performance](fig6_model_comparison.png) | **Multi-Metric Comparison:** SVM (Linear) dominates in accuracy and F1-score. Naive Bayes excels at recall (73.24%), catching more disease cases despite lower precision. SVM (RBF) provides a middle ground with balanced performance. |

**Ranking by Metric:**
- **Accuracy:** SVM (Linear) 90.58% > SVM (RBF) 90.15% > KNN 89.87%
- **Recall:** Naive Bayes 73.24% > SVM (RBF) 61.27% > KNN 58.96%
- **F1-Score:** SVM (Linear) 0.6412 > Naive Bayes 0.6367 > KNN 0.6195

---

## 5. Confusion Matrices & Detailed Analysis

| Visualization |
| :---: |
| ![Figure 7 - Confusion Matrices for All Classifiers](fig7_confusion_matrices.png) |

### Interpretation Guide

For each model's confusion matrix:
- **True Negatives (TN):** ✅ Correctly identified healthy individuals
- **False Positives (FP):** ⚠️ Healthy individuals flagged as diseased (unnecessary follow-up)
- **False Negatives (FN):** 🚨 Diseased individuals missed (clinical risk—**MOST CRITICAL**)
- **True Positives (TP):** ✅ Correctly identified diseased individuals

### Clinical Implications

In medical screening, **False Negatives are 10x more dangerous than False Positives:**
- **False Negative (FN):** Patient sent home with undetected disease → heart attack, mortality
- **False Positive (FP):** Patient receives secondary diagnostic tests (ECG, angiogram) → quickly corrected

→ **Recall is prioritized over Precision in this domain**

---

## 🩺 Clinical Recommendations

### Primary Recommendation for Screening
**Use Naive Bayes for initial triage screening:**
- **Recall: 73.24%** — Catches nearly 3 out of 4 disease cases
- **Speed:** Extremely fast inference, suitable for high-throughput screening
- **Simplicity:** No hyperparameter tuning required

### Confirmation Strategy (Two-Stage Approach)
1. **Stage 1 (Triage):** Naive Bayes flags suspicious cases
2. **Stage 2 (Confirmation):** SVM (Linear) confirms with high precision (70.65%)
   - This dual approach minimizes both false negatives and false positives

### Model-Specific Use Cases

| Model | Use Case | Rationale |
| :--- | :--- | :--- |
| **SVM (Linear)** | Operational screening in resource-rich settings | High accuracy (90.58%) & precision (70.65%) |
| **Naive Bayes** | Primary screening / mass population surveys | Highest recall (73.24%), no scaling required |
| **KNN (K=5)** | When model interpretability is critical | Distance-based predictions are easily explainable |
| **SVM (RBF)** | Research/academic comparisons | Balanced performance across all metrics |

---

## 📂 Project Structure

```bash
ML-Assignment/
├── ML Assignment_Rakib.ipynb          # Core Jupyter Notebook with full pipeline
├── README.md                          # Project documentation (this file)
├── heart_2020_cleaned.csv             # CDC BRFSS Dataset (319,795 records)
│
├── [Generated Visualizations]
│   ├── fig1_class_distribution.png    # Pie & bar charts of target distribution
│   ├── fig2_bmi_distribution.png      # Overlaid histograms by disease status
│   ├── fig3_age_prevalence.png        # Bar chart of prevalence by age group
│   ├── fig4_risk_factors.png          # Grouped bar chart of risk factors
│   ├── fig5_knn_k_comparison.png      # Line/bar plot of K-value performance
│   ├── fig6_model_comparison.png      # 4-subplot metric comparison
│   └── fig7_confusion_matrices.png    # 4-heatmap confusion matrix grid
│
└── requirements.txt                   # Python dependencies
