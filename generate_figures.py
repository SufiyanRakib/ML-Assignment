"""
Script to generate figures from ML Assignment notebook
Generates: Figure 1-7 as PNG images
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Color scheme
DARK_BLUE = '#1f77b4'
LIGHT_BLUE = '#aec7e8'
ORANGE = '#ff7f0e'
GREEN = '#2ca02c'

print("Loading dataset...")
# Load dataset
df = pd.read_csv('heart_2020_cleaned.csv')
print(f'Shape: {df.shape}')

# Preprocessing
print("Preprocessing data...")
df2 = df.copy()
df2['HeartDisease'] = df2['HeartDisease'].map({'Yes': 1, 'No': 0})
cat_cols = df2.select_dtypes(include='object').columns.tolist()
df2 = pd.get_dummies(df2, columns=cat_cols, drop_first=True)

# Train-test split
X = df2.drop('HeartDisease', axis=1)
y = df2['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("Training models...")
# Model Training
svm_lin = SVC(kernel='linear', C=1.0, random_state=42).fit(X_train_s, y_train)
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42).fit(X_train_s, y_train)

y_svm_lin = svm_lin.predict(X_test_s)
y_svm_rbf = svm_rbf.predict(X_test_s)

# KNN with different k values
knn_preds = {}
for k in [3, 5, 11]:
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train_s, y_train)
    knn_preds[k] = knn.predict(X_test_s)

best_k = 5
y_knn = knn_preds[best_k]

# Naive Bayes
gnb = GaussianNB().fit(X_train_s, y_train)
y_gnb = gnb.predict(X_test_s)

print("Generating figures...")

# Figure 1: Class Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Figure 1 — Target Class Distribution', fontsize=13, fontweight='bold', color=DARK_BLUE)

counts = df['HeartDisease'].value_counts()
ax1.bar(['No', 'Yes'], [counts['No'], counts['Yes']], color=[LIGHT_BLUE, ORANGE], edgecolor='black')
ax1.set_ylabel('Count', fontweight='bold')
ax1.set_title('Absolute', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

pcts = df['HeartDisease'].value_counts(normalize=True) * 100
ax2.pie([pcts['No'], pcts['Yes']], labels=['No', 'Yes'], colors=[LIGHT_BLUE, ORANGE],
        autopct='%1.1f%%', startangle=90, textprops={'fontweight': 'bold'})
ax2.set_title('Proportion', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/figure_1_class_distribution.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ Figure 1 saved")

# Figure 2: BMI Distribution by Class
fig, ax = plt.subplots(figsize=(9, 4))
fig.suptitle('Figure 2 — BMI Distribution by Heart Disease Status', fontsize=13, fontweight='bold', color=DARK_BLUE)

bmi_no = df[df['HeartDisease'] == 'No']['BMI']
bmi_yes = df[df['HeartDisease'] == 'Yes']['BMI']
ax.hist(bmi_no, bins=30, alpha=0.6, label='No Disease', color=LIGHT_BLUE, edgecolor='black')
ax.hist(bmi_yes, bins=30, alpha=0.6, label='Disease', color=ORANGE, edgecolor='black')
ax.set_xlabel('BMI', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/figure_2_bmi_distribution.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ Figure 2 saved")

# Figure 3: Heart Disease Rate by Age Category
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Figure 3 — Heart Disease Rate by Age Category', fontsize=13, fontweight='bold', color=DARK_BLUE)

age_cats = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older']
df_tmp = df.copy()
df_tmp['HD_bin'] = (df_tmp['HeartDisease'] == 'Yes').astype(int)
hd_rates = [df_tmp[df_tmp['AgeCategory'] == cat]['HD_bin'].mean() * 100 for cat in age_cats]

ax.plot(age_cats, hd_rates, marker='o', linewidth=2, markersize=8, color=DARK_BLUE)
ax.fill_between(range(len(age_cats)), hd_rates, alpha=0.3, color=LIGHT_BLUE)
ax.set_ylabel('Heart Disease Rate (%)', fontweight='bold')
ax.set_xlabel('Age Category', fontweight='bold')
ax.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/figure_3_age_rate.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ Figure 3 saved")

# Figure 4: Risk Factors
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Figure 4 — Risk Factors Comparison', fontsize=13, fontweight='bold', color=DARK_BLUE)

no = df[df['HeartDisease'] == 'No']
yes = df[df['HeartDisease'] == 'Yes']
factors = ['Smoking', 'Diabetic', 'Physical Activity']
no_vals = [(no['Smoking'] == 'Yes').mean() * 100, 
           (no['Diabetic'] != 'No').mean() * 100,
           (no['PhysicalActivity'] == 'No').mean() * 100]
yes_vals = [(yes['Smoking'] == 'Yes').mean() * 100,
            (yes['Diabetic'] != 'No').mean() * 100,
            (yes['PhysicalActivity'] == 'No').mean() * 100]

x = np.arange(len(factors))
width = 0.35
ax.bar(x - width/2, no_vals, width, label='No Disease', color=LIGHT_BLUE, edgecolor='black')
ax.bar(x + width/2, yes_vals, width, label='Disease', color=ORANGE, edgecolor='black')
ax.set_ylabel('Prevalence (%)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(factors)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/figure_4_risk_factors.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ Figure 4 saved")

# Figure 5: KNN K-value Comparison
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Figure 5 — KNN K-value Comparison', fontsize=13, fontweight='bold', color=DARK_BLUE)

k_vals = [3, 5, 11]
acc_k = [accuracy_score(y_test, knn_preds[k]) for k in k_vals]
rec_k = [recall_score(y_test, knn_preds[k]) for k in k_vals]
f1_k = [f1_score(y_test, knn_preds[k]) for k in k_vals]

x = np.arange(len(k_vals))
width = 0.25
ax.bar(x - width, acc_k, width, label='Accuracy', color=DARK_BLUE, edgecolor='black')
ax.bar(x, rec_k, width, label='Recall', color=ORANGE, edgecolor='black')
ax.bar(x + width, f1_k, width, label='F1-Score', color=GREEN, edgecolor='black')
ax.set_ylabel('Score', fontweight='bold')
ax.set_xlabel('K Value', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(k_vals)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig('figures/figure_5_knn_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ Figure 5 saved")

# Figure 6: Cross-Model Comparison
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Figure 6 — Cross-Model Performance Metrics', fontsize=13, fontweight='bold', color=DARK_BLUE)

model_names = ['SVM\nLinear', 'SVM\nRBF', f'KNN\nK={best_k}', 'Naive\nBayes']
all_preds = [y_svm_lin, y_svm_rbf, y_knn, y_gnb]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for idx, metric in enumerate(metrics):
    values = []
    for preds in all_preds:
        if metric == 'Accuracy':
            values.append(accuracy_score(y_test, preds))
        elif metric == 'Precision':
            values.append(precision_score(y_test, preds))
        elif metric == 'Recall':
            values.append(recall_score(y_test, preds))
        else:  # F1-Score
            values.append(f1_score(y_test, preds))
    
    axes[idx].bar(model_names, values, color=[DARK_BLUE, LIGHT_BLUE, ORANGE, GREEN], edgecolor='black')
    axes[idx].set_ylabel('Score', fontweight='bold')
    axes[idx].set_title(metric, fontweight='bold')
    axes[idx].set_ylim([0, 1])
    axes[idx].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(values):
        axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/figure_6_model_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ Figure 6 saved")

# Figure 7: Confusion Matrices
models = [('SVM (Linear)', y_svm_lin), ('SVM (RBF)', y_svm_rbf), (f'KNN (K={best_k})', y_knn), ('Naive Bayes', y_gnb)]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Figure 7 — Confusion Matrices', fontsize=13, fontweight='bold', color=DARK_BLUE)

for idx, (model_name, preds) in enumerate(models):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    axes[idx].set_title(model_name, fontweight='bold')
    axes[idx].set_ylabel('True Label', fontweight='bold')
    axes[idx].set_xlabel('Predicted Label', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/figure_7_confusion_matrices.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ Figure 7 saved")

print("\n✅ All figures generated successfully!")
print("Figures saved in: figures/ directory")
