# Fraud Detection Case Study - Real-World ML Application

**Dataset**: Anti-Fraud (AF) Dataset  
**Problem Type**: Binary Classification with Extreme Class Imbalance  
**Fraud Rate**: 0.78% (1:127 ratio - highly imbalanced!)  
**Notebooks**: 05_fraud_data_processing.ipynb, 06_fraud_model_training.ipynb, 07_hyperparameter_tuning.ipynb

---

## Table of Contents

1. [Problem Overview](#problem-overview)
2. [Data Processing Pipeline](#data-processing-pipeline)
3. [Feature Engineering](#feature-engineering)
4. [Handling Class Imbalance](#handling-class-imbalance)
5. [Model Training & Evaluation](#model-training--evaluation)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Performance Expectations](#performance-expectations)
8. [Production Deployment](#production-deployment)
9. [Key Learnings](#key-learnings)

---

## Problem Overview

### Why Fraud Detection is Hard

**Challenges**:

- ⚠️ **Extreme Class Imbalance**: Only 0.78% of transactions are fraudulent (1:127 ratio)
- 🎯 **Adversarial Nature**: Fraudsters actively adapt their tactics to evade detection
- 🔍 **Limited Context**: Missing device fingerprints, location data, user history
- 🤝 **Similar Patterns**: Legitimate and fraudulent transactions often look very similar
- 💰 **High Cost of Error**: False negatives (missed fraud) are very costly

**Dataset Characteristics**:

```text
Total Transactions: ~20,000-30,000
Legitimate: 99.22%
Fraudulent: 0.78%
Class Ratio: 127:1

Feature Types:
- Transaction details (amount, payment method)
- Merchant information (IDs, categories)
- Temporal data (timestamp, hour, day)
- Response codes and statuses
```

### Setting Realistic Expectations

**Performance Targets for 0.78% Fraud Rate**:

| Metric | Baseline | Good | Excellent | Unrealistic |
| -------- | ---------- | ------ | ----------- | ------------- |
| **F1-Score** | 60-65% | 65-75% | 75-85% | >95% (likely data leakage!) |
| **Precision** | 50-60% | 60-70% | 70-80% | >90% |
| **Recall** | 85-90% | 90-95% | 95%+ | 100% |
| **ROC-AUC** | 0.85-0.90 | 0.90-0.95 | 0.95+ | >0.99 |

**⚠️ Red Flags Indicating Data Leakage**:

- F1-Score > 95% on test set
- Perfect train accuracy (100%) with high test accuracy
- Recall < 80% (missing too many frauds)
- Test performance significantly better than cross-validation

**Why These Targets?**

- Extreme imbalance makes high precision/recall difficult
- Fraudulent and legitimate transactions share similar features
- Missing context data (device, location, behavioral history)
- Fraudsters constantly evolve tactics

---

## Data Processing Pipeline

### Step 1: Initial Data Inspection

```python
import pandas as pd
import numpy as np

# Load raw data
df = pd.read_csv('../data/af_dataset.csv')

print(f'Dataset shape: {df.shape}')
print(f'\nTarget distribution:')
print(df['is_fraud'].value_counts())
print(f'\nFraud rate: {df["is_fraud"].mean() * 100:.2f}%')

# Check for missing values
print(f'\nMissing values:')
print(df.isnull().sum())

# Data types
print(f'\nData types:')
print(df.dtypes)
```

### Step 2: Handle Missing Values

**Strategy based on feature type**:

```python
from sklearn.impute import SimpleImputer

# Numerical features: Impute with median (robust to outliers)
numerical_features = df.select_dtypes(include=[np.number]).columns
imputer_num = SimpleImputer(strategy='median')
df[numerical_features] = imputer_num.fit_transform(df[numerical_features])

# Categorical features: Impute with mode (most frequent)
categorical_features = df.select_dtypes(include=['object']).columns
imputer_cat = SimpleImputer(strategy='most_frequent')
df[categorical_features] = imputer_cat.fit_transform(df[categorical_features])

# Or create 'Unknown' category for missing categorical values
for col in categorical_features:
    df[col] = df[col].fillna('UNKNOWN')
```

### Step 3: Identify and Exclude Non-Predictive Columns

**Columns to exclude**:

- **ID columns**: `payload_id`, `request_id` (no predictive value)
- **Target variable**: `is_fraud` (what we're predicting)
- **Outcome-related**: `response`, `response_code` (leakage - result of fraud check)
- **Timestamps**: `created_at` (use derived features instead)

```python
# Exclude columns
exclude_cols = [
    'payload_id',      # Unique identifier
    'request_id',      # Unique identifier
    'is_fraud',        # Target variable
    'response',        # Outcome (leakage!)
    'response_code',   # Outcome (leakage!)
    'created_at'       # Raw timestamp (use derived features)
]

# Remove datetime columns
datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
exclude_cols.extend(datetime_cols)

# Separate features and target
X = df.drop(columns=exclude_cols)
y = df['is_fraud']
```

---

## Feature Engineering

### Temporal Features

```python
# Convert timestamp to datetime
df['created_at'] = pd.to_datetime(df['created_at'])

# Extract time-based features
df['hour_of_day'] = df['created_at'].dt.hour
df['day_of_week'] = df['created_at'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_night'] = ((df['hour_of_day'] < 6) | (df['hour_of_day'] >= 22)).astype(int)
df['day_of_month'] = df['created_at'].dt.day
df['month'] = df['created_at'].dt.month

# Business hours flag (9 AM - 5 PM, weekdays)
df['is_business_hours'] = (
    (df['hour_of_day'] >= 9) & 
    (df['hour_of_day'] <= 17) & 
    (df['day_of_week'] < 5)
).astype(int)
```

### Transaction Amount Features

```python
# Amount statistics
df['amount_log'] = np.log1p(df['amount'])  # Log transform for skewed amounts

# Amount percentile by payment method
df['amount_percentile'] = df.groupby('payment_method')['amount'].transform(
    lambda x: x.rank(pct=True)
)

# Rounded amounts (fraudsters often use round numbers)
df['amount_is_rounded'] = (df['amount'] % 1 == 0).astype(int)
df['amount_is_multiple_10'] = (df['amount'] % 10 == 0).astype(int)
df['amount_is_multiple_100'] = (df['amount'] % 100 == 0).astype(int)
```

### User Behavior Features

**⚠️ CRITICAL**: Calculate aggregations on training data only to prevent data leakage!

```python
# Calculate user-level statistics on TRAINING data only
def create_user_features(X_train, X_test, y_train):
    """
    Create user behavior features without data leakage
    """
    # Combine training features and target for aggregation
    train_df = X_train.copy()
    train_df['is_fraud'] = y_train.values
    
    # User statistics (calculated on training data only)
    user_stats = train_df.groupby('user_id').agg({
        'amount': ['mean', 'std', 'min', 'max', 'count'],
        'is_fraud': 'mean'  # User fraud rate
    }).fillna(0)
    
    user_stats.columns = [
        'user_avg_amount', 'user_std_amount', 'user_min_amount',
        'user_max_amount', 'user_transaction_count', 'user_fraud_rate'
    ]
    
    # Apply to training data
    X_train_enhanced = X_train.merge(
        user_stats, 
        left_on='user_id', 
        right_index=True, 
        how='left'
    ).fillna(0)
    
    # Apply to test data (unseen users get 0)
    X_test_enhanced = X_test.merge(
        user_stats, 
        left_on='user_id', 
        right_index=True, 
        how='left'
    ).fillna(0)
    
    # Deviation from user's typical amount
    X_train_enhanced['amount_deviation'] = (
        (X_train_enhanced['amount'] - X_train_enhanced['user_avg_amount']) / 
        (X_train_enhanced['user_std_amount'] + 1e-6)
    )
    
    X_test_enhanced['amount_deviation'] = (
        (X_test_enhanced['amount'] - X_test_enhanced['user_avg_amount']) / 
        (X_test_enhanced['user_std_amount'] + 1e-6)
    )
    
    return X_train_enhanced, X_test_enhanced

# Apply after train-test split
X_train, X_test = create_user_features(X_train, X_test, y_train)
```

### Merchant Features

```python
# Merchant statistics (on training data only)
merchant_stats = train_df.groupby('merchant_id').agg({
    'amount': ['mean', 'count'],
    'is_fraud': 'mean'  # Merchant fraud rate
}).fillna(0)

merchant_stats.columns = [
    'merchant_avg_amount', 'merchant_transaction_count', 'merchant_fraud_rate'
]
```

---

## Encoding Categorical Variables

### Strategy: Low vs High Cardinality

```python
# Separate by cardinality
low_cardinality_cols = []
high_cardinality_cols = []

for col in categorical_features:
    n_unique = X[col].nunique()
    if n_unique <= 10:
        low_cardinality_cols.append(col)
    else:
        high_cardinality_cols.append(col)
```

### Label Encoding (Low Cardinality ≤10 unique values)

```python
from sklearn.preprocessing import LabelEncoder

label_encoders = {}

for col in low_cardinality_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le
```

### Target Encoding (High Cardinality >10 unique values)

**⚠️ CRITICAL**: Must be done AFTER train-test split to prevent data leakage!

```python
def target_encode_feature(X_train, X_test, y_train, col, min_samples=10):
    """
    Target encoding with smoothing to prevent overfitting
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        col: Column to encode
        min_samples: Minimum samples before trusting category mean
    
    Returns:
        X_train, X_test: Encoded features
        encoding_dict: Encoding mapping for production
    """
    # Calculate fraud rate per category (on training data only)
    fraud_rates = pd.DataFrame({
        'category': X_train[col],
        'target': y_train.values
    })
    
    global_mean = y_train.mean()
    category_stats = fraud_rates.groupby('category')['target'].agg(['mean', 'count'])
    
    # Apply smoothing: weighted average of category mean and global mean
    # More samples → trust category mean more
    # Fewer samples → trust global mean more
    category_stats['smoothed_mean'] = (
        (category_stats['mean'] * category_stats['count'] + global_mean * min_samples) /
        (category_stats['count'] + min_samples)
    )
    
    encoding_dict = category_stats['smoothed_mean'].to_dict()
    
    # Apply encoding
    X_train[col] = X_train[col].map(encoding_dict).fillna(global_mean)
    X_test[col] = X_test[col].map(encoding_dict).fillna(global_mean)
    
    return X_train, X_test, {'encoding_dict': encoding_dict, 'global_mean': global_mean}

# Apply to all high cardinality features
target_encoders = {}
for col in high_cardinality_cols:
    X_train, X_test, encoder_info = target_encode_feature(
        X_train, X_test, y_train, col
    )
    target_encoders[col] = encoder_info
```

---

## Handling Class Imbalance

### Problem: Naive Approach (WRONG!)

```python
# ❌ WRONG - DATA LEAKAGE!
from imblearn.over_sampling import SMOTE

# This creates data leakage because validation folds see synthetic samples
# that are "neighbors" of training samples
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Cross-validation on balanced data → LEAKAGE!
cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5)
# Result: Artificially inflated scores (e.g., 99.99% F1)
```

**Why this is wrong**:

1. SMOTE creates synthetic samples by interpolating between existing samples
2. When you balance FIRST, then split for CV, validation folds contain synthetic samples
3. These synthetic samples are "neighbors" of training samples
4. Model sees leaked information → unrealistic performance

### Solution: SMOTE Inside CV Pipeline (CORRECT!)

```python
# ✅ CORRECT - NO LEAKAGE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier

# Create pipeline
smote_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42, sampling_strategy='auto')),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# SMOTE applied separately in each fold
cv_scores = cross_val_score(smote_pipeline, X_train, y_train, cv=5, scoring='f1')
# Result: Realistic scores (e.g., 70% F1)
```

**How it works**:

```text
For each CV fold:
  1. Split data → 80% train, 20% validation (UNBALANCED)
  2. Apply SMOTE to training portion only
  3. Train model on balanced training data
  4. Evaluate on UNBALANCED validation data
  5. Validation data never used in SMOTE → no leakage!

Fold 1: [SMOTE→Train][Train][Train][Train][Val] → F1: 0.68
Fold 2: [Train][SMOTE→Train][Train][Train][Val] → F1: 0.72
Fold 3: [Train][Train][SMOTE→Train][Train][Val] → F1: 0.70
...
Average: 0.70 ± 0.02 (realistic!)
```

### Alternative Strategies

#### 1. Class Weights (No Oversampling)

```python
# For algorithms that support class_weight
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
    # Note: GradientBoostingClassifier doesn't have class_weight parameter
    # Use sample_weight in fit() instead
)

# Calculate sample weights
sample_weights = np.where(
    y_train == 1,
    len(y_train) / (2 * (y_train == 1).sum()),  # Weight for fraud
    len(y_train) / (2 * (y_train == 0).sum())   # Weight for legitimate
)

model.fit(X_train, y_train, sample_weight=sample_weights)
```

#### 2. Different SMOTE Strategies

```python
# SMOTE variations
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE

# Standard SMOTE: Balance to 1:1 ratio
smote = SMOTE(random_state=42, sampling_strategy='auto')

# Partial balance: Balance to 1:3 ratio (fraud:legitimate)
smote = SMOTE(random_state=42, sampling_strategy=0.33)

# ADASYN: Focus on harder-to-learn samples
smote = ADASYN(random_state=42)

# Borderline SMOTE: Only create samples near decision boundary
smote = BorderlineSMOTE(random_state=42)
```

---

## Model Training & Evaluation

### Evaluation Metrics for Imbalanced Data

**⚠️ DON'T USE ACCURACY!**

```python
# Accuracy is misleading for imbalanced data
# If 99.22% of transactions are legitimate:
# A model that predicts "legitimate" for everything gets 99.22% accuracy!
# But catches 0% of fraud (useless!)
```

**Use These Metrics Instead**:

```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Key metrics
precision = precision_score(y_test, y_pred)  # Of flagged cases, how many are fraud?
recall = recall_score(y_test, y_pred)        # Of all frauds, how many did we catch?
f1 = f1_score(y_test, y_pred)                # Harmonic mean (balanced metric)
roc_auc = roc_auc_score(y_test, y_pred_proba)  # Overall discrimination ability

print(f'Precision: {precision:.4f}')  # Target: 60-70%
print(f'Recall: {recall:.4f}')        # Target: 90-95%
print(f'F1-Score: {f1:.4f}')          # Target: 65-75%
print(f'ROC-AUC: {roc_auc:.4f}')      # Target: 0.90-0.95
```

### Confusion Matrix Interpretation

```python
cm = confusion_matrix(y_test, y_pred)

print('Confusion Matrix:')
print(f'True Negatives (TN): {cm[0][0]:,}')   # Correctly identified legitimate
print(f'False Positives (FP): {cm[0][1]:,}')  # False alarms (legitimate flagged as fraud)
print(f'False Negatives (FN): {cm[1][0]:,}')  # MISSED FRAUD (most costly!)
print(f'True Positives (TP): {cm[1][1]:,}')   # Correctly caught fraud
```

**Example Interpretation**:

```text
             Predicted
             Legit  Fraud
Actual Legit  4950    50    ← 50 false alarms (acceptable)
       Fraud     5    45    ← 5 missed frauds (need to reduce!)

Precision = 45/(45+50) = 47.4%  (of 95 flagged, 45 are real fraud)
Recall = 45/(45+5) = 90.0%      (caught 45 of 50 total frauds)
F1-Score = 2 * (0.474 * 0.90) / (0.474 + 0.90) = 62.0%
```

### Model Comparison

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Train multiple models with SMOTE pipeline
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = []
for name, model in models.items():
    # Create SMOTE pipeline
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    
    # Train and evaluate
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    results.append({
        'Model': name,
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    })

# Compare
results_df = pd.DataFrame(results).sort_values('F1-Score', ascending=False)
print(results_df)
```

**Typical Results**:

```text
Model                Precision  Recall  F1-Score  ROC-AUC
Gradient Boosting       0.65     0.92     0.76     0.94
Random Forest           0.62     0.88     0.73     0.92
Logistic Regression     0.58     0.85     0.69     0.88
```

---

## Hyperparameter Tuning

### Tuning Strategy for Gradient Boosting

**Key Parameters**:

| Parameter | Purpose | Good Range | Impact |
| ----------- | --------- | ------------ | -------- |
| `n_estimators` | Number of boosting stages | [50, 100, 150, 200, 300] | More = better, but diminishing returns |
| `learning_rate` | Shrinkage parameter | [0.01, 0.05, 0.1, 0.15, 0.2] | Lower = slower learning, needs more trees |
| `max_depth` | Tree depth | [3, 4, 5, 6, 7, 8] | Deeper = more complex, risk overfitting |
| `min_samples_split` | Min samples to split | [2, 5, 10, 15, 20] | Higher = more regularization |
| `min_samples_leaf` | Min samples in leaf | [1, 2, 4, 6, 8] | Higher = smoother predictions |
| `subsample` | Fraction of samples | [0.6, 0.7, 0.8, 0.9, 1.0] | <1.0 adds stochasticity (regularization) |
| `max_features` | Features per split | ['sqrt', 'log2', None, 0.5, 0.7] | Adds diversity, prevents overfitting |

### RandomizedSearchCV with SMOTE Pipeline

```python
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer

# Define search space
param_distributions = {
    'classifier__n_estimators': [50, 100, 150, 200, 300],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'classifier__max_depth': [3, 4, 5, 6, 7, 8],
    'classifier__min_samples_split': [2, 5, 10, 15, 20],
    'classifier__min_samples_leaf': [1, 2, 4, 6, 8],
    'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'classifier__max_features': ['sqrt', 'log2', None, 0.5, 0.7]
}

# Total combinations: 5 × 5 × 6 × 5 × 5 × 5 × 5 = 46,875 (too many!)
# RandomizedSearch will sample 50 random combinations

# Create SMOTE pipeline
smote_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Setup RandomizedSearchCV
f1_scorer = make_scorer(f1_score)

random_search = RandomizedSearchCV(
    estimator=smote_pipeline,
    param_distributions=param_distributions,
    n_iter=50,  # Test 50 random combinations
    scoring=f1_scorer,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    verbose=2,
    random_state=42,
    n_jobs=-1,  # Use all CPU cores
    return_train_score=True
)

# Fit on UNBALANCED data (SMOTE applied inside each fold)
print('🚀 Starting hyperparameter search (this may take 10-20 minutes)...')
random_search.fit(X_train, y_train)

print(f'\n🏆 Best F1-Score (CV): {random_search.best_score_:.4f}')
print(f'\n📋 Best Parameters:')
for param, value in random_search.best_params_.items():
    clean_param = param.replace('classifier__', '')
    print(f'   {clean_param}: {value}')

# Best model
best_model = random_search.best_estimator_
```

### Analyze Hyperparameter Impact

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create results DataFrame
results_df = pd.DataFrame(random_search.cv_results_)
results_df = results_df.sort_values('mean_test_score', ascending=False)

# Plot parameter impact
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Hyperparameter Impact on F1-Score', fontsize=16)

params_to_plot = [
    'param_classifier__n_estimators',
    'param_classifier__learning_rate',
    'param_classifier__max_depth',
    'param_classifier__min_samples_split',
    'param_classifier__subsample',
    'param_classifier__max_features'
]

for idx, param in enumerate(params_to_plot):
    ax = axes[idx // 3, idx % 3]
    param_impact = results_df.groupby(param)['mean_test_score'].mean().sort_index()
    
    if param_impact.index.dtype == 'object':
        param_impact.plot(kind='bar', ax=ax, color='steelblue')
    else:
        ax.plot(param_impact.index, param_impact.values, 'o-', linewidth=2, markersize=8)
    
    ax.set_xlabel(param.split('__')[1].replace('_', ' ').title())
    ax.set_ylabel('Mean F1-Score')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Evaluate Tuned Model

```python
# Predictions with best model
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print('='*70)
print('📊 TUNED MODEL PERFORMANCE')
print('='*70)
print(f'\nPrecision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'ROC-AUC: {roc_auc:.4f}')
```

---

## Performance Expectations

### Typical Results Journey

**Baseline (No Tuning)**:

```text
Model: Gradient Boosting (default parameters)
Precision: 0.5500
Recall: 0.8800
F1-Score: 0.6800
ROC-AUC: 0.8800

Interpretation: Decent baseline, but room for improvement
```

**After SMOTE Pipeline (Proper Implementation)**:

```text
Model: Gradient Boosting + SMOTE (inside CV)
Precision: 0.6200
Recall: 0.9100
F1-Score: 0.7400
ROC-AUC: 0.9200

Improvement: +6pp F1-Score by properly handling imbalance
```

**After Hyperparameter Tuning**:

```text
Model: Gradient Boosting + SMOTE + Tuned Hyperparameters
Precision: 0.6500
Recall: 0.9200
F1-Score: 0.7600
ROC-AUC: 0.9400

Improvement: +2pp F1-Score through optimization
Total Improvement: +8pp F1-Score from baseline
```

### What Good Performance Looks Like

**For 0.78% Fraud Rate**:

✅ **Good Performance**:

- F1-Score: 70-75%
- Precision: 60-65%
- Recall: 90-92%
- ROC-AUC: 0.92-0.94
- Few missed frauds (FN < 10%)

🎯 **Excellent Performance**:

- F1-Score: 75-80%
- Precision: 65-70%
- Recall: 92-95%
- ROC-AUC: 0.94-0.96
- Very few missed frauds (FN < 5%)

🚨 **Suspicious Performance** (likely data leakage):

- F1-Score: >95%
- Precision: >90%
- Recall: >98%
- Perfect or near-perfect confusion matrix

---

## Production Deployment

### Save Model and Preprocessing

```python
import joblib
from datetime import datetime

# Save best model
joblib.dump(best_model, '../models/fraud_detection_model_tuned.pkl')

# Save scaler
joblib.dump(scaler, '../models/scaler.pkl')

# Save encoders
joblib.dump(label_encoders, '../models/label_encoders.pkl')
joblib.dump(target_encoders, '../models/target_encoders.pkl')

# Save feature names
joblib.dump(X_train.columns.tolist(), '../models/feature_names.pkl')

# Save metadata
metadata = {
    'model_name': 'Gradient Boosting (Tuned)',
    'training_date': datetime.now().isoformat(),
    'best_params': random_search.best_params_,
    'test_metrics': {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    },
    'fraud_rate': y_train.mean(),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'features': X_train.columns.tolist()
}

joblib.dump(metadata, '../models/model_metadata_tuned.pkl')
```

### Prediction Pipeline

```python
def predict_fraud(transaction):
    """
    Predict fraud probability for a single transaction
    
    Args:
        transaction: dict with transaction features
        
    Returns:
        dict with prediction and probability
    """
    # Load model and preprocessing objects
    model = joblib.load('../models/fraud_detection_model_tuned.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    label_encoders = joblib.load('../models/label_encoders.pkl')
    target_encoders = joblib.load('../models/target_encoders.pkl')
    feature_names = joblib.load('../models/feature_names.pkl')
    
    # Convert to DataFrame
    df = pd.DataFrame([transaction])
    
    # Apply label encoding
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))
    
    # Apply target encoding
    for col, encoder_info in target_encoders.items():
        if col in df.columns:
            encoding_dict = encoder_info['encoding_dict']
            global_mean = encoder_info['global_mean']
            df[col] = df[col].map(encoding_dict).fillna(global_mean)
    
    # Ensure feature order matches training
    df = df[feature_names]
    
    # Scale features
    df_scaled = scaler.transform(df)
    
    # Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0, 1]
    
    return {
        'is_fraud': bool(prediction),
        'fraud_probability': float(probability),
        'risk_level': 'HIGH' if probability > 0.8 else 
                      'MEDIUM' if probability > 0.5 else 'LOW',
        'recommendation': 'BLOCK' if probability > 0.8 else
                         'MANUAL_REVIEW' if probability > 0.5 else 'ALLOW'
    }

# Example usage
transaction = {
    'amount': 150.50,
    'payment_method': 'credit_card',
    'merchant_id': 'M12345',
    'hour_of_day': 14,
    'is_weekend': 0,
    # ... other features
}

result = predict_fraud(transaction)
print(result)
# {'is_fraud': False, 'fraud_probability': 0.23, 'risk_level': 'LOW', 'recommendation': 'ALLOW'}
```

### Monitoring in Production

```python
class FraudModelMonitor:
    """
    Monitor fraud detection model performance in production
    """
    def __init__(self, window_size=1000):
        self.predictions = []
        self.actuals = []
        self.probabilities = []
        self.timestamps = []
        self.window_size = window_size
    
    def log_prediction(self, prediction, probability, actual=None):
        self.predictions.append(prediction)
        self.probabilities.append(probability)
        self.timestamps.append(datetime.now())
        
        if actual is not None:
            self.actuals.append(actual)
        
        # Keep only recent predictions
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
            self.probabilities.pop(0)
            self.timestamps.pop(0)
            if self.actuals:
                self.actuals.pop(0)
    
    def get_metrics(self):
        if len(self.actuals) == 0:
            return None
        
        precision = precision_score(self.actuals, self.predictions)
        recall = recall_score(self.actuals, self.predictions)
        f1 = f1_score(self.actuals, self.predictions)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fraud_rate': np.mean(self.actuals),
            'avg_fraud_prob': np.mean([p for p, a in zip(self.probabilities, self.actuals) if a == 1]),
            'avg_legit_prob': np.mean([p for p, a in zip(self.probabilities, self.actuals) if a == 0])
        }
    
    def check_drift(self, baseline_f1=0.76, threshold=0.05):
        """
        Check for significant performance degradation
        """
        metrics = self.get_metrics()
        if metrics is None:
            return False, "Not enough data"
        
        f1_drop = baseline_f1 - metrics['f1']
        if f1_drop > threshold:
            return True, f"F1 dropped by {f1_drop:.2%} (current: {metrics['f1']:.4f})"
        
        return False, "Performance stable"

# Usage
monitor = FraudModelMonitor()

# In production
for transaction in streaming_transactions:
    result = predict_fraud(transaction)
    monitor.log_prediction(
        result['is_fraud'],
        result['fraud_probability']
    )
    
    # When we get actual fraud label (after investigation)
    if transaction_investigated:
        monitor.log_prediction(
            result['is_fraud'],
            result['fraud_probability'],
            actual=transaction.actual_fraud_label
        )
    
    # Check every 100 transactions
    if len(monitor.actuals) % 100 == 0:
        metrics = monitor.get_metrics()
        print(f'Current F1: {metrics["f1"]:.4f}')
        
        drift_detected, message = monitor.check_drift()
        if drift_detected:
            print(f'⚠️ ALERT: {message}')
            # Trigger retraining pipeline
```

---

## Key Learnings

### Critical Success Factors

1. ✅ **Proper SMOTE Implementation**
   - Apply SMOTE inside CV pipeline, not before
   - Prevents data leakage that inflates metrics

2. ✅ **Target Encoding for High Cardinality**
   - Calculate statistics on training data only
   - Apply smoothing to prevent overfitting
   - Handle unseen categories gracefully

3. ✅ **Right Metrics for Imbalanced Data**
   - Don't use accuracy!
   - Focus on F1-Score, Precision, Recall, ROC-AUC
   - Understand business cost of false negatives vs false positives

4. ✅ **Realistic Expectations**
   - F1-Score of 70-75% is good for 0.78% fraud rate
   - >95% F1 is likely data leakage
   - Missing some fraud is inevitable

5. ✅ **Feature Engineering Matters**
   - Temporal features (hour, day, weekend)
   - User behavior (deviation from typical)
   - Merchant statistics (fraud rate)
   - Amount patterns (rounded, percentiles)

### Common Mistakes to Avoid

1. ❌ Applying SMOTE before cross-validation
2. ❌ Using accuracy as primary metric
3. ❌ Calculating aggregations on full dataset (includes test data)
4. ❌ Not using stratified splits
5. ❌ Including outcome-related features (data leakage)
6. ❌ Expecting >95% F1-Score on real fraud data
7. ❌ Not monitoring model performance over time

### Next Steps for Further Improvement

1. **Try Advanced Algorithms**
   - XGBoost or LightGBM (often better than sklearn's GradientBoosting)
   - CatBoost (handles categorical features natively)
   - Neural networks for complex patterns

2. **Ensemble Methods**
   - Stacking multiple models
   - Voting classifiers
   - Blend predictions from different approaches

3. **Cost-Sensitive Learning**
   - Assign different costs to FP vs FN
   - Optimize for business metrics, not just F1

4. **More Feature Engineering**
   - Interaction features (amount × hour)
   - Time-series features (velocity, frequency)
   - Graph features (user-merchant network)

5. **Threshold Optimization**
   - Find optimal threshold for precision/recall trade-off
   - Different thresholds for different risk levels
   - A/B test threshold changes

---

## Summary

**Fraud Detection Pipeline**:

```text
Raw Data → Feature Engineering → Encoding → Train-Test Split (Stratified) →
Scale Features → SMOTE (inside CV!) → Model Training → Hyperparameter Tuning →
Evaluation → Deployment → Monitoring
```

**Key Metrics Achieved**:

- Baseline: F1 = 68% (no tuning, proper SMOTE)
- After Tuning: F1 = 76% (optimized hyperparameters)
- ROC-AUC: 0.94 (excellent discrimination)
- Recall: 92% (catching 92% of fraud cases)

**Critical Lessons**:

- ⚠️ SMOTE inside CV pipeline prevents data leakage
- 🎯 F1-Score of 70-76% is realistic and good for 0.78% fraud rate
- 📊 Right metrics matter: F1 > Accuracy for imbalanced data
- 🔧 Feature engineering and proper encoding are crucial
- 🚨 Monitor performance in production for drift

**This case study demonstrates**:

- Real-world challenges of imbalanced classification
- Proper techniques to avoid data leakage
- Setting realistic expectations
- End-to-end ML pipeline from data to production

---

*Based on notebooks 05_fraud_data_processing.ipynb, 06_fraud_model_training.ipynb, 07_hyperparameter_tuning.ipynb*  
*Last Updated: January 2, 2026*
