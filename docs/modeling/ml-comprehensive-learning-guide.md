# Comprehensive Machine Learning Learning Guide

**Purpose**: Complete ML workflow from basics to production  
**Audience**: Beginners to intermediate ML practitioners  
**Based on**: Iris classification notebooks (01-04) + fraud detection project experience

> 📚 **Real-World Case Study**: See [Fraud Detection Case Study](fraud-detection-case-study.md) for detailed implementation of these concepts on an imbalanced classification problem.

---

## Table of Contents

1. [ML Fundamentals](#ml-fundamentals)
2. [Data Preparation](#data-preparation)
3. [Algorithm Selection](#algorithm-selection)
4. [Data Quality Assessment](#data-quality-assessment)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Common Pitfalls & Solutions](#common-pitfalls--solutions)
7. [Production Considerations](#production-considerations)

---

## ML Fundamentals

### What is Machine Learning?

**Machine Learning** = Teaching computers to learn patterns from data without explicit programming

### Types of ML

1. **Supervised Learning**: Learn from labeled examples (classification, regression)
2. **Unsupervised Learning**: Find patterns in unlabeled data (clustering, dimensionality reduction)
3. **Reinforcement Learning**: Learn through trial and error (game playing, robotics)

### Classification Workflow

```text
1. Load Data → 2. Explore → 3. Prepare → 4. Train → 5. Evaluate → 6. Deploy
     ↓             ↓            ↓          ↓          ↓            ↓
  CSV file    Visualize    Split/Scale  Choose     Metrics    Production
             Statistics    Features     Algorithm  CV/Test    API/Service
```

### Key Concepts

#### Train-Test Split

**Purpose**: Evaluate model on unseen data

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # Reproducibility
    stratify=y          # Maintain class distribution
)
```

**Why stratify?**

- Ensures test set has same class balance as full dataset
- Critical for imbalanced data (see [Fraud Detection Case Study](fraud-detection-case-study.md) for 0.78% fraud rate example)
- Prevents lucky/unlucky splits

#### Feature Scaling

**When needed**: Distance-based algorithms (KNN, SVM, Neural Networks)  
**Not needed**: Tree-based algorithms (Decision Tree, Random Forest, Gradient Boosting)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Learn mean/std from training
X_test_scaled = scaler.transform(X_test)        # Apply same transformation
```

**⚠️ CRITICAL**: Always fit scaler on training data only! Fitting on test data causes data leakage.

#### Cross-Validation

**Purpose**: More robust performance estimate than single train-test split

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')

print(f'Mean: {scores.mean():.4f}')
print(f'Std: {scores.std():.4f}')
```

**K-Fold Process**:

```text
Fold 1: [Train][Train][Train][Train][Test] → Score: 0.95
Fold 2: [Train][Train][Train][Test][Train] → Score: 0.93
Fold 3: [Train][Train][Test][Train][Train] → Score: 0.97
Fold 4: [Train][Test][Train][Train][Train] → Score: 0.94
Fold 5: [Test][Train][Train][Train][Train] → Score: 0.96
         ↓
Average: 0.950 ± 0.015
```

**Why use CV?**

- Single split can be lucky/unlucky
- CV averages across multiple splits
- More reliable performance estimate
- Helps detect overfitting

---

## Data Preparation

### Handling Missing Values

```python
# Check for missing values
print(df.isnull().sum())

# Strategies:
# 1. Drop rows (if few missing)
df_clean = df.dropna()

# 2. Impute with mean/median
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# 3. Impute with mode (categorical)
imputer = SimpleImputer(strategy='most_frequent')
```

### Encoding Categorical Features

#### Label Encoding (Low Cardinality)

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])
```

**When to use**: Ordinal data (small, medium, large) or low cardinality (<10 categories)

#### One-Hot Encoding (Nominal Categories)

```python
df_encoded = pd.get_dummies(df, columns=['color', 'size'])
```

**When to use**: Nominal categories with no inherent order

#### Target Encoding (High Cardinality)

```python
# For fraud detection with high-cardinality categorical features
# Calculate fraud rate for each category
global_mean = y_train.mean()
category_stats = X_train.groupby('merchant_id')['fraud'].agg(['mean', 'count'])

# Smooth with global mean (regularization)
min_samples = 10
category_stats['smoothed_mean'] = (
    (category_stats['mean'] * category_stats['count'] + global_mean * min_samples) /
    (category_stats['count'] + min_samples)
)

# Apply encoding
X_train['merchant_id_encoded'] = X_train['merchant_id'].map(
    category_stats['smoothed_mean']
).fillna(global_mean)
```

**⚠️ CRITICAL**: Only calculate statistics on training data, then apply to test data!

### Feature Engineering

```python
# 1. Interaction features
df['feature1_x_feature2'] = df['feature1'] * df['feature2']

# 2. Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# 3. Domain-specific features (fraud example)
df['hour_of_day'] = pd.to_datetime(df['created_at']).dt.hour
df['is_weekend'] = pd.to_datetime(df['created_at']).dt.dayofweek >= 5
df['amount_deviation'] = df['amount'] - df.groupby('user_id')['amount'].transform('mean')
```

---

## Algorithm Selection

### Algorithm Comparison Matrix

| Algorithm | Pros | Cons | Best For |
| ----------- | ------ | ------ | ---------- |
| **KNN** | Simple, no training time, intuitive | Slow predictions, curse of dimensionality | Small datasets, low dimensions |
| **Decision Tree** | Interpretable, fast, handles non-linear | Overfits easily, unstable | Quick baseline, interpretability |
| **Random Forest** | Robust, handles overfitting, feature importance | Slower, less interpretable | General purpose, high accuracy |
| **SVM** | Excellent for high-dim, flexible kernels | Slow training, hard to tune | Text, images, clear margins |
| **Gradient Boosting** | Highest accuracy, handles imbalance | Slow training, risk of overfitting | Competitions, production (when tuned) |

### Algorithm Selection Guide

```text
Start Here: Random Forest (robust, good out-of-box performance)
    ↓
High accuracy needed? → Try Gradient Boosting (XGBoost, LightGBM)
    ↓
Need interpretability? → Decision Tree or Logistic Regression
    ↓
High dimensions (>1000 features)? → SVM with linear kernel
    ↓
Small dataset (<1000 samples)? → KNN or SVM
    ↓
Real-time predictions critical? → Logistic Regression or simple tree
```

### Model Functions

```python
# 1. fit() - Train the model
model.fit(X_train, y_train)

# 2. predict() - Get predictions
y_pred = model.predict(X_test)  # Returns: [0, 1, 2, 1, ...]

# 3. predict_proba() - Get probabilities
y_proba = model.predict_proba(X_test)  # Returns: [[0.1, 0.8, 0.1], ...]

# 4. score() - Get accuracy (predict + compare)
accuracy = model.score(X_test, y_test)  # Returns: 0.96
```

### Evaluation Metrics

#### For Balanced Data

```python
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

#### For Imbalanced Data (e.g., Fraud Detection)

```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)

precision = precision_score(y_test, y_pred)  # True positives / Predicted positives
recall = recall_score(y_test, y_pred)        # True positives / Actual positives
f1 = f1_score(y_test, y_pred)                # Harmonic mean of precision/recall
roc_auc = roc_auc_score(y_test, y_proba)    # Area under ROC curve

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f'True Negatives: {cm[0][0]}')
print(f'False Positives: {cm[0][1]}')  # Type I Error
print(f'False Negatives: {cm[1][0]}')  # Type II Error
print(f'True Positives: {cm[1][1]}')
```

**Metric Selection Guide**:

- **Accuracy**: Good for balanced classes only
- **Precision**: When false positives are costly (spam detection)
- **Recall**: When false negatives are costly (medical diagnosis, fraud)
- **F1-Score**: Balance between precision and recall (imbalanced data)
- **ROC-AUC**: Overall performance across all thresholds

> 📚 **Deep Dive**: See [Fraud Detection Case Study](fraud-detection-case-study.md) for detailed metrics analysis on a highly imbalanced dataset.

---

## Data Quality Assessment

### Noise Indicators

#### 1. Cross-Validation Variability

```python
cv_scores = cross_val_score(model, X_train, y_train, cv=10)
std = cv_scores.std()

# Interpretation:
# std < 0.02 → Low noise ✅
# std 0.02-0.05 → Medium noise ⚠️
# std > 0.05 → High noise 🔴
```

#### 2. Train-Test Gap

```python
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
gap = train_score - test_score

# Interpretation:
# gap < 5% → Good generalization ✅
# gap 5-10% → Acceptable ⚠️
# gap > 10% → Overfitting 🔴
```

#### 3. Model Agreement

```python
# Train multiple models
models = [knn, decision_tree, random_forest, svm]
predictions = [model.predict(X_test) for model in models]

# Check agreement
from scipy.stats import mode
consensus = mode(predictions, axis=0)[0]
disagreements = (predictions != consensus).sum(axis=0)
disagreement_rate = (disagreements > 0).mean()

# Interpretation:
# rate < 5% → Low noise ✅
# rate 5-15% → Medium noise ⚠️
# rate > 15% → High noise 🔴
```

#### 4. Outlier Detection

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20)
outliers = lof.fit_predict(X_train_scaled)
outlier_ratio = (outliers == -1).mean()

# Interpretation:
# ratio < 5% → Low noise ✅
# ratio 5-15% → Medium noise ⚠️
# ratio > 15% → High noise 🔴
```

#### 5. Feature-Target Correlation

```python
correlations = df.corr()['target'].abs().sort_values(ascending=False)
max_correlation = correlations.iloc[1]  # Skip target-target correlation

# Interpretation:
# max_corr > 0.7 → Strong signal ✅
# max_corr 0.4-0.7 → Medium signal ⚠️
# max_corr < 0.4 → Weak signal 🔴
```

#### 6. Visual Separation

```python
import seaborn as sns

# Pairplot colored by class
sns.pairplot(df, hue='target', diag_kind='kde')

# Look for:
# - Clear clusters → Low noise ✅
# - Some overlap → Medium noise ⚠️
# - Heavy mixing → High noise 🔴
```

#### 7. Learning Curves

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# Plot
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')

# Interpretation:
# Validation plateaus at > 90% → Low noise ✅
# Validation plateaus at 75-90% → Medium noise ⚠️
# Validation plateaus at < 75% → High noise 🔴
```

### Setting Realistic Expectations

**For fraud detection with 0.78% fraud rate:**

- **F1-Score of 65-75%** is realistic and good
- **F1-Score > 80%** is excellent
- **F1-Score of 99%** is likely data leakage!

**Why?**

- Extreme class imbalance (1:127 ratio)
- Fraudulent and legitimate transactions often similar
- Fraudsters adapt their behavior (adversarial)
- Missing context features (device, location, history)

---

## Hyperparameter Tuning

### What are Hyperparameters?

**Hyperparameters** = Settings you choose BEFORE training that control HOW the model learns

| Aspect | Parameters | Hyperparameters |
| -------- | ------------ | ----------------- |
| **Set When** | During training (learned) | Before training (chosen) |
| **Example** | Tree split values, weights | max_depth, learning_rate |
| **Optimized Via** | Loss minimization | Cross-validation |

### Key Hyperparameters by Algorithm

#### K-Nearest Neighbors

```python
# n_neighbors: Number of neighbors to consider
# Too small (1-3) → overfitting
# Too large (50+) → underfitting
# Good range: [3, 5, 7, 9, 11, 13, 15]

# weights: Voting strategy
# 'uniform' → All neighbors equal
# 'distance' → Closer neighbors matter more (usually better)

# metric: Distance calculation
# 'euclidean' → Straight-line distance
# 'manhattan' → Grid-like distance (better for high dimensions)
```

#### Decision Tree

```python
# max_depth: Maximum tree levels
# None → Unlimited (overfits!)
# 3-5 → Simple model (may underfit)
# 7-15 → Usually optimal

# min_samples_split: Minimum samples to split node
# 2 → More splits (overfitting risk)
# 10-20 → Fewer splits (regularization)

# min_samples_leaf: Minimum samples in leaf
# 1 → Can create single-sample leaves (overfit!)
# 5-10 → Smoother predictions

# criterion: Split quality measure
# 'gini' → Faster (default)
# 'entropy' → More precise (minimal difference)
```

#### Random Forest

```python
# n_estimators: Number of trees
# 10-50 → Fast but unstable
# 100 → Good balance (default)
# 200-500 → More stable, diminishing returns

# max_depth: Depth of each tree
# None → Unlimited (Random Forest handles overfitting)
# 5-15 → Prevents individual tree overfitting

# max_features: Features per split
# 'sqrt' → sqrt(n_features) - recommended for classification
# 'log2' → log2(n_features) - also good
# None → All features (loses diversity!)

# min_samples_split, min_samples_leaf: Same as Decision Tree
```

#### Gradient Boosting

```python
# n_estimators: Number of boosting stages
# 50-100 → Quick baseline
# 100-200 → Good performance
# 200+ → Diminishing returns, overfitting risk

# learning_rate: Shrinkage parameter
# 0.01 → Slow learning (need more trees)
# 0.1 → Balanced (default)
# 0.2+ → Aggressive (overfitting risk)

# max_depth: Tree depth
# 3-5 → Shallow trees (usually best for boosting)
# 6-8 → Deeper (more complex)

# subsample: Fraction of samples per tree
# 0.6-0.8 → Stochastic gradient boosting (regularization)
# 1.0 → Use all samples

# min_samples_split, min_samples_leaf: Regularization
```

### GridSearchCV - Exhaustive Search

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 7, 8],
}

# Total combinations: 4 × 4 × 6 = 96

grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,                    # 5-fold cross-validation
    scoring='f1',            # Optimize for F1-score
    n_jobs=-1,               # Use all CPU cores
    verbose=2
)

grid_search.fit(X_train, y_train)

print(f'Best params: {grid_search.best_params_}')
print(f'Best CV score: {grid_search.best_score_:.4f}')
```

**Total model trainings**: 96 combinations × 5 folds = **480 models**!

### RandomizedSearchCV - Faster Alternative

```python
from sklearn.model_selection import RandomizedSearchCV

# Same parameter grid
param_distributions = {
    'n_estimators': [50, 100, 150, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
}

# Total combinations: 5 × 5 × 6 × 5 × 5 × 5 = 18,750 (too many!)

random_search = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,               # Test only 50 random combinations
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42,
    verbose=2
)

random_search.fit(X_train, y_train)
```

**Total model trainings**: 50 combinations × 5 folds = **250 models** (much faster!)

### When to Use Which?

**GridSearchCV**:

- ✅ Small grid (< 100 combinations)
- ✅ Need guaranteed best in grid
- ✅ Final fine-tuning
- ❌ Large grids (too slow)

**RandomizedSearchCV**:

- ✅ Large grid (> 100 combinations)
- ✅ Exploring wide space
- ✅ Time/compute limited
- ✅ Initial coarse search
- ❌ When you need guaranteed optimum

### Tuning Strategy

```text
Stage 1: Coarse RandomizedSearch (wide range, n_iter=30-50)
   ↓
Identify promising region
   ↓
Stage 2: Fine GridSearch (narrow range around best)
   ↓
Final best parameters
```

---

## Common Pitfalls & Solutions

### 1. 🚨 Data Leakage - SMOTE Before Cross-Validation

**Problem**: Applying SMOTE to entire training set before CV

```python
# ❌ WRONG - DATA LEAKAGE!
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# CV on balanced data → LEAKAGE!
cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5)
# Result: Artificially inflated scores (e.g., 99.99% F1)
```

**Why it's wrong**:

- SMOTE creates synthetic samples by interpolating between existing samples
- When you balance FIRST, then split for CV, validation folds contain synthetic samples that are "neighbors" of training samples
- Model sees leaked information → unrealistic performance

**Solution**: Apply SMOTE inside each CV fold using Pipeline

```python
# ✅ CORRECT - NO LEAKAGE
from imblearn.pipeline import Pipeline as ImbPipeline

smote_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# SMOTE applied separately in each fold
cv_scores = cross_val_score(smote_pipeline, X_train, y_train, cv=5)
# Result: Realistic scores (e.g., 70% F1)
```

**For GridSearchCV/RandomizedSearchCV with Pipeline**:

```python
# Parameter names need 'classifier__' prefix
param_distributions = {
    'classifier__n_estimators': [50, 100, 150, 200],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__max_depth': [3, 4, 5, 6, 7, 8],
}

random_search = RandomizedSearchCV(
    estimator=smote_pipeline,
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring='f1'
)

# Fit on UNBALANCED data (SMOTE applied inside each fold)
random_search.fit(X_train, y_train)
```

### 2. Fitting Scaler on Full Dataset

**Problem**:

```python
# ❌ WRONG - DATA LEAKAGE!
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)
X_train, X_test = train_test_split(X_all_scaled, ...)
```

**Solution**:

```python
# ✅ CORRECT
X_train, X_test = train_test_split(X_all, ...)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Learn from training only
X_test_scaled = scaler.transform(X_test)          # Apply learned transformation
```

### 3. Using Test Set for Hyperparameter Tuning

**Problem**: Tuning on test set causes overfitting to test data

**Solution**: Use nested cross-validation or separate validation set

```python
# ✅ CORRECT: Three-way split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# Tune on train, validate on val, final eval on test
```

### 4. Not Using Stratified Splits for Imbalanced Data

**Problem**:

```python
# ❌ WRONG for imbalanced data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

**Solution**:

```python
# ✅ CORRECT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)
```

### 5. Ignoring Class Imbalance

**Problem**: Using accuracy for imbalanced data

```text
99% legitimate, 1% fraud
Model predicts everything as legitimate → 99% accuracy!
But catches 0% fraud (useless!)
```

**Solution**: Use appropriate metrics and sampling

```python
# Use F1-Score, Precision, Recall, ROC-AUC
from sklearn.metrics import f1_score

# Handle imbalance:
# Option 1: SMOTE (inside CV!)
from imblearn.over_sampling import SMOTE

# Option 2: Class weights
model = GradientBoostingClassifier(
    class_weight='balanced'  # Automatically adjust weights
)

# Option 3: Stratified sampling
cv = StratifiedKFold(n_splits=5, shuffle=True)
```

### 6. Not Checking for Overfitting

**Problem**: Only looking at test accuracy

**Solution**: Always compare train vs test vs CV

```python
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
cv_scores = cross_val_score(model, X_train, y_train, cv=5)

print(f'Train: {train_score:.4f}')
print(f'Test: {test_score:.4f}')
print(f'CV Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

# If train >> test → overfitting
# If CV mean << test → lucky test split
```

---

## Production Considerations

### Model Deployment Pipeline

```python
import joblib

# 1. Save trained model
joblib.dump(model, 'fraud_detection_model.pkl')

# 2. Save preprocessing objects
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

# 3. Save feature names
joblib.dump(feature_names, 'feature_names.pkl')

# 4. Save metadata
metadata = {
    'model_name': 'Gradient Boosting',
    'training_date': datetime.now().isoformat(),
    'best_params': best_params,
    'cv_f1_score': cv_f1,
    'test_metrics': {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    },
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'features': feature_names,
}
joblib.dump(metadata, 'model_metadata.pkl')
```

### Prediction Pipeline

```python
# Load all components
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
feature_names = joblib.load('feature_names.pkl')

def predict_fraud(transaction):
    """
    Predict fraud probability for a transaction
    
    Args:
        transaction: dict with transaction features
    
    Returns:
        dict with prediction and probability
    """
    # 1. Convert to DataFrame
    df = pd.DataFrame([transaction])
    
    # 2. Apply label encoding
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))
    
    # 3. Ensure feature order matches training
    df = df[feature_names]
    
    # 4. Scale features
    df_scaled = scaler.transform(df)
    
    # 5. Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0, 1]
    
    return {
        'is_fraud': bool(prediction),
        'fraud_probability': float(probability),
        'risk_level': 'HIGH' if probability > 0.8 else 
                      'MEDIUM' if probability > 0.5 else 'LOW'
    }

# Example usage
transaction = {
    'amount': 150.50,
    'merchant_id': 'M12345',
    'payment_method': 'credit_card',
    'hour_of_day': 14,
    'is_weekend': False,
    # ... other features
}

result = predict_fraud(transaction)
print(result)
# {'is_fraud': False, 'fraud_probability': 0.23, 'risk_level': 'LOW'}
```

### Monitoring Model Performance

```python
# Track metrics over time
from collections import deque
import numpy as np

class ModelMonitor:
    def __init__(self, window_size=1000):
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        self.probabilities = deque(maxlen=window_size)
    
    def log_prediction(self, y_pred, y_prob, y_true=None):
        self.predictions.append(y_pred)
        self.probabilities.append(y_prob)
        if y_true is not None:
            self.actuals.append(y_true)
    
    def get_metrics(self):
        if len(self.actuals) == 0:
            return None
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        return {
            'precision': precision_score(self.actuals, self.predictions),
            'recall': recall_score(self.actuals, self.predictions),
            'f1': f1_score(self.actuals, self.predictions),
            'avg_fraud_prob': np.mean([p for p, a in zip(self.probabilities, self.actuals) if a == 1]),
            'avg_legit_prob': np.mean([p for p, a in zip(self.probabilities, self.actuals) if a == 0]),
        }
    
    def check_drift(self, threshold=0.05):
        """Check for significant performance degradation"""
        current_metrics = self.get_metrics()
        # Compare with baseline metrics
        # Alert if F1 drops below threshold
        return current_metrics

# Usage
monitor = ModelMonitor()
for transaction, label in streaming_data:
    prediction = predict_fraud(transaction)
    monitor.log_prediction(
        prediction['is_fraud'], 
        prediction['fraud_probability'], 
        label
    )
    
    # Check every 100 predictions
    if len(monitor.actuals) % 100 == 0:
        metrics = monitor.get_metrics()
        print(f'Current F1: {metrics["f1"]:.4f}')
        if metrics['f1'] < 0.60:  # Alert threshold
            print('⚠️ Performance degradation detected!')
```

---

## Quick Reference Checklist

### ✅ Before Training

- [ ] Load and explore data (`df.head()`, `df.info()`, `df.describe()`)
- [ ] Check for missing values (`df.isnull().sum()`)
- [ ] Visualize distributions and relationships (`sns.pairplot()`)
- [ ] Split data with stratification (`train_test_split(..., stratify=y)`)
- [ ] Scale features (distance-based algorithms only)
- [ ] Encode categorical variables
- [ ] Create derived features (domain knowledge)

### ✅ During Training

- [ ] Use cross-validation for evaluation
- [ ] Check train vs test performance (detect overfitting)
- [ ] Use appropriate metric (F1 for imbalanced data)
- [ ] Apply SMOTE inside CV pipeline (avoid data leakage)
- [ ] Tune hyperparameters with GridSearch or RandomizedSearch
- [ ] Compare multiple algorithms

### ✅ After Training

- [ ] Evaluate on held-out test set
- [ ] Visualize confusion matrix
- [ ] Analyze feature importance
- [ ] Check for data leakage (unrealistic scores)
- [ ] Save model and preprocessing objects
- [ ] Document model metadata
- [ ] Set up monitoring for production

### ✅ Production Deployment

- [ ] Create prediction pipeline
- [ ] Handle edge cases and errors gracefully
- [ ] Log predictions for monitoring
- [ ] Track performance metrics over time
- [ ] Set up alerts for performance degradation
- [ ] Plan for model retraining

---

## Further Learning Resources

### Books

- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron
- "The Elements of Statistical Learning" by Hastie, Tibshirani, Friedman
- "Machine Learning Yearning" by Andrew Ng (free)

### Online Courses

- Coursera: Machine Learning by Andrew Ng
- Fast.ai: Practical Deep Learning for Coders
- Kaggle Learn: Free micro-courses

### Datasets for Practice

- Kaggle Datasets: kaggle.com/datasets
- UCI Machine Learning Repository
- sklearn.datasets (built-in datasets)

### Communities

- Kaggle Competitions and Discussions
- Reddit: r/MachineLearning, r/learnmachinelearning
- Stack Overflow: machine-learning tag

---

## Summary

**Key Takeaways**:

1. **Data Quality Matters Most**: Clean data > fancy algorithms
2. **Always Use Cross-Validation**: Single train-test split can be misleading
3. **Avoid Data Leakage**: Apply SMOTE/transformations inside CV
4. **Set Realistic Expectations**: Different problems have different performance ceilings
5. **Monitor in Production**: Performance degrades over time
6. **Document Everything**: Future you will thank present you

> 📚 **Real-World Example**: See [Fraud Detection Case Study](fraud-detection-case-study.md) for a complete implementation showing these principles in action on a highly imbalanced dataset.

**The ML Workflow**:

```text
Data → EDA → Preprocessing → Feature Engineering → Model Training → 
Evaluation → Hyperparameter Tuning → Final Evaluation → Deployment → Monitoring
     ↑                                                                    ↓
     └────────────────────── Continuous Improvement ────────────────────┘
```

**Remember**: Machine learning is iterative. Start simple, measure everything, iterate based on data, and never stop learning!

---

*This guide consolidates lessons learned from Iris classification notebooks (01-04) and production machine learning projects. For a detailed real-world case study, see [Fraud Detection Case Study](fraud-detection-case-study.md).*
*Last Updated: January 2, 2026*
