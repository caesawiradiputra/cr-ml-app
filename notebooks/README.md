# Notebooks Overview

This directory contains two types of notebooks: **learning exercises** and **production-ready fraud detection** models.

---

## 📚 Part 1: Learning Exercises (Iris Dataset)

These notebooks use the classic Iris dataset to learn fundamental ML concepts:

### [01_iris_classification.ipynb](01_iris_classification.ipynb)

**Purpose**: Introduction to classification

- Basic data exploration and visualization
- Train-test split
- First classification model (Decision Tree)
- Model evaluation basics

### [02_algorithm_comparison.ipynb](02_algorithm_comparison.ipynb)

**Purpose**: Compare multiple ML algorithms

- Logistic Regression, Decision Tree, Random Forest, SVM
- Performance metrics comparison
- Understanding algorithm strengths/weaknesses

### [03_noise_level_assessment.ipynb](03_noise_level_assessment.ipynb)

**Purpose**: Learn about model robustness

- Introduce artificial noise to data
- Assess how models handle noisy data
- Understand overfitting vs underfitting

### [04_hyperparameter_tuning.ipynb](04_hyperparameter_tuning.ipynb)

**Purpose**: Optimize model performance

- GridSearchCV and RandomizedSearchCV
- Cross-validation strategies
- Parameter tuning best practices

---

## 🚀 Part 2: Production Fraud Detection (Real Anti-Fraud Application)

These notebooks implement a complete fraud detection system for production use:

### [05_fraud_data_processing.ipynb](05_fraud_data_processing.ipynb)

**Purpose**: Data preprocessing and feature engineering

- Load and explore fraud dataset (126,530 transactions)
- Handle missing values and duplicates
- Create binary target (`is_fraud`)
- Handle severe class imbalance (0.78% fraud rate)
- Output: `af_dataset_processed.csv`

### [06_fraud_model_training.ipynb](06_fraud_model_training.ipynb) ⭐

**Purpose**: Train and evaluate fraud detection models

- Target encoding for high-cardinality features
- SMOTE for class imbalance
- Train 3 models: Logistic Regression, Random Forest, Gradient Boosting
- Comprehensive evaluation (Precision, Recall, F1-Score, ROC-AUC)
- **Best Model**: Gradient Boosting (F1=69.16%, Recall=93.43%)
- Save best model and preprocessing objects
- Output: Trained model + preprocessing artifacts in `../models/`

### [07_hyperparameter_tuning.ipynb](07_hyperparameter_tuning.ipynb)

**Purpose**: Optimize Gradient Boosting model

- RandomizedSearchCV with 50 parameter combinations
- Optimize for F1-Score (best metric for imbalanced data)
- 5-fold stratified cross-validation
- Compare tuned model vs baseline
- Save optimized model for deployment

---

## 📂 Model Artifacts (from Notebook 06 & 07)

When you run the training notebooks, several files are saved to `../models/`:

### 1. `fraud_detection_model.pkl`

**What**: The trained Gradient Boosting classifier
**Purpose**: The core model that predicts fraud (0=Legitimate, 1=Fraud)
**Usage**: Load with `joblib.load()` to make predictions on new transactions

### 2. `scaler.pkl`

**What**: StandardScaler fitted on training data
**Purpose**: Normalizes features to mean=0, std=1
**Usage**: **Must** apply to new data before prediction to match training distribution

```python
scaled_features = scaler.transform(new_data)
```

### 3. `label_encoders.pkl`

**What**: Dictionary of LabelEncoder objects for low-cardinality features
**Purpose**: Converts categorical text to numbers (e.g., "Yes"→1, "No"→0)
**Usage**: Apply to categorical columns before scaling

```python
for col, encoder in label_encoders.items():
    new_data[col] = encoder.transform(new_data[col])
```

### 4. `target_encoders.pkl`

**What**: Dictionary with encoding maps for high-cardinality features
**Purpose**: Replaces each category with its fraud rate (e.g., city "Jakarta" → 0.012)
**Usage**: Apply to high-cardinality columns (cities, agent IDs, subdistricts)

```python
for col, encoder_info in target_encoders.items():
    new_data[col] = new_data[col].map(encoder_info['encoding_dict'])
    new_data[col].fillna(encoder_info['global_mean'], inplace=True)
```

### 5. `feature_names.pkl`

**What**: List of feature column names in exact training order
**Purpose**: Ensures new data has features in correct order
**Usage**: Reorder columns before prediction

```python
new_data = new_data[feature_names]
```

### 6. `model_metadata.pkl`

**What**: Dictionary with model information
**Purpose**: Documentation and reproducibility
**Contains**:

- `model_name`: "Gradient Boosting"
- `training_date`: When model was trained
- `metrics`: Precision, Recall, F1-Score, ROC-AUC
- `training_samples`: 101,224
- `test_samples`: 25,306
- `features`: List of all feature names
- `smote_used`: True (indicates SMOTE was applied)

---

## 🔄 Production Deployment Workflow

To use the trained model in production:

```python
import joblib
import pandas as pd

# 1. Load all artifacts
model = joblib.load('models/fraud_detection_model.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')
target_encoders = joblib.load('models/target_encoders.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# 2. Prepare new transaction data
new_transaction = pd.DataFrame({...})  # Your new data

# 3. Apply same preprocessing pipeline
# a) Label encoding for low-cardinality features
for col, encoder in label_encoders.items():
    new_transaction[col] = encoder.transform(new_transaction[col])

# b) Target encoding for high-cardinality features
for col, encoder_info in target_encoders.items():
    new_transaction[col] = new_transaction[col].map(encoder_info['encoding_dict'])
    new_transaction[col].fillna(encoder_info['global_mean'], inplace=True)

# c) Ensure correct feature order
new_transaction = new_transaction[feature_names]

# d) Scale features
new_transaction_scaled = scaler.transform(new_transaction)

# 4. Make prediction
prediction = model.predict(new_transaction_scaled)[0]
fraud_probability = model.predict_proba(new_transaction_scaled)[0, 1]

# 5. Interpret result
if prediction == 1:
    print(f"⚠️ FRAUD DETECTED (confidence: {fraud_probability*100:.1f}%)")
else:
    print(f"✅ Legitimate (fraud probability: {fraud_probability*100:.1f}%)")
```

---

## 📊 Key Performance Metrics

### Baseline Model (Gradient Boosting - Notebook 06)

- **Precision**: 54.90% (55% of flagged transactions are real fraud)
- **Recall**: 93.43% (catches 93% of all fraud cases)
- **F1-Score**: 69.16% (best balance for fraud detection)
- **ROC-AUC**: 99.64% (excellent discrimination ability)
- **False Positives**: 152 (out of 25,108 legitimate transactions)
- **False Negatives**: 13 (only 13 frauds missed out of 198)

### Why These Metrics Matter

- **High Recall (93.43%)** is critical: We catch almost all fraud, minimizing financial loss
- **Moderate Precision (54.90%)** is acceptable: 152 false alarms vs 185 real fraud detections
- **F1-Score (69.16%)** shows good balance between precision and recall
- **Low False Negatives (13)** means we miss very little fraud

---

## 🎯 Next Steps

1. **Run Hyperparameter Tuning** (Notebook 07)
   - May improve F1-Score beyond 69.16%
   - Typically takes 10-20 minutes

2. **Threshold Optimization**
   - Current threshold: 0.5 (default)
   - Adjust threshold to tune precision/recall trade-off
   - Lower threshold → Higher recall, more false alarms
   - Higher threshold → Higher precision, more missed fraud

3. **Deploy to Production**
   - Create API endpoint for real-time scoring
   - Implement monitoring and alerting
   - Set up feedback loop for continuous improvement

4. **Monitor Model Performance**
   - Track precision/recall over time
   - Watch for concept drift (fraud patterns change)
   - Retrain periodically with new data

---

## 📝 Notes

- **Class Imbalance**: Original data has 127:1 ratio (legitimate:fraud)
- **SMOTE Used**: Training data balanced to 1:1 using synthetic oversampling
- **Test Set**: Kept imbalanced to reflect real-world conditions
- **Target Encoding**: Critical for handling 10,000+ unique values in city/agent/subdistrict fields
- **Decision**: Gradient Boosting chosen over Random Forest and Logistic Regression based on F1-Score

---

## 🔗 Related Files

- **Data**: `../data/af_dataset_processed.csv` (126,530 rows, processed)
- **Models**: `../models/*.pkl` (trained models and preprocessing objects)
- **Raw Data**: `../data/af_dataset.csv` (original dataset)
