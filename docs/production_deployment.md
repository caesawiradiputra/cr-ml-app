# Production Deployment Guide

## Fraud Detection Inference System

This guide explains how to deploy and use the fraud detection models in production.

## Quick Start

### 1. Basic Usage

```bash
# Activate environment
conda activate cr-ml-app-env

# Process a CSV file with both models
python src/production_inference.py --input data/new_transactions.csv --output data/predictions.csv
```

### 2. Command-Line Options

```bash
# Use robust model only (recommended for production)
python src/production_inference.py \
    --input data/new_transactions.csv \
    --output data/predictions.csv \
    --robust-only

# Specify custom models directory
python src/production_inference.py \
    --input data/new_transactions.csv \
    --output data/predictions.csv \
    --models-dir /path/to/models

# Preview predictions without saving
python src/production_inference.py --input data/new_transactions.csv
```

## Input Requirements

### CSV Format

Your input CSV must contain the same columns as the training dataset:

**Required Columns:**
- Transaction identifiers: `payload_id`, `request_id`
- Geographic features: `payload_ktp_subdistrict`, `payload_address_subdistrict`, `payload_address_city`
- Agent features: `payload_agent_marketingid`
- Identity features: `spouse_id`, `spouse_dob`
- Vehicle features: vehicle license information
- Validation features: validation flags and counts
- All other features used in training

**Note:** The script handles unknown categories gracefully by mapping them to `<UNKNOWN>`.

## Output Format

The output CSV includes all original columns plus:

### Leaky Model Predictions (if included)
- `leaky_model_prediction`: Binary prediction (0=Legitimate, 1=Fraud)
- `leaky_model_probability`: Fraud probability (0.0 to 1.0)
- `leaky_model_fraud_flag`: Same as prediction (for clarity)

### Robust Model Predictions (always included)
- `robust_model_prediction`: Binary prediction (0=Legitimate, 1=Fraud)
- `robust_model_probability`: Fraud probability (0.0 to 1.0)
- `robust_model_fraud_flag`: Same as prediction (for clarity)

### Comparison Metrics (if both models included)
- `models_agree`: 1 if both models agree, 0 if they disagree
- `prediction_timestamp`: When predictions were generated

## Example Output

```csv
payload_id,request_id,...,leaky_model_prediction,leaky_model_probability,robust_model_prediction,robust_model_probability,models_agree,prediction_timestamp
TXN001,REQ001,...,0,0.0234,0,0.0156,1,2026-01-05 10:30:45
TXN002,REQ002,...,1,0.9823,1,0.8912,1,2026-01-05 10:30:45
TXN003,REQ003,...,1,0.7234,0,0.3456,0,2026-01-05 10:30:45
```

## Production Recommendations

### 1. Use Robust Model Only

```python
from pathlib import Path
from src.production_inference import FraudDetectionInference

# Initialize
inference = FraudDetectionInference(models_dir='models')

# Process with robust model only
df_results = inference.process_file(
    input_path='data/new_transactions.csv',
    output_path='data/predictions.csv',
    include_both_models=False  # Skip leaky model
)
```

### 2. Handle Model Disagreements

When both models are used, pay special attention to cases where they disagree:

```python
# Load predictions
df = pd.read_csv('data/predictions.csv')

# Find disagreements
disagree = df[df['models_agree'] == 0]

# Flag for manual review
high_risk_disagree = disagree[
    (disagree['robust_model_probability'] > 0.5) | 
    (disagree['leaky_model_probability'] > 0.5)
]

print(f'High-risk disagreements: {len(high_risk_disagree)}')
print('→ Manual review recommended!')
```

### 3. Batch Processing

For large datasets, process in batches:

```python
import pandas as pd

# Read in chunks
chunk_size = 10000
chunks = pd.read_csv('large_file.csv', chunksize=chunk_size)

results = []
for i, chunk in enumerate(chunks):
    print(f'Processing chunk {i+1}...')
    
    # Save chunk
    chunk.to_csv('temp_chunk.csv', index=False)
    
    # Process
    chunk_results = inference.process_file(
        input_path='temp_chunk.csv',
        include_both_models=False
    )
    
    results.append(chunk_results)

# Combine all results
final_results = pd.concat(results, ignore_index=True)
final_results.to_csv('data/all_predictions.csv', index=False)
```

### 4. Unknown Category Monitoring

The system logs unknown categories for monitoring:

```python
# Track unknown categories over time
robust_preds = inference.predict_robust(df)

if robust_preds['unknown_categories']:
    print('⚠️ New categories detected:')
    for col, values in robust_preds['unknown_categories'].items():
        print(f'   {col}: {len(values)} new values')
    
    # Log for retraining decision
    log_unknown_categories(robust_preds['unknown_categories'])
```

## Performance Expectations

### Leaky Model (Notebook 07)
- **F1-Score**: ~91.78%
- **Issue**: Memorizes rare locations
- **Risk**: Poor performance on new/unseen locations
- **Use Case**: Historical analysis only (NOT recommended for production)

### Robust Model (Notebook 08)
- **F1-Score**: ~70-80% (realistic for 0.78% fraud rate)
- **Strengths**: Generalizes to new locations, production-ready
- **Use Case**: ✅ **RECOMMENDED FOR PRODUCTION**

## Monitoring and Maintenance

### 1. Track Fraud Detection Rate

```python
# Calculate detection rate over time
df['date'] = pd.to_datetime(df['prediction_timestamp']).dt.date
daily_fraud_rate = df.groupby('date')['robust_model_fraud_flag'].mean()

# Alert if rate deviates significantly
baseline_rate = 0.0078  # Training fraud rate
current_rate = df['robust_model_fraud_flag'].mean()

if abs(current_rate - baseline_rate) > 0.01:  # 1% deviation
    print(f'⚠️ Alert: Fraud rate changed from {baseline_rate:.2%} to {current_rate:.2%}')
    print('   Consider model retraining!')
```

### 2. Monitor Model Confidence

```python
# Track average prediction confidence
avg_confidence = df['robust_model_probability'].mean()
low_confidence = df[
    (df['robust_model_probability'] > 0.3) & 
    (df['robust_model_probability'] < 0.7)
]

print(f'Avg confidence: {avg_confidence:.4f}')
print(f'Low confidence cases: {len(low_confidence)} ({len(low_confidence)/len(df)*100:.1f}%)')
```

### 3. Retraining Triggers

Consider retraining when:
- Unknown categories exceed 5% of traffic
- Fraud detection rate deviates >10% from baseline
- Model confidence drops significantly
- New fraud patterns emerge (manual review)
- Quarterly schedule (best practice)

## API Integration (Future Enhancement)

Convert to REST API using FastAPI:

```python
from fastapi import FastAPI, File, UploadFile
import pandas as pd

app = FastAPI()
inference = FraudDetectionInference()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded CSV
    df = pd.read_csv(file.file)
    
    # Generate predictions
    results = inference.predict_robust(df)
    
    return {
        'predictions': results['predictions'].tolist(),
        'probabilities': results['probabilities'].tolist()
    }
```

## Troubleshooting

### Issue: "Error loading models"
**Solution**: Ensure all model files exist in `models/` directory:
- `fraud_detection_model.pkl`
- `fraud_detection_model_robust.pkl`
- `scaler.pkl`, `scaler_robust.pkl`
- `label_encoders.pkl`, `label_encoders_robust.pkl`
- `model_metadata.pkl`, `model_metadata_robust.pkl`
- `preprocessing_config.pkl`

### Issue: "Column not found"
**Solution**: Input CSV must have same columns as training data. Check for:
- Typos in column names
- Missing required columns
- Different data types

### Issue: High unknown category rate
**Solution**: 
1. Review new categories in production data
2. Retrain model with updated data
3. Update preprocessing config

## Security Considerations

1. **PII Protection**: Ensure predictions don't expose sensitive data
2. **Access Control**: Restrict model file access
3. **Audit Logging**: Log all predictions for compliance
4. **Data Validation**: Validate input data before prediction

## Support

For issues or questions:
1. Check this documentation
2. Review notebook 08 for model details
3. Contact ML team for retraining requests
