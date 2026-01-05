# Production Inference - Examples

## Example 1: Basic Inference (Robust Model Only)

```bash
# Activate environment
conda activate cr-ml-app-env

# Create sample data from processed dataset
python -c "import pandas as pd; df = pd.read_csv('data/af_dataset_processed.csv').sample(100, random_state=42).drop(columns=['is_fraud'], errors='ignore'); df.to_csv('data/sample_input.csv', index=False)"

# Run inference (recommended)
python src/production_inference.py \
    --input data/sample_input.csv \
    --output data/sample_output.csv \
    --robust-only

# Check results
python -c "import pandas as pd; df = pd.read_csv('data/sample_output.csv'); print(f'Total: {len(df)} transactions'); print(f'Fraud detected: {df[\"robust_model_fraud_flag\"].sum()} cases ({df[\"robust_model_fraud_flag\"].sum()/len(df)*100:.2f}%)'); print(f'Avg probability: {df[\"robust_model_probability\"].mean():.6f}')"
```

**Expected Output**:
```
Total: 100 transactions
Fraud detected: 1 cases (1.00%)
Avg probability: 0.010000
```

---

## Example 2: Python Script Integration

```python
import pandas as pd
from src.production_inference import FraudDetectionInference

# Initialize inference system
inference = FraudDetectionInference(models_dir='models')

# Load your data
df = pd.read_csv('data/new_transactions.csv')

# Generate predictions (robust model only)
results = inference.process_file(
    input_path='data/new_transactions.csv',
    output_path='data/predictions.csv',
    include_both_models=False  # Robust model only
)

# Filter fraud cases
fraud_cases = results[results['robust_model_fraud_flag'] == 1]

print(f"Total transactions: {len(results)}")
print(f"Fraud detected: {len(fraud_cases)} ({len(fraud_cases)/len(results)*100:.2f}%)")

# High confidence fraud (probability > 50%)
high_conf = results[results['robust_model_probability'] > 0.5]
print(f"High confidence fraud: {len(high_conf)}")

# Export fraud cases for review
fraud_cases.to_csv('data/fraud_cases_for_review.csv', index=False)
```

---

## Example 3: Batch Processing Large Files

```python
import pandas as pd
from src.production_inference import FraudDetectionInference

# Initialize once
inference = FraudDetectionInference(models_dir='models')

# Process in chunks for large files
chunk_size = 10000
results_list = []

for i, chunk in enumerate(pd.read_csv('data/large_file.csv', chunksize=chunk_size)):
    print(f"Processing chunk {i+1}...")
    
    # Save chunk temporarily
    chunk.to_csv('data/temp_chunk.csv', index=False)
    
    # Process chunk
    chunk_results = inference.process_file(
        input_path='data/temp_chunk.csv',
        include_both_models=False
    )
    
    results_list.append(chunk_results)

# Combine all results
final_results = pd.concat(results_list, ignore_index=True)
final_results.to_csv('data/large_file_predictions.csv', index=False)

print(f"Total processed: {len(final_results):,} transactions")
print(f"Total fraud: {final_results['robust_model_fraud_flag'].sum():,}")
```

---

## Example 4: Filter and Act on Predictions

```python
import pandas as pd

# Load predictions
df = pd.read_csv('data/predictions.csv')

# Define risk thresholds
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.3

# Categorize by risk level
df['risk_level'] = 'Low'
df.loc[df['robust_model_probability'] > MEDIUM_RISK_THRESHOLD, 'risk_level'] = 'Medium'
df.loc[df['robust_model_probability'] > HIGH_RISK_THRESHOLD, 'risk_level'] = 'High'

# Count by risk level
risk_counts = df['risk_level'].value_counts()
print("\nRisk Distribution:")
print(risk_counts)

# Export by risk level
for risk_level in ['High', 'Medium', 'Low']:
    risk_df = df[df['risk_level'] == risk_level]
    risk_df.to_csv(f'data/risk_{risk_level.lower()}_cases.csv', index=False)
    print(f"\n{risk_level} risk cases: {len(risk_df)} ({len(risk_df)/len(df)*100:.2f}%)")

# Auto-approve low risk (fraud probability < 1%)
auto_approve = df[df['robust_model_probability'] < 0.01]
print(f"\nAuto-approve: {len(auto_approve)} transactions ({len(auto_approve)/len(df)*100:.1f}%)")

# Manual review required
manual_review = df[df['robust_model_probability'] >= 0.01]
print(f"Manual review: {len(manual_review)} transactions ({len(manual_review)/len(df)*100:.1f}%)")
```

---

## Example 5: Daily Batch Job (Scheduled Task)

Create a script `run_daily_fraud_detection.py`:

```python
#!/usr/bin/env python3
"""
Daily fraud detection batch job.
Processes new transactions and flags suspicious cases.
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from src.production_inference import FraudDetectionInference

def run_daily_detection():
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Paths
    input_file = f'data/transactions_{today}.csv'
    output_file = f'data/predictions_{today}.csv'
    fraud_file = f'data/fraud_cases_{today}.csv'
    
    print(f"Starting fraud detection for {today}...")
    
    # Check if input exists
    if not Path(input_file).exists():
        print(f"No input file found: {input_file}")
        return
    
    # Initialize inference
    inference = FraudDetectionInference(models_dir='models')
    
    # Process file
    results = inference.process_file(
        input_path=input_file,
        output_path=output_file,
        include_both_models=False
    )
    
    # Extract fraud cases
    fraud = results[results['robust_model_fraud_flag'] == 1]
    fraud.to_csv(fraud_file, index=False)
    
    # Summary
    print(f"\nSummary for {today}:")
    print(f"  Total transactions: {len(results):,}")
    print(f"  Fraud detected: {len(fraud):,} ({len(fraud)/len(results)*100:.2f}%)")
    print(f"  Output: {output_file}")
    print(f"  Fraud cases: {fraud_file}")
    
    # Alert if fraud rate is unusual
    fraud_rate = len(fraud) / len(results)
    baseline_rate = 0.0078
    
    if abs(fraud_rate - baseline_rate) > 0.01:
        print(f"\n⚠️  ALERT: Fraud rate ({fraud_rate:.2%}) deviates from baseline ({baseline_rate:.2%})")
        print("  Consider manual review or model retraining.")

if __name__ == '__main__':
    run_daily_detection()
```

Run as scheduled task (Windows Task Scheduler or cron):

```bash
# Windows Task Scheduler
# Action: Start a program
# Program: C:\Users\YourUser\AppData\Local\miniconda3\envs\cr-ml-app-env\python.exe
# Arguments: run_daily_fraud_detection.py
# Start in: C:\path\to\cr-ml-app

# Linux cron
# 0 1 * * * cd /path/to/cr-ml-app && /path/to/conda/envs/cr-ml-app-env/bin/python run_daily_fraud_detection.py
```

---

## Example 6: Performance Monitoring

```python
import pandas as pd
from datetime import datetime, timedelta

# Load predictions from last 30 days
date_range = pd.date_range(
    end=datetime.now(),
    periods=30,
    freq='D'
)

daily_stats = []

for date in date_range:
    date_str = date.strftime('%Y-%m-%d')
    pred_file = f'data/predictions_{date_str}.csv'
    
    try:
        df = pd.read_csv(pred_file)
        
        stats = {
            'date': date_str,
            'total_transactions': len(df),
            'fraud_detected': df['robust_model_fraud_flag'].sum(),
            'fraud_rate': df['robust_model_fraud_flag'].mean(),
            'avg_probability': df['robust_model_probability'].mean(),
            'high_confidence_fraud': (df['robust_model_probability'] > 0.7).sum()
        }
        
        daily_stats.append(stats)
    except FileNotFoundError:
        continue

# Create monitoring dashboard
stats_df = pd.DataFrame(daily_stats)
print("\nLast 30 Days - Fraud Detection Statistics")
print("="*70)
print(stats_df.to_string(index=False))

# Calculate trends
print("\nTrends:")
print(f"  Avg fraud rate: {stats_df['fraud_rate'].mean():.4f}")
print(f"  Fraud rate std: {stats_df['fraud_rate'].std():.4f}")
print(f"  Avg confidence: {stats_df['avg_probability'].mean():.6f}")

# Visualize (if matplotlib available)
try:
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Fraud rate over time
    ax1.plot(stats_df['date'], stats_df['fraud_rate'], marker='o')
    ax1.set_title('Daily Fraud Detection Rate')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Fraud Rate')
    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45)
    
    # Average probability over time
    ax2.plot(stats_df['date'], stats_df['avg_probability'], marker='o', color='orange')
    ax2.set_title('Average Fraud Probability')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Probability')
    ax2.grid(True)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('data/fraud_detection_trends.png', dpi=300)
    print("\n  Chart saved: data/fraud_detection_trends.png")
    
except ImportError:
    print("\n  (matplotlib not available for visualization)")
```

---

## Common Issues and Solutions

### Issue 1: "Models not loaded"
```bash
# Check if model files exist
ls models/*.pkl

# Expected files:
# - fraud_detection_model_robust.pkl
# - scaler_robust.pkl
# - label_encoders_robust.pkl
# - model_metadata_robust.pkl
# - preprocessing_config.pkl
```

### Issue 2: "Column not found"
```python
# Verify input CSV has required columns
import pandas as pd

required_cols = pd.read_csv('data/af_dataset_processed.csv', nrows=1).columns
input_df = pd.read_csv('data/your_input.csv')

missing_cols = set(required_cols) - set(input_df.columns)
if missing_cols:
    print(f"Missing columns: {missing_cols}")
else:
    print("✅ All required columns present")
```

### Issue 3: High unknown category rate
```python
# Check unknown categories
from src.production_inference import FraudDetectionInference

inference = FraudDetectionInference()
predictions = inference.predict_robust(df)

if predictions['unknown_categories']:
    print("Unknown categories detected:")
    for col, values in predictions['unknown_categories'].items():
        print(f"  {col}: {len(values)} new categories")
        print(f"    Examples: {values[:5]}")
    
    print("\nConsider retraining model with new data.")
```

---

## Next Steps

- **Production API**: Convert to FastAPI for real-time inference
- **Monitoring Dashboard**: Build Grafana/Streamlit dashboard
- **Automated Retraining**: Set up MLOps pipeline
- **Model Registry**: Version control for models

See [docs/production_deployment.md](docs/production_deployment.md) for more details.
