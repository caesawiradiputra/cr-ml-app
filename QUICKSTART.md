# Quick Start Example

## Create a test file from existing processed data

```bash
# Activate environment
conda activate cr-ml-app-env

# Create sample test data (100 rows from processed dataset)
python -c "import pandas as pd; df = pd.read_csv('data/af_dataset_processed.csv').sample(100, random_state=42).drop(columns=['is_fraud'], errors='ignore'); df.to_csv('data/sample_input.csv', index=False); print(f'Created sample_input.csv with {len(df)} rows')"

# Run inference on both models
python src/production_inference.py --input data/sample_input.csv --output data/sample_output.csv

# Check results
python -c "import pandas as pd; df = pd.read_csv('data/sample_output.csv'); print(f'\\nResults: {len(df)} rows'); print(f'\\nNew columns: {[c for c in df.columns if \"model\" in c or \"agree\" in c or \"timestamp\" in c]}'); print(f'\\nLeaky model fraud: {df[\"leaky_model_fraud_flag\"].sum()}'); print(f'Robust model fraud: {df[\"robust_model_fraud_flag\"].sum()}'); print(f'Models agree: {df[\"models_agree\"].mean()*100:.1f}%')"
```

## Or use robust model only

```bash
python src/production_inference.py --input data/sample_input.csv --output data/sample_output_robust.csv --robust-only
```
