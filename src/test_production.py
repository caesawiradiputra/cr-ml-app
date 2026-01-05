"""
Test script for production inference system.

This script creates a sample dataset and tests both models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Force UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.production_inference import FraudDetectionInference


def create_sample_data(n_samples=100, output_path='data/temp/sample_transactions.csv'):
    """
    Create sample transaction data for testing.
    
    Uses the PROCESSED data (same format as training).
    """
    print('🔧 Creating sample test data...')
    
    # Load PROCESSED data (same format as training)
    processed_data_path = Path('data/af_dataset_processed.csv')
    
    if not processed_data_path.exists():
        print(f'❌ Processed data not found: {processed_data_path}')
        print('   Cannot create sample data without template.')
        print('   Please run notebook 05 first to create af_dataset_processed.csv')
        return None
    
    # Load and sample
    print(f'   Loading processed data from: {processed_data_path}')
    df_original = pd.read_csv(processed_data_path)
    df_sample = df_original.sample(n=min(n_samples, len(df_original)), random_state=42)
    
    # Remove target column if present (simulate production data without labels)
    if 'is_fraud' in df_sample.columns:
        actual_fraud_count = df_sample['is_fraud'].sum()
        print(f'   📊 Actual fraud in sample: {actual_fraud_count} / {len(df_sample)} ({actual_fraud_count/len(df_sample)*100:.1f}%)')
        df_sample = df_sample.drop(columns=['is_fraud'])
    
    # Save sample
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(output_path, index=False)
    
    print(f'✅ Sample data created: {output_path}')
    print(f'   {len(df_sample)} samples × {df_sample.shape[1]} columns')
    print(f'   Data type: PROCESSED (same format as training)')
    
    return output_path


def test_inference_system():
    """Test the inference system with sample data."""
    print('='*70)
    print('🧪 TESTING PRODUCTION INFERENCE SYSTEM')
    print('='*70)
    
    # Step 1: Create sample data
    sample_path = create_sample_data(n_samples=100)
    
    if sample_path is None:
        print('❌ Test failed: Could not create sample data')
        return False
    
    # Step 2: Initialize inference system
    print('\n' + '='*70)
    print('📦 Initializing Inference System')
    print('='*70)
    
    try:
        inference = FraudDetectionInference(models_dir='models')
    except Exception as e:
        print(f'❌ Test failed: Could not initialize inference system')
        print(f'   Error: {e}')
        return False
    
    # Step 3: Test predictions
    print('\n' + '='*70)
    print('🔮 Testing Predictions')
    print('='*70)
    
    try:
        output_path = Path('data/temp/sample_predictions.csv')
        df_results = inference.process_file(
            input_path=sample_path,
            output_path=output_path,
            include_both_models=True
        )
        
        print(f'\n✅ Predictions generated successfully!')
        print(f'   Output saved: {output_path}')
        
    except Exception as e:
        print(f'❌ Test failed: Error during prediction')
        print(f'   Error: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Validate output
    print('\n' + '='*70)
    print('✅ VALIDATION')
    print('='*70)
    
    # Check columns
    expected_new_cols = [
        'leaky_model_prediction',
        'leaky_model_probability',
        'leaky_model_fraud_flag',
        'robust_model_prediction',
        'robust_model_probability',
        'robust_model_fraud_flag',
        'models_agree',
        'prediction_timestamp'
    ]
    
    missing_cols = [col for col in expected_new_cols if col not in df_results.columns]
    
    if missing_cols:
        print(f'❌ Missing columns: {missing_cols}')
        return False
    
    print(f'✅ All expected columns present')
    
    # Check predictions are valid
    leaky_preds = df_results['leaky_model_prediction'].unique()
    robust_preds = df_results['robust_model_prediction'].unique()
    
    if not all(p in [0, 1] for p in leaky_preds):
        print(f'❌ Invalid leaky predictions: {leaky_preds}')
        return False
    
    if not all(p in [0, 1] for p in robust_preds):
        print(f'❌ Invalid robust predictions: {robust_preds}')
        return False
    
    print(f'✅ Predictions are valid (0 or 1)')
    
    # Check probabilities
    if not (0 <= df_results['leaky_model_probability'].min() <= 1):
        print(f'❌ Invalid leaky probabilities')
        return False
    
    if not (0 <= df_results['robust_model_probability'].min() <= 1):
        print(f'❌ Invalid robust probabilities')
        return False
    
    print(f'✅ Probabilities are in valid range [0, 1]')
    
    # Show sample results
    print('\n' + '='*70)
    print('📊 SAMPLE RESULTS (First 5 rows)')
    print('='*70)
    
    display_cols = [
        'payload_id',
        'leaky_model_prediction',
        'leaky_model_probability',
        'robust_model_prediction',
        'robust_model_probability',
        'models_agree'
    ]
    
    if 'payload_id' in df_results.columns:
        print(df_results[display_cols].head().to_string(index=False))
    
    # Summary statistics
    print('\n' + '='*70)
    print('📈 SUMMARY STATISTICS')
    print('='*70)
    
    print(f'\nLeaky Model:')
    print(f'   Fraud detected: {df_results["leaky_model_fraud_flag"].sum()} / {len(df_results)} '
          f'({df_results["leaky_model_fraud_flag"].mean()*100:.1f}%)')
    print(f'   Avg probability: {df_results["leaky_model_probability"].mean():.4f}')
    
    print(f'\nRobust Model:')
    print(f'   Fraud detected: {df_results["robust_model_fraud_flag"].sum()} / {len(df_results)} '
          f'({df_results["robust_model_fraud_flag"].mean()*100:.1f}%)')
    print(f'   Avg probability: {df_results["robust_model_probability"].mean():.4f}')
    
    agreement = df_results['models_agree'].mean() * 100
    print(f'\nModel Agreement: {agreement:.1f}%')
    
    disagree_count = len(df_results[df_results['models_agree'] == 0])
    if disagree_count > 0:
        print(f'   ⚠️ Models disagree on {disagree_count} cases ({disagree_count/len(df_results)*100:.1f}%)')
    
    print('\n' + '='*70)
    print('✅ ALL TESTS PASSED!')
    print('='*70)
    print('\n🎉 Production inference system is working correctly!')
    
    return True


if __name__ == '__main__':
    success = test_inference_system()
    sys.exit(0 if success else 1)
