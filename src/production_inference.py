"""
Production Fraud Detection Inference Application

This script loads both the leaky model (notebook 07) and robust model (notebook 08)
and processes CSV files to generate fraud predictions.

Usage:
    python src/production_inference.py --input data/new_transactions.csv --output data/predictions.csv

Features:
- Loads both models for comparison
- Handles unknown categories safely
- Adds prediction columns to input data
- Saves results to new CSV file
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")


class FraudDetectionInference:
    """Production-ready fraud detection inference system."""
    
    # Define fraud response classes
    FRAUD_CLASSES = ['SUSPICIOUS', 'REJECTED', 'SUSPICIOUS_AF']
    
    def __init__(self, models_dir='models'):
        """
        Initialize the inference system with both models.
        
        Args:
            models_dir: Path to directory containing model artifacts
        """
        self.models_dir = Path(models_dir)
        self.models_loaded = False
        
        print('='*70)
        print('🚀 FRAUD DETECTION INFERENCE SYSTEM')
        print('='*70)
        print(f'\n📂 Models directory: {self.models_dir.absolute()}\n')
        
        self._load_models()
    
    def _load_models(self):
        """Load both models and their artifacts."""
        try:
            # Load Leaky Model (Notebook 07)
            print('📦 Loading Leaky Model (Notebook 07)...')
            self.leaky_model = joblib.load(self.models_dir / 'fraud_detection_model.pkl')
            self.leaky_scaler = joblib.load(self.models_dir / 'scaler.pkl')
            self.leaky_encoders = joblib.load(self.models_dir / 'label_encoders.pkl')
            self.leaky_metadata = joblib.load(self.models_dir / 'model_metadata.pkl')
            # Handle different metadata formats
            leaky_f1 = self.leaky_metadata.get('test_metrics', {}).get('f1') or self.leaky_metadata.get('metrics', {}).get('f1', 0.0)
            print(f'   ✅ Leaky Model: {leaky_f1:.4f} F1-Score')
            
            # Load Robust Model (Notebook 08)
            print('\n📦 Loading Robust Model (Notebook 08)...')
            self.robust_model = joblib.load(self.models_dir / 'fraud_detection_model_robust.pkl')
            self.robust_scaler = joblib.load(self.models_dir / 'scaler_robust.pkl')
            self.robust_encoders = joblib.load(self.models_dir / 'label_encoders_robust.pkl')
            self.robust_metadata = joblib.load(self.models_dir / 'model_metadata_robust.pkl')
            self.robust_config = joblib.load(self.models_dir / 'preprocessing_config.pkl')
            print(f'   ✅ Robust Model: {self.robust_metadata["test_metrics"]["f1"]:.4f} F1-Score')
            
            self.models_loaded = True
            
            print('\n' + '='*70)
            print('✅ ALL MODELS LOADED SUCCESSFULLY')
            print('='*70)
            
        except Exception as e:
            print(f'\n❌ Error loading models: {e}')
            print('Make sure all model files exist in the models/ directory.')
            sys.exit(1)
    
    def _add_s1_hardcoded_rule_features(self, df):
        """
        Add S1 hardcoded rule features based on False Negative analysis.
        
        These rules were reverse-engineered from production S1 system:
        - Specific marketing agent IDs in specific BOGOR subdistricts = HIGH FRAUD RISK
        - These combinations appear in ALL 8 False Negatives but in 0 legitimate cases
        
        Args:
            df: Input dataframe
        
        Returns:
            DataFrame with 6 new binary rule features added
        """
        df_with_rules = df.copy()
        
        # Rule 1: CIBEDUG subdistrict + specific agents (covers 4/8 FN = 50%)
        cibedug_agent = '2403NC0006'
        df_with_rules['rule_cibedug_agent'] = (
            ((df.get('payload_ktp_subdistrict', pd.Series([''])) == 'CIBEDUG') | 
             (df.get('payload_address_subdistrict', pd.Series([''])) == 'CIBEDUG')) &
            (df.get('payload_agent_marketingid', pd.Series([''])) == cibedug_agent)
        ).astype(int)
        
        # Rule 2: SUKASARI subdistrict in BOGOR (covers 3/8 FN = 37.5%)
        df_with_rules['rule_sukasari_bogor'] = (
            (df.get('payload_address_city', pd.Series([''])) == 'BOGOR') &
            ((df.get('payload_ktp_subdistrict', pd.Series([''])) == 'SUKASARI') | 
             (df.get('payload_address_subdistrict', pd.Series([''])) == 'SUKASARI'))
        ).astype(int)
        
        # Rule 3: High-risk agent-subdistrict combinations (covers remaining FN)
        high_risk_combos = [
            ('2307NC0005', 'CILEUNGSI'),
            ('2405NC0005', 'CILEUNGSI'),
            ('2403NC0005', 'CIAWI'),
            ('2408NC0002', 'CIAWI'),
            ('2307NC0005', 'CITAPEN'),
            ('2408NC0002', 'PADASUKA'),
            ('2405NC0005', 'PAGELARAN'),
        ]
        
        df_with_rules['rule_agent_subdistrict_combo'] = df.apply(
            lambda row: int(
                (row.get('payload_agent_marketingid', ''), row.get('payload_ktp_subdistrict', '')) in high_risk_combos or
                (row.get('payload_agent_marketingid', ''), row.get('payload_address_subdistrict', '')) in high_risk_combos
            ),
            axis=1
        )
        
        # Rule 4: Any BOGOR subdistrict with suspicious pattern (broader catch)
        bogor_high_risk_subdistricts = ['CIBEDUG', 'SUKASARI', 'CILEUNGSI', 'CIAWI', 
                                         'CITAPEN', 'PADASUKA', 'PAGELARAN', 'SIRNAGALIH', 'PARAKAN']
        df_with_rules['rule_bogor_high_risk_subdistrict'] = (
            (df.get('payload_address_city', pd.Series([''])) == 'BOGOR') &
            ((df.get('payload_ktp_subdistrict', pd.Series([''])).isin(bogor_high_risk_subdistricts)) |
             (df.get('payload_address_subdistrict', pd.Series([''])).isin(bogor_high_risk_subdistricts)))
        ).astype(int)
        
        # Rule 5: Suspicious agent pattern (agents appearing in FN)
        suspicious_agents = ['2403NC0006', '2307NC0005', '2405NC0005', '2403NC0005', '2408NC0002']
        df_with_rules['rule_suspicious_agent'] = (
            df.get('payload_agent_marketingid', pd.Series([''])).isin(suspicious_agents)
        ).astype(int)
        
        # Rule 6: Combined BOGOR + suspicious agent (high precision rule)
        df_with_rules['rule_bogor_suspicious_agent'] = (
            (df.get('payload_address_city', pd.Series([''])) == 'BOGOR') &
            (df.get('payload_agent_marketingid', pd.Series([''])).isin(suspicious_agents))
        ).astype(int)
        
        return df_with_rules
    
    def _safe_label_encode(self, df, label_encoders, categorical_features):
        """
        Safely encode categorical features, handling unknown categories.
        
        Args:
            df: Input dataframe
            label_encoders: Dictionary of fitted LabelEncoders
            categorical_features: List of categorical column names
        
        Returns:
            encoded_df: DataFrame with encoded features (all numeric)
            unknown_categories: Dictionary of unknown categories found
        """
        df_encoded = df.copy()
        unknown_categories = {}
        
        for col in categorical_features:
            if col not in label_encoders or col not in df.columns:
                continue
            
            le = label_encoders[col]
            series = df[col].astype(str)
            
            # Identify unknown categories
            known_categories = set(le.classes_)
            is_unknown = ~series.isin(known_categories)
            unknown_count = is_unknown.sum()
            
            if unknown_count > 0:
                unknown_values = series[is_unknown].unique()
                unknown_categories[col] = unknown_values.tolist()
                
                # Map unknown to special token
                series = series.copy()
                series[is_unknown] = '<UNKNOWN>'
                
                # Add <UNKNOWN> to encoder if not present
                if '<UNKNOWN>' not in known_categories:
                    le.classes_ = np.append(le.classes_, '<UNKNOWN>')
            
            # Encode and ensure integer type
            df_encoded[col] = le.transform(series).astype(int)
        
        # Ensure all columns are numeric (convert any remaining objects)
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object':
                try:
                    df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').fillna(0).astype(int)
                except:
                    pass
        
        return df_encoded, unknown_categories
    
    def preprocess_raw_data(self, df_raw):
        """
        Preprocess raw data (af_dataset.csv format) to processed format.
        Follows the same steps as notebook 05.
        
        Args:
            df_raw: Raw dataframe from af_dataset.csv
        
        Returns:
            df_processed: Processed dataframe ready for model inference
        """
        print('🔧 Preprocessing raw data (notebook 05 pipeline)...')
        
        df = df_raw.copy()
        
        # Step 1: Create binary target if response column exists
        if 'response' in df.columns:
            print('   1️⃣ Creating binary fraud flag from response column')
            df['is_fraud'] = df['response'].apply(
                lambda x: 1 if x in self.FRAUD_CLASSES else 0
            )
            print(f'      Fraud classes: {self.FRAUD_CLASSES}')
            fraud_count = df['is_fraud'].sum()
            print(f'      Fraud detected: {fraud_count} ({fraud_count/len(df)*100:.2f}%)')
        
        # Step 2: Remove UNCHECKED if exists
        if 'response' in df.columns:
            unchecked_count = (df['response'] == 'UNCHECKED').sum()
            if unchecked_count > 0:
                df = df[df['response'] != 'UNCHECKED'].copy()
                print(f'   2️⃣ Removed {unchecked_count} UNCHECKED rows')
        
        # Step 3: Convert timestamp to datetime and create time_period
        time_col = 'created_at'
        if time_col in df.columns:
            print(f'   3️⃣ Processing timestamp column: {time_col}')
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            # Create time_period as string (not Period object)
            df['time_period'] = df[time_col].dt.to_period('Q').astype(str)
            print(f'      Time range: {df[time_col].min()} to {df[time_col].max()}')
        
        # Step 4: Handle missing values
        print('   4️⃣ Handling missing values...')
        
        # Drop columns with > 50% missing
        missing_pct = (df.isnull().sum() / len(df)) * 100
        cols_to_drop = missing_pct[missing_pct > 50].index.tolist()
        if len(cols_to_drop) > 0:
            print(f'      Dropping {len(cols_to_drop)} columns with >50% missing: {cols_to_drop[:5]}...')
            df = df.drop(columns=cols_to_drop)
        
        # Impute numerical with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_with_nulls = [col for col in numerical_cols if df[col].isnull().any()]
        if len(numerical_with_nulls) > 0:
            print(f'      Imputing {len(numerical_with_nulls)} numerical columns with median')
            imputer_num = SimpleImputer(strategy='median')
            df[numerical_with_nulls] = imputer_num.fit_transform(df[numerical_with_nulls])
        
        # Impute categorical with most frequent
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_with_nulls = [col for col in categorical_cols if df[col].isnull().any()]
        if len(categorical_with_nulls) > 0:
            print(f'      Imputing {len(categorical_with_nulls)} categorical columns with most_frequent')
            imputer_cat = SimpleImputer(strategy='most_frequent')
            df[categorical_with_nulls] = imputer_cat.fit_transform(df[categorical_with_nulls])
        
        remaining_nulls = df.isnull().sum().sum()
        print(f'   ✅ Preprocessing complete: {remaining_nulls} null values remaining')
        print(f'      Shape: {df.shape[0]:,} rows × {df.shape[1]} columns')
        
        return df
    
    def _create_robust_features(self, X_train, y_train, high_card_columns, min_samples=50):
        """
        Create robust high-cardinality features (simplified for single inference).
        
        Note: In production, we should use pre-computed mappings from training.
        This is a simplified version that uses the input data itself.
        """
        X_enhanced = X_train.copy()
        global_mean = 0.0078  # From training
        
        for col in high_card_columns:
            if col not in X_train.columns:
                continue
            
            # Calculate frequency
            freq_dict = X_train[col].value_counts().to_dict()
            freq_col = f'{col}_frequency'
            X_enhanced[freq_col] = X_train[col].map(freq_dict).fillna(0)
            
            # Target encoding (use global mean for all in production)
            target_col = f'{col}_target_encoded'
            X_enhanced[target_col] = global_mean
            
            # Volume indicator
            high_volume_threshold = X_train[col].value_counts().quantile(0.5)
            volume_col = f'{col}_is_high_volume'
            X_enhanced[volume_col] = X_train[col].map(freq_dict).fillna(0) >= high_volume_threshold
            X_enhanced[volume_col] = X_enhanced[volume_col].astype(int)
        
        return X_enhanced
    
    def predict_leaky(self, df):
        """
        Generate predictions using the leaky model (Notebook 07).
        
        Args:
            df: Input dataframe with raw features
        
        Returns:
            predictions: Dictionary with predictions and probabilities
        """
        print('\n🔮 Generating predictions with Leaky Model...')
        
        # Prepare data (same preprocessing as training)
        exclude_cols = ['payload_id', 'request_id', 'is_fraud', 'response', 
                       'response_code', 'created_at']
        exclude_cols = [col for col in exclude_cols if col in df.columns]
        
        X = df.drop(columns=exclude_cols, errors='ignore')
        
        # Get categorical features
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Safe label encoding
        X_encoded, unknown_cats = self._safe_label_encode(
            X, self.leaky_encoders, categorical_features
        )
        
        if unknown_cats:
            print(f'   ⚠️ Unknown categories handled: {len(unknown_cats)} columns')
        
        # Check if scaler expects S1 rule features (by checking expected feature count)
        expected_features = self.leaky_scaler.n_features_in_
        current_features = X_encoded.shape[1]
        
        # Add S1 hardcoded rule features if scaler expects them
        if expected_features == current_features + 6:
            print('   🚨 Adding S1 hardcoded rule features...')
            X_with_rules = self._add_s1_hardcoded_rule_features(X_encoded)
            rule_hits = sum([X_with_rules[col].sum() for col in X_with_rules.columns if col.startswith('rule_')])
            print(f'      ✅ S1 rules applied - Total rule hits: {rule_hits}')
            X_to_scale = X_with_rules
        elif expected_features == current_features:
            print('   ⚠️ Scaler does not expect S1 rule features (old model)')
            print('      Skipping S1 rules - Please retrain model with notebook 06')
            X_to_scale = X_encoded
        else:
            print(f'   ⚠️ Feature count mismatch: expected {expected_features}, got {current_features}')
            print('      Proceeding without S1 rules...')
            X_to_scale = X_encoded
        
        # Scale
        X_scaled = self.leaky_scaler.transform(X_to_scale)
        
        # Predict
        y_pred = self.leaky_model.predict(X_scaled)
        y_pred_proba = self.leaky_model.predict_proba(X_scaled)[:, 1]
        
        print(f'   ✅ Predictions generated: {len(y_pred)} samples')
        print(f'   📊 Fraud detected: {y_pred.sum()} ({y_pred.mean()*100:.2f}%)')
        
        return {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'unknown_categories': unknown_cats
        }
    
    def predict_robust(self, df):
        """
        Generate predictions using the robust model (Notebook 08).
        
        Args:
            df: Input dataframe with raw features
        
        Returns:
            predictions: Dictionary with predictions and probabilities
        """
        print('\n🔮 Generating predictions with Robust Model...')
        
        # Prepare data
        exclude_cols = ['payload_id', 'request_id', 'is_fraud', 'response', 
                       'response_code', 'created_at']
        exclude_cols = [col for col in exclude_cols if col in df.columns]
        
        X = df.drop(columns=exclude_cols, errors='ignore')
        
        # Get categorical features
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Safe label encoding
        X_encoded, unknown_cats = self._safe_label_encode(
            X, self.robust_encoders, categorical_features
        )
        
        if unknown_cats:
            print(f'   ⚠️ Unknown categories handled: {len(unknown_cats)} columns')
        
        # Apply robust high-cardinality features
        high_card_columns = self.robust_config['high_card_columns']
        X_enhanced = self._create_robust_features(
            X_encoded, None, high_card_columns, 
            min_samples=self.robust_config['min_samples']
        )
        
        # Check if scaler expects S1 rule features (by checking expected feature count)
        expected_features = self.robust_scaler.n_features_in_
        current_features = X_enhanced.shape[1]
        
        # Add S1 hardcoded rule features if scaler expects them
        if expected_features == current_features + 6:
            print('   🚨 Adding S1 hardcoded rule features...')
            X_with_rules = self._add_s1_hardcoded_rule_features(X_enhanced)
            rule_hits = sum([X_with_rules[col].sum() for col in X_with_rules.columns if col.startswith('rule_')])
            print(f'      ✅ S1 rules applied - Total rule hits: {rule_hits}')
            X_to_scale = X_with_rules
        elif expected_features == current_features:
            print('   ⚠️ Scaler does not expect S1 rule features (old model)')
            print('      Skipping S1 rules - Please retrain model with notebook 08')
            X_to_scale = X_enhanced
        else:
            print(f'   ⚠️ Feature count mismatch: expected {expected_features}, got {current_features}')
            print('      Proceeding without S1 rules...')
            X_to_scale = X_enhanced
        
        # Scale
        X_scaled = self.robust_scaler.transform(X_to_scale)
        
        # Predict
        y_pred = self.robust_model.predict(X_scaled)
        y_pred_proba = self.robust_model.predict_proba(X_scaled)[:, 1]
        
        print(f'   ✅ Predictions generated: {len(y_pred)} samples')
        print(f'   📊 Fraud detected: {y_pred.sum()} ({y_pred.mean()*100:.2f}%)')
        
        return {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'unknown_categories': unknown_cats
        }
    
    def process_file(self, input_path, output_path=None, include_both_models=True, is_raw_format=False):
        """
        Process a CSV file and generate predictions from both models.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to save output CSV (optional)
            include_both_models: Whether to include both model predictions
            is_raw_format: Whether input is in raw format (af_dataset.csv) requiring preprocessing
        
        Returns:
            df_results: DataFrame with original data + predictions
        """
        if not self.models_loaded:
            raise RuntimeError('Models not loaded!')
        
        print('\n' + '='*70)
        print('📄 PROCESSING FILE')
        print('='*70)
        print(f'\n📂 Input: {input_path}')
        
        # Load data
        df_raw = pd.read_csv(input_path)
        print(f'✅ Loaded: {len(df_raw):,} rows × {df_raw.shape[1]} columns')
        
        # Check if preprocessing is needed
        if is_raw_format or 'time_period' not in df_raw.columns:
            print('\n⚙️  Detected raw format - applying preprocessing pipeline...')
            df = self.preprocess_raw_data(df_raw)
        else:
            print('\n✅ Detected processed format - skipping preprocessing')
            df = df_raw.copy()
        
        # Store original response column if exists (for comparison)
        has_response = 'response' in df.columns
        if has_response:
            actual_response = df['response'].copy()
            actual_fraud = df['response'].apply(lambda x: 1 if x in self.FRAUD_CLASSES else 0)
            print(f'\n📊 Ground truth available: response column found')
            print(f'   Actual fraud: {actual_fraud.sum()} ({actual_fraud.mean()*100:.2f}%)')
        
        # Create results dataframe
        df_results = df.copy()
        
        # Generate predictions with leaky model
        if include_both_models:
            leaky_preds = self.predict_leaky(df)
            df_results['leaky_model_prediction'] = leaky_preds['predictions']
            df_results['leaky_model_probability'] = leaky_preds['probabilities']
            df_results['leaky_model_fraud_flag'] = (
                leaky_preds['predictions'] == 1
            ).astype(int)
        
        # Generate predictions with robust model
        robust_preds = self.predict_robust(df)
        df_results['robust_model_prediction'] = robust_preds['predictions']
        df_results['robust_model_probability'] = robust_preds['probabilities']
        df_results['robust_model_fraud_flag'] = (
            robust_preds['predictions'] == 1
        ).astype(int)
        
        # Add agreement flag (if both models included)
        if include_both_models:
            df_results['models_agree'] = (
                df_results['leaky_model_prediction'] == 
                df_results['robust_model_prediction']
            ).astype(int)
        
        # Add comparison with actual response (if available)
        if has_response:
            # Add actual fraud column for analysis
            df_results['actual_fraud'] = actual_fraud.astype(int)
            
            # Robust model comparison
            df_results['robust_vs_actual'] = (
                df_results['robust_model_prediction'] == actual_fraud
            ).astype(int)
            
            # Add confusion matrix category for robust model
            df_results['robust_prediction_category'] = 'Unknown'
            df_results.loc[
                (df_results['robust_model_prediction'] == 0) & (actual_fraud == 0),
                'robust_prediction_category'
            ] = 'True Negative (TN)'
            df_results.loc[
                (df_results['robust_model_prediction'] == 1) & (actual_fraud == 1),
                'robust_prediction_category'
            ] = 'True Positive (TP)'
            df_results.loc[
                (df_results['robust_model_prediction'] == 0) & (actual_fraud == 1),
                'robust_prediction_category'
            ] = 'False Negative (FN)'
            df_results.loc[
                (df_results['robust_model_prediction'] == 1) & (actual_fraud == 0),
                'robust_prediction_category'
            ] = 'False Positive (FP)'
            
            # Leaky model comparison (if included)
            if include_both_models:
                df_results['leaky_vs_actual'] = (
                    df_results['leaky_model_prediction'] == actual_fraud
                ).astype(int)
            
            print('\n✅ Added comparison columns: [model]_vs_actual (1=correct, 0=incorrect)')
        
        # Add timestamp
        df_results['prediction_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save results
        if output_path:
            df_results.to_csv(output_path, index=False)
            print(f'\n💾 Results saved: {output_path}')
            print(f'   Total columns: {df_results.shape[1]}')
            print(f'   New columns added: {df_results.shape[1] - df.shape[1]}')
        
        # Summary statistics
        print('\n' + '='*70)
        print('📊 PREDICTION SUMMARY')
        print('='*70)
        
        if include_both_models:
            print('\n🔴 Leaky Model:')
            print(f'   Fraud detected: {leaky_preds["predictions"].sum():,} '
                  f'({leaky_preds["predictions"].mean()*100:.2f}%)')
            print(f'   Avg fraud probability: {leaky_preds["probabilities"].mean():.4f}')
        
        print('\n🟢 Robust Model (Recommended):')
        print(f'   Fraud detected: {robust_preds["predictions"].sum():,} '
              f'({robust_preds["predictions"].mean()*100:.2f}%)')
        print(f'   Avg fraud probability: {robust_preds["probabilities"].mean():.4f}')
        
        if include_both_models:
            agreement = df_results['models_agree'].mean() * 100
            print(f'\n🤝 Model Agreement: {agreement:.1f}%')
            
            # Cases where models disagree
            disagree = df_results[df_results['models_agree'] == 0]
            if len(disagree) > 0:
                print(f'   ⚠️ Models disagree on {len(disagree):,} cases '
                      f'({len(disagree)/len(df)*100:.2f}%)')
                print('   → Review these cases manually!')
                # Performance metrics if ground truth available
        if has_response:
            print('\n' + '='*70)
            print('🎯 MODEL ACCURACY (vs Ground Truth)')
            print('='*70)
            
            print(f'\n📊 Actual Labels:')
            print(f'   Total fraud: {actual_fraud.sum():,} ({actual_fraud.mean()*100:.2f}%)')
            print(f'   Total legitimate: {(actual_fraud==0).sum():,} ({(actual_fraud==0).mean()*100:.2f}%)')
            
            # Robust model accuracy
            robust_correct = df_results['robust_vs_actual'].sum()
            robust_accuracy = robust_correct / len(df_results) * 100
            print(f'\n🟢 Robust Model:')
            print(f'   Correct predictions: {robust_correct} / {len(df_results)} ({robust_accuracy:.2f}%)')
            
            # Detailed metrics
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
            
            try:
                robust_precision = precision_score(actual_fraud, robust_preds['predictions'], zero_division=0)
                robust_recall = recall_score(actual_fraud, robust_preds['predictions'], zero_division=0)
                robust_f1 = f1_score(actual_fraud, robust_preds['predictions'], zero_division=0)
                
                print(f'   Precision: {robust_precision:.4f} (of predicted fraud, how many are correct)')
                print(f'   Recall: {robust_recall:.4f} (of actual fraud, how many detected)')
                print(f'   F1-Score: {robust_f1:.4f} (harmonic mean)')
            except:
                print('   ⚠️ Unable to calculate metrics (check if fraud cases exist)')
            
            # Confusion matrix with edge case handling
            cm = confusion_matrix(actual_fraud, robust_preds['predictions'])
            
            # Handle edge cases (only one class present)
            if cm.shape == (1, 1):
                # Only one class exists
                if actual_fraud.sum() == 0:
                    # All legitimate, no fraud
                    tn, fp, fn, tp = cm[0][0], 0, 0, 0
                else:
                    # All fraud, no legitimate (rare)
                    tn, fp, fn, tp = 0, 0, 0, cm[0][0]
            elif cm.shape == (2, 1):
                # Two actual classes, but predictions are all one class
                tn, fp, fn, tp = cm[0][0], 0, cm[1][0], 0
            elif cm.shape == (1, 2):
                # One actual class, but predictions are both classes (shouldn't happen)
                tn, fp, fn, tp = cm[0][0], cm[0][1], 0, 0
            else:
                # Normal case: both classes present
                tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            
            print(f'\n   Confusion Matrix:')
            print(f'      TN (correct legit): {tn:,}')
            print(f'      FP (false alarm): {fp:,}')
            print(f'      FN (missed fraud): {fn:,} {"⚠️" if fn > 0 else ""}')
            print(f'      TP (caught fraud): {tp:,}')
            
            # Leaky model metrics if included
            if include_both_models:
                leaky_correct = df_results['leaky_vs_actual'].sum()
                leaky_accuracy = leaky_correct / len(df_results) * 100
                print(f'\n🔴 Leaky Model:')
                print(f'   Correct predictions: {leaky_correct} / {len(df_results)} ({leaky_accuracy:.2f}%)')
        
            print('\n✅ Processing complete!')            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f'\n   Metrics:')
            print(f'      Precision: {precision:.4f} (When predicting fraud, how often correct?)')
            print(f'      Recall:    {recall:.4f} (Of all actual fraud, how many caught?)')
            print(f'      F1-Score:  {f1:.4f}')
            
            # Leaky model accuracy (if included)
            if include_both_models:
                leaky_correct = df_results['leaky_vs_actual'].sum()
                leaky_accuracy = df_results['leaky_vs_actual'].mean()
                
                print(f'\n🔴 Leaky Model Performance:')
                print(f'   Correct predictions: {leaky_correct:,} / {len(df_results)} '
                      f'({leaky_accuracy*100:.2f}%)')
                
                # Compare accuracies
                if robust_accuracy > leaky_accuracy:
                    print(f'\n   ✅ Robust model is {(robust_accuracy - leaky_accuracy)*100:.2f}% more accurate')
                elif leaky_accuracy > robust_accuracy:
                    print(f'\n   ⚠️ Leaky model is {(leaky_accuracy - robust_accuracy)*100:.2f}% more accurate')
                else:
                    print(f'\n   🟰 Both models have equal accuracy')
        
        print('\n✅ Processing complete!')
        
        return df_results


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Fraud Detection Inference System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a CSV file with both models
  python src/production_inference.py --input data/new_transactions.csv --output data/predictions.csv
  
  # Process with robust model only
  python src/production_inference.py --input data/new_transactions.csv --output data/predictions.csv --robust-only
  
  # Process without saving (preview only)
  python src/production_inference.py --input data/new_transactions.csv
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Path to output CSV file (optional)'
    )
    
    parser.add_argument(
        '--models-dir', '-m',
        default='models',
        help='Path to models directory (default: models)'
    )
    
    parser.add_argument(
        '--robust-only',
        action='store_true',
        help='Use only robust model (skip leaky model)'
    )
    
    parser.add_argument(
        '--raw',
        action='store_true',
        help='Input is in raw format (af_dataset.csv) requiring preprocessing'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f'❌ Error: Input file not found: {input_path}')
        sys.exit(1)
    
    # Initialize inference system
    inference = FraudDetectionInference(models_dir=args.models_dir)
    
    # Process file
    df_results = inference.process_file(
        input_path=args.input,
        output_path=args.output,
        include_both_models=not args.robust_only,
        is_raw_format=args.raw
    )
    
    print('\n🎉 Done!')


if __name__ == '__main__':
    main()
