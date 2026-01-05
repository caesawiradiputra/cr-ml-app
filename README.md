# ML Learning Project

A Python-based machine learning workspace focused on fraud detection with production-ready deployment.

## 🎯 Project Status

**Current Focus**: Fraud Detection System  
**Deployment Status**: ✅ **PRODUCTION READY**  
**Latest Model**: Robust GradientBoosting (94.82% F1-Score)

## Project Structure

```
cr-ml-app/
├── data/                 # Datasets and sample files
│   ├── af_dataset.csv              # Raw fraud detection data
│   ├── af_dataset_processed.csv    # Processed data with features
│   ├── sample_input.csv            # Sample input for testing
│   └── sample_output.csv           # Sample predictions
├── models/               # Trained models and artifacts
│   ├── fraud_detection_model_robust.pkl     # Robust model (recommended)
│   ├── fraud_detection_model.pkl            # Leaky model (not recommended)
│   ├── scaler_robust.pkl                    # Feature scaler
│   ├── label_encoders_robust.pkl            # Category encoders
│   ├── model_metadata_robust.pkl            # Model metadata
│   └── preprocessing_config.pkl             # Preprocessing configuration
├── notebooks/            # Jupyter notebooks for exploration and analysis
│   ├── 01_iris_classification.ipynb         # Learning: Iris classification
│   ├── 02_algorithm_comparison.ipynb        # Learning: Algorithm comparison
│   ├── 03_noise_level_assessment.ipynb      # Learning: Noise assessment
│   ├── 04_hyperparameter_tuning.ipynb       # Learning: Hyperparameter tuning
│   ├── 05_fraud_data_processing.ipynb       # Fraud: Data preprocessing
│   ├── 07_hyperparameter_tuning.ipynb       # Fraud: Leaky model (91.78% F1)
│   └── 08_retrain_without_leakage.ipynb     # Fraud: Robust model (94.82% F1) ✅
├── src/                  # Source code and production scripts
│   ├── production_inference.py              # Production inference system ✅
│   └── test_production.py                   # Test script
├── docs/                 # Documentation
│   ├── DEPLOYMENT_SUMMARY.md               # Deployment summary ✅
│   └── production_deployment.md            # Detailed deployment guide
├── QUICKSTART.md         # Quick start guide ✅
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Quick Start - Production Inference

### 1. Activate Environment

```bash
conda activate cr-ml-app-env
```

### 2. Run Fraud Detection

```bash
# Process your CSV file
python src/production_inference.py \
    --input data/your_transactions.csv \
    --output data/predictions.csv \
    --robust-only
```

### 3. Review Results

```bash
# Check predictions
python -c "import pandas as pd; df = pd.read_csv('data/predictions.csv'); print(f'Fraud detected: {df[\"robust_model_fraud_flag\"].sum()} / {len(df)}')"
```

**See [QUICKSTART.md](QUICKSTART.md) for detailed examples.**

## 📊 Model Performance

| Model | Notebook | F1-Score | Status | Production |
|-------|----------|----------|--------|------------|
| Leaky Model | 07 | 91.78% | ⚠️ Memorization | ❌ Not recommended |
| Robust Model | 08 | 94.82% | ✅ Generalization | ✅ **RECOMMENDED** |

### Robust Model Features
- ✅ Handles unknown categories gracefully
- ✅ No data leakage (proper train/test split)
- ✅ Realistic fraud detection (0.78% fraud rate)
- ✅ Production-safe preprocessing
- ✅ High-cardinality encoding (subdistricts, cities, marketing IDs)

## Getting Started - Development

### 1. Set Up Python Environment

```bash
# Using conda (recommended)
conda create -n cr-ml-app-env python=3.11
conda activate cr-ml-app-env

# Or using venv
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook notebooks/
```

## 📚 Documentation

- **[Deployment Summary](docs/DEPLOYMENT_SUMMARY.md)** - Production deployment overview
- **[Production Guide](docs/production_deployment.md)** - Detailed deployment instructions
- **[Quick Start](QUICKSTART.md)** - Quick command reference

## 🔧 Key Notebooks

### Learning Notebooks (Iris Dataset)
- `01_iris_classification.ipynb` - Basic classification
- `02_algorithm_comparison.ipynb` - Compare ML algorithms
- `03_noise_level_assessment.ipynb` - Handle noisy data
- `04_hyperparameter_tuning.ipynb` - Optimize hyperparameters

### Fraud Detection Notebooks
- `05_fraud_data_processing.ipynb` - Data preprocessing and feature engineering
- `07_hyperparameter_tuning.ipynb` - Initial model (has data leakage)
- `08_retrain_without_leakage.ipynb` - **Production model** ✅

## Libraries Included

- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms
- **Imbalanced-learn** - SMOTE and imbalanced data handling
- **Matplotlib & Seaborn** - Data visualization
- **Joblib** - Model serialization
- **Jupyter** - Interactive notebooks

## Learning Resources

- Kaggle: https://www.kaggle.com/
- Fast.ai: https://www.fast.ai/
- Coursera: https://www.coursera.org/

## Next Steps

1. Create a new Jupyter notebook in `notebooks/`
2. Start with exploratory data analysis (EDA)
3. Build and train ML models
4. Evaluate and optimize models
