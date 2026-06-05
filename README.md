# Melting Point Prediction

Machine learning models for predicting small-molecule melting points from molecular descriptors computed with RDKit. The pipeline covers data processing, feature engineering, model development (LightGBM, Random Forest, Neural Network), and evaluation.

## Repository Structure

| Folder | Description |
|---|---|
| `0_data/` | Raw and processed datasets; data augmentation |
| `1_feature_engineering/` | RDKit descriptor generation and feature selection |
| `2_model_development/` | LightGBM model training (All / High / Low subsets) |
| `3_model_evaluation/` | Test-set evaluation and result CSVs |
| `4_feature_analysis/` | SHAP-based feature importance |
| `5_similarity_analysis/` | Morgan fingerprint and Vendi score diversity analysis |
| `6_EDA/` | Exploratory data analysis |
| `7_classifier/` | Binary high/low melting point classifiers |
| `8_other_models/` | Random Forest and Neural Network alternatives |
| `9_figures/` | Publication figures |



## Models

Four LightGBM regression models are trained on different data splits:

- `best_model_LGB_All` — full dataset
- `best_model_LGB_H` — high melting point subset
- `best_model_LGB_L` — low melting point subset
- `best_model_LGB_L_undersample` — low subset with undersampling

## Setup

```bash
conda env create -f environment.yml
conda activate melting_point
```

Key dependencies: `rdkit`, `lightgbm`, `scikit-learn`, `pytorch`, `optuna`, `imbalanced-learn`.

