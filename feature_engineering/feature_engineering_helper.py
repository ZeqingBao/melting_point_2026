from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import seaborn as sns
import itertools



def smiles_to_features(dataframe, feature_type):

    data_with_features = dataframe.copy()
    
    # Extract RDKit descriptors
    if 'rdkit' in feature_type:
        # Get all RDKit descriptor names and functions (name, func)
        rdkit_descriptors = [func for name, func in Descriptors.descList]
        rdkit_names = [name for name, func in Descriptors.descList]
        
        # Extract features for each molecule
        rdkit_features = []
        for smiles in dataframe['SMILES']:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    features = [desc(mol) for desc in rdkit_descriptors]
                else:
                    features = [np.nan] * len(rdkit_descriptors)
            except:
                features = [np.nan] * len(rdkit_descriptors)
            rdkit_features.append(features)
        
        # Add features to dataframe with prefix to avoid column name conflicts
        for i, name in enumerate(rdkit_names):
            data_with_features[f'RDKit_{name}'] = [f[i] for f in rdkit_features]
        
        print(f"✓ RDKit: Added {len(rdkit_names)} features")
    
    # Extract MACCS Keys
    if 'maccs' in feature_type:
        maccs_features = []
        for smiles in dataframe['SMILES']:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    maccs = MACCSkeys.GenMACCSKeys(mol)
                    maccs_array = np.array(maccs)
                else:
                    maccs_array = np.array([np.nan] * 167)
            except:
                maccs_array = np.array([np.nan] * 167)
            maccs_features.append(maccs_array)
        
        # Add features to dataframe
        maccs_array = np.array(maccs_features)
        for i in range(maccs_array.shape[1]):
            data_with_features[f'MACCS_{i}'] = maccs_array[:, i]
        
        print(f"✓ MACCS: Added 167 features")
    
    return data_with_features



def standardize_features(data, all_feature_cols, scaler_path=None, fit=True):

    df_X = data[all_feature_cols].copy()

    if fit:
        # Fit a new scaler
        scaler = StandardScaler()
        df_X_scaled = scaler.fit_transform(df_X)
        
        # Save scaler if path is provided
        if scaler_path:
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"✓ Scaler saved to: {scaler_path}")
    else:
        # Load existing scaler
        if scaler_path is None:
            raise ValueError("scaler_path must be provided when fit=False")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✓ Scaler loaded from: {scaler_path}")
        df_X_scaled = scaler.transform(df_X)
    
    # Convert back to dataframe with original feature names and index
    df_X_scaled = pd.DataFrame(df_X_scaled, columns=df_X.columns, index=df_X.index)
    
    print(f"✓ Standardization complete. Shape: {df_X_scaled.shape}")
    
    return df_X_scaled


def reduce_features_by_variance(df_X, variance_threshold=0.01):
    
    original_features = df_X.shape[1]
    
    variances = df_X.var()
    
    selected_features = variances[variances >= variance_threshold].index.tolist()
    df_X_reduced = df_X[selected_features]
    
    remaining_features = df_X_reduced.shape[1]
    removed_features = original_features - remaining_features
    
    print(f"Original features: {original_features}")
    print(f"Removed features: {removed_features}")
    print(f"Remaining features: {remaining_features}")
    
    return df_X_reduced




def reduce_features_by_RFE(df_features, df_target, n_features_to_select, step=1, 
                          metric='rmse', cv_strategy=None):
    """
    Perform Recursive Feature Elimination (RFE) with XGBoost and cross-validation.
    """
    
    # Set default CV strategy
    if cv_strategy is None:
        cv_strategy = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Flatten target if needed
    y = df_target.values.ravel() if hasattr(df_target, 'values') else df_target
    
    # Set up scoring function based on metric
    if metric.lower() == 'rmse':
        scoring = 'neg_mean_squared_error'
        def score_to_metric(scores):
            return np.sqrt(-scores)
    elif metric.lower() == 'mae':
        scoring = 'neg_mean_absolute_error'
        def score_to_metric(scores):
            return -scores
    elif metric.lower() == 'r2':
        scoring = 'r2'
        def score_to_metric(scores):
            return scores
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'rmse', 'mae', or 'r2'")
    
    # Initialize results
    results = []
    
    # Start with all features
    current_features = df_features.columns.tolist()
    iteration = 0
    
    # Create base estimator (XGBoost)
    estimator = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0, n_jobs=-1)
    
    best_score = None
    best_features = None
    
    # Calculate number of iterations for progress bar
    n_iterations = 0
    temp_n = len(current_features)
    while temp_n > n_features_to_select:
        temp_n = max(n_features_to_select, temp_n - step)
        n_iterations += 1
        if temp_n == n_features_to_select:
            break
    
    # Iteratively eliminate features with progress bar
    pbar = tqdm(total=n_iterations, desc="RFE Feature Selection", unit="iteration", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {unit}')
    
    while len(current_features) >= n_features_to_select:
        # Create RFE selector to reduce to next step
        n_features_next = max(n_features_to_select, len(current_features) - step)
        
        # Subset features
        X_current = df_features[current_features]
        
        # Perform RFE
        rfe = RFE(estimator=estimator, n_features_to_select=n_features_next, step=step)
        rfe.fit(X_current, y)
        
        # Get selected features and removed features
        selected_mask = rfe.support_
        selected_features = [current_features[i] for i in range(len(current_features)) if selected_mask[i]]
        removed_features = [current_features[i] for i in range(len(current_features)) if not selected_mask[i]]
        
        # Evaluate with cross-validation
        X_selected = X_current[selected_features]
        cv_scores = cross_val_score(estimator, X_selected, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)
        cv_scores_metric = score_to_metric(cv_scores)
        
        # Track best features
        if best_score is None or (metric.lower() == 'r2' and cv_scores_metric.mean() > best_score) or \
           (metric.lower() != 'r2' and cv_scores_metric.mean() < best_score):
            best_score = cv_scores_metric.mean()
            best_features = selected_features.copy()
        
        # Format removed features for display
        removed_str = ', '.join(removed_features)
        
        # Store results
        results.append({
            'iteration': iteration,
            'n_features': len(selected_features),
            f'{metric}_mean': cv_scores_metric.mean(),
            f'{metric}_std': cv_scores_metric.std(),
            'selected_features': selected_features.copy(),
            'removed_features': removed_features.copy()
        })
        
        # Print iteration info
        if iteration % 10 == 0:
            print(f"Iteration {iteration}/{n_iterations} | Features: {len(selected_features)} | {metric.upper()}: {cv_scores_metric.mean():.4f} ± {cv_scores_metric.std():.4f} | Removed: [{removed_str}]")
        
        # Update progress bar
        pbar.update(1)
        
        # Update for next iteration
        current_features = selected_features
        iteration += 1
        
        # Stop if we've reached the target
        if len(current_features) == n_features_to_select:
            break
    
    pbar.close()
    
    # Create summary dataframe
    summary_df = pd.DataFrame(results)
    
    # Reduce features to best set
    df_best_features = df_features[best_features]
    
    print(f"\n✓ RFE Feature Selection Complete")
    print(f"  Best number of features: {len(best_features)}")
    print(f"  Best {metric.upper()}: {best_score:.4f}")
    print(f"  Best features: {best_features[:5]}{'...' if len(best_features) > 5 else ''}")
    
    return {
        'summary': summary_df,
        'best_features': best_features,
        'n_best_features': len(best_features),
        'df_best_features': df_best_features
    }


def RFE_plot(RFE_results):
    summary = RFE_results['summary'].copy()
    
    # Sort by n_features to ensure line plot is correct
    summary = summary.sort_values(by='n_features')
    
    # Identify metric columns
    metric_col = [col for col in summary.columns if '_mean' in col][0]
    metric_std_col = metric_col.replace('_mean', '_std')
    metric_label = metric_col.split('_')[0].upper()
    
    # Identify best performance
    if 'r2' in metric_col.lower():
        best_idx = summary[metric_col].idxmax()
        direction = 'max'
    else:
        best_idx = summary[metric_col].idxmin()
        direction = 'min'
        
    best_n = summary.loc[best_idx, 'n_features']
    best_score = summary.loc[best_idx, metric_col]
    
    # --- Plotting Style ---
    sns.set_theme(style="ticks", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors
    line_color = "#2c3e50"     # Dark slate
    fill_color = "#3498db"     # Bright blue
    best_point_color = "#e74c3c" # Red
    
    # 1. Uncertainty Band (Std Dev)
    if metric_std_col in summary.columns:
        ax.fill_between(
            summary['n_features'], 
            summary[metric_col] - summary[metric_std_col],
            summary[metric_col] + summary[metric_std_col],
            color=fill_color, alpha=0.2, label='±1 Std. Dev.', zorder=1
        )
    
    # 2. Mean Score Line
    ax.plot(
        summary['n_features'], summary[metric_col], 
        color=line_color, linestyle='-', linewidth=2, 
        marker='o', markersize=5, markerfacecolor='white', markeredgewidth=1.5,
        label=f'Mean {metric_label}', zorder=2
    )
    
    # 3. Highlight Best Point
    ax.scatter(
        best_n, best_score, 
        color=best_point_color, s=150, zorder=3, 
        edgecolor='white', linewidth=2, label=f'Optimal ({best_n})'
    )
    
    # 4. Vertical Dropline for Best Point (Optional, adds readability)
    ax.vlines(x=best_n, ymin=ax.get_ylim()[0], ymax=best_score, 
              colors=best_point_color, linestyles=':', alpha=0.6, zorder=0)

    # 5. Annotation
    va_align = 'bottom' if direction == 'min' else 'top'
    text_offset = (0, 15) if direction == 'min' else (0, -15)
    
    ax.annotate(
        f'{metric_label} = {best_score:.4f}',
        xy=(best_n, best_score), xytext=text_offset,
        textcoords='offset points', ha='center', va=va_align,
        fontsize=10, fontweight='bold', color=best_point_color,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=best_point_color, alpha=0.9)
    )

    # Labels & Title
    ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Performance ({metric_label})', fontsize=12, fontweight='bold')
    ax.set_title('RFE Performance vs. Feature Count', fontsize=14, pad=15)
    
    # Aesthetics
    sns.despine(trim=True, offset=5)
    ax.grid(True, which='major', linestyle='--', alpha=0.6)
    ax.legend(loc='best', frameon=True, framealpha=0.95, facecolor='white')
    
    plt.tight_layout()
    plt.show()

    print(f"  Optimal Feature Set: {best_n} features")
    print(f"  Best {metric_label}: {best_score:.4f}")



def feature_interaction(df):
    
    # Get all feature columns
    features = df.columns.tolist()
    
    # Generate combinations of 2 features
    # usage of combinations implies NO self-interaction (A X A)
    combinations = list(itertools.combinations(features, 2))
    
    print(f"Generating {len(combinations)} interaction features from {len(features)} original features...")
    
    # Collect new features in a dictionary
    new_features = {}
    for f1, f2 in combinations:
        feature_name = f"{f1} X {f2}"
        new_features[feature_name] = df[f1] * df[f2]
        
    # Create DataFrame from new features
    df_interactions = pd.DataFrame(new_features, index=df.index)
    
    # Concatenate with original dataframe
    df_final = pd.concat([df, df_interactions], axis=1)
    
    return df_final


def dataset_featurization(data_with_smiles, selected_features, path):

    smiles = data_with_smiles[['SMILES']]
    data_with_features = smiles_to_features(smiles, ['rdkit', 'maccs']).drop(columns=['SMILES'], axis=1)
    data_with_feature_interactions = feature_interaction(data_with_features)
    selected_features = data_with_feature_interactions[selected_features]
    final_dataset = pd.concat([data_with_smiles[['SMILES', 'MP', 'Type']], selected_features], axis=1)

    # Save final dataset
    final_dataset.to_parquet(f'{path}.parquet', index=False)
    print(f"{path} dataset saved.")

    return final_dataset