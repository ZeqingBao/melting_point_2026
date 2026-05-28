# =============================================================================
# classifier_helper.py
# Helper functions and model development for LightGBM binary classifiers
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import joblib
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, RocCurveDisplay)
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler


# =============================================================================
# PLOTTING & EVALUATION HELPERS
# =============================================================================

def plot_classifier_performance(results_dict, model_type):
    trials = sorted(results_dict.keys())
    means  = np.array([results_dict[t]['mean_score'] for t in trials])
    stds   = np.array([results_dict[t]['std_score']  for t in trials])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trials, means, marker='o', color='forestgreen', label='Mean F1-Score')
    ax.fill_between(trials, means - stds, means + stds, alpha=0.2, color='forestgreen')

    best_trial = trials[np.argmax(means)]
    best_f1    = np.max(means)
    ax.scatter([best_trial], [best_f1], color='red', zorder=5,
               label=f'Best (Trial {best_trial})')
    ax.set_ylabel('F1-Score (Weighted)')
    ax.set_title(f'Classifier Optimization (F1-Weighted) — {model_type}')
    ax.legend()
    plt.show()


def plot_cv_confusion_matrix(model, X, y, model_type, cv=10):
    y_pred_cv = cross_val_predict(model, X, y, cv=cv, method='predict')
    y_prob_cv = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]

    cm = confusion_matrix(y, y_pred_cv)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=['Low MP (0)', 'High MP (1)']).plot(
        cmap='Blues', ax=ax)
    ax.set_title(f'10-Fold CV Confusion Matrix ({model_type})')
    plt.show()

    return y_pred_cv, y_prob_cv


def print_cv_metrics(y, y_pred_cv, y_prob_cv, model_type):
    print(f"--- Performance Metrics: {model_type} Strategy ---")
    report = classification_report(y, y_pred_cv,
                                   target_names=['Low MP', 'High MP'],
                                   output_dict=True)
    print(f"{'':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    print()
    for label_name in ['Low MP', 'High MP']:
        row = report[label_name]
        print(f"{label_name:>12} {row['precision']:>10.4f} {row['recall']:>10.4f} "
              f"{row['f1-score']:>10.4f} {int(row['support']):>10}")
    print()
    for avg in ['accuracy', 'macro avg', 'weighted avg']:
        row = report[avg]
        if avg == 'accuracy':
            print(f"{'accuracy':>12} {'':>10} {'':>10} {row:>10.4f} "
                  f"{int(report['macro avg']['support']):>10}")
        else:
            print(f"{avg:>12} {row['precision']:>10.4f} {row['recall']:>10.4f} "
                  f"{row['f1-score']:>10.4f} {int(row['support']):>10}")
    print(f"\nAUC-ROC: {roc_auc_score(y, y_prob_cv):.4f}")


def evaluate_test_set(model, df_test, non_features, output, model_type):
    X_test = df_test.drop(columns=non_features)
    y_test = df_test[output]

    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]

    # Classification report (4 decimal places)
    print(f"--- Final Test Results: {model_type} Strategy ---")
    report = classification_report(y_test, y_pred_test,
                                   target_names=['Low MP', 'High MP'],
                                   output_dict=True)
    print(f"{'':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    print()
    for label_name in ['Low MP', 'High MP']:
        row = report[label_name]
        print(f"{label_name:>12} {row['precision']:>10.4f} {row['recall']:>10.4f} "
              f"{row['f1-score']:>10.4f} {int(row['support']):>10}")
    print()
    for avg in ['accuracy', 'macro avg', 'weighted avg']:
        row = report[avg]
        if avg == 'accuracy':
            print(f"{'accuracy':>12} {'':>10} {'':>10} {row:>10.4f} "
                  f"{int(report['macro avg']['support']):>10}")
        else:
            print(f"{avg:>12} {row['precision']:>10.4f} {row['recall']:>10.4f} "
                  f"{row['f1-score']:>10.4f} {int(row['support']):>10}")
    print(f"\nAUC-ROC: {roc_auc_score(y_test, y_prob_test):.4f}")

    # Confusion matrix
    cm_test = confusion_matrix(y_test, y_pred_test)
    ConfusionMatrixDisplay(confusion_matrix=cm_test,
                           display_labels=['Low', 'High']).plot(cmap='Greens')
    plt.title(f'Test Set Confusion Matrix ({model_type})')
    plt.show()

    # ROC curve
    RocCurveDisplay.from_predictions(y_test, y_prob_test)
    plt.title(f'ROC Curve ({model_type})')
    plt.show()


def save_results(results, model, model_name, model_type):
    results_filename = f'model_development_results_{model_name}_{model_type}.pkl'
    model_filename   = f'best_model_{model_name}_{model_type}.joblib'
    with open(results_filename, 'wb') as f:
        pickle.dump(results, f)
    joblib.dump(model, model_filename, compress=3)
    print(f"Successfully saved trial results to {results_filename}")
    print(f"Successfully saved best model to {model_filename}")


def plot_comparison(train_results, test_results, strategies, color_scheme):
    colors  = [color_scheme['L'], color_scheme['H'],
               color_scheme['All'], color_scheme['Purple']]
    metrics = list(train_results.keys())
    x       = np.arange(len(strategies))
    width   = 0.6

    for split_label, results in [('Training (CV)', train_results), ('Test Set', test_results)]:
        fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=True)
        fig.suptitle(f'Classifier Strategy Comparison — {split_label}', fontsize=14)

        for col, metric in enumerate(metrics):
            ax   = axes[col]
            bars = ax.bar(x, results[metric], width=width, color=colors, edgecolor=colors)

            ax.set_ylim(0, 1)
            ax.set_xticks(x)
            ax.set_xticklabels(strategies, rotation=25, ha='right', fontsize=9)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.yaxis.set_tick_params(labelsize=9)
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_title(metric, fontsize=11)

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=7.5)

        legend_patches = [mpatches.Patch(color=c, label=s)
                          for c, s in zip(colors, strategies)]
        fig.legend(handles=legend_patches, loc='lower center', ncol=4,
                   bbox_to_anchor=(0.5, -0.08), fontsize=10, frameon=False)

        plt.tight_layout()
        plt.savefig(
            f'classifier_comparison_{split_label.replace(" ", "_").replace("(", "").replace(")", "")}.png',
            dpi=150, bbox_inches='tight')
        plt.show()


# =============================================================================
# MODEL DEVELOPMENT
# =============================================================================

def _build_smote_pipeline():
    """Builds the SMOTE oversampling pipeline."""
    return ImbPipeline(steps=[
        ('sampling',       SMOTE(sampling_strategy='minority', random_state=42)),
        ('classification', lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)),
    ])


def _build_undersampling_pipeline():
    """Builds the ENN + RandomUnderSampler pipeline."""
    return ImbPipeline(steps=[
        ('cleaning',       EditedNearestNeighbours(sampling_strategy='majority')),
        ('balancing',      RandomUnderSampler(sampling_strategy='majority', random_state=42)),
        ('classification', lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)),
    ])


def model_development_classifier(data, non_feature_cols, target_col, trials,
                                  search_space, use_pipeline=False,
                                  pipeline_type='smote'):
    """
    Generic model development function shared across all four strategies.

    Parameters
    ----------
    data            : pd.DataFrame — training data (original, non-resampled)
    non_feature_cols: list         — columns to exclude from features
    target_col      : str          — binary target column name
    trials          : int          — number of Bayesian optimisation trials
    search_space    : dict         — skopt search space for BayesSearchCV
    use_pipeline    : bool         — True for SMOTE/undersampling (ImbPipeline),
                                     False for unweighted/weighted (plain LGBM)
    pipeline_type   : str          — 'smote' or 'undersampling' (only used when
                                     use_pipeline=True)

    Returns
    -------
    trial_results   : dict         — mean/std F1 per trial (0 = default baseline)
    best_estimator  : estimator    — best model or pipeline found by BayesSearchCV
    """
    X = data.drop(columns=non_feature_cols)
    y = data[target_col].values

    skf   = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    folds = list(skf.split(X, y))

    # ── Trial 0: Default hyperparameters ─────────────────────────────────────
    trial_results  = {}
    fold_f1_scores = []

    if use_pipeline:
        default_estimator = (_build_smote_pipeline() if pipeline_type == 'smote'
                             else _build_undersampling_pipeline())
    else:
        default_estimator = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)

    for train_idx, val_idx in folds:
        X_train_f, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train_f, y_val = y[train_idx], y[val_idx]
        default_estimator.fit(X_train_f, y_train_f)
        preds = default_estimator.predict(X_val)
        fold_f1_scores.append(f1_score(y_val, preds, average='weighted'))

    mean_0 = float(np.mean(fold_f1_scores))
    std_0  = float(np.std(fold_f1_scores))
    trial_results[0] = {'mean_score': mean_0, 'std_score': std_0}
    print(f"Trial  0 (default) | mean F1: {mean_0:.4f} ± {std_0:.4f}")

    # ── Trials 1-N: BayesSearchCV ─────────────────────────────────────────────
    if use_pipeline:
        base_estimator = (_build_smote_pipeline() if pipeline_type == 'smote'
                          else _build_undersampling_pipeline())
    else:
        base_estimator = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)

    opt = BayesSearchCV(
        base_estimator,
        search_space,
        n_iter=trials,
        cv=folds,
        scoring='f1_weighted',
        random_state=42,
        n_jobs=1,
        refit=True,
    )
    opt.fit(X, y)

    for i in range(trials):
        mean_score = opt.cv_results_['mean_test_score'][i]
        std_score  = opt.cv_results_['std_test_score'][i]
        trial_results[i + 1] = {'mean_score': mean_score, 'std_score': std_score}
        print(f"Trial {i+1:>2d} | mean F1: {mean_score:.4f} ± {std_score:.4f}")

    return trial_results, opt.best_estimator_
