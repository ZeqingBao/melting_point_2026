import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpecFromSubplotSpec
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams


# settings
non_feature_cols = ['SMILES', 'MP', 'MP_label', 'Type']
figure_output_dir = "../Figures/final_figures/"
data_types = ['L', 'H', 'All']
color_scheme = {
    'L': '#4c72b0', 
    'H': '#dd8452',
    'All': '#55a868', 
    'Purple': '#8172b3',
    'Red': "#e74c3c",
    'Extra': '#4d4d4d'}


def RFE_plot(RFE_results, tolerance, data_type, ann, model_type='LGB', ax=None, color_scheme=None):
    
    summary = RFE_results[(model_type, data_type)]['summary'].copy()
    
    # Sort by n_features to ensure line plot is correct
    summary = summary.sort_values(by='n_features')
    
    # Identify metric columns
    metric_col = [col for col in summary.columns if '_mean' in col][0]
    metric_std_col = metric_col.replace('_mean', '_std')
    metric_label = metric_col.split('_')[0].upper()
    
    # Identify best performance (Parsimonious: least features within tolerance of global best)
    if 'r2' in metric_col.lower():
        global_best = summary[metric_col].max()
        threshold = global_best - abs(global_best) * tolerance
        candidates = summary[summary[metric_col] >= threshold]
        best_idx = candidates['n_features'].idxmin()
        direction = 'max'
    else:
        global_best = summary[metric_col].min()
        threshold = global_best + abs(global_best) * tolerance
        candidates = summary[summary[metric_col] <= threshold]
        best_idx = candidates['n_features'].idxmin()
        direction = 'min'
        
    best_n = summary.loc[best_idx, 'n_features']
    best_score = summary.loc[best_idx, metric_col]
    
    # --- Plotting Style ---
    sns.set_theme(style="ticks", font_scale=1.1)
    
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors from color_scheme if provided
    if color_scheme is not None:
        line_color = color_scheme.get(data_type, "#2c3e50")
        fill_color = color_scheme.get(data_type, "#2c3e50")
        best_point_color = color_scheme.get('Extra', "#e74c3c")
    else:
        line_color = "#2c3e50"
        fill_color = "#3498db"
        best_point_color = "#e74c3c"
    
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
        edgecolor='white', linewidth=2, label=f'Optimal ({best_n} features)'
    )
    
    # 4. Vertical Dropline for Best Point
    ax.vlines(x=best_n, ymin=ax.get_ylim()[0], ymax=best_score, 
              colors=best_point_color, linestyles=':', alpha=0.6, zorder=0)

    # 5. Annotation
    va_align = 'bottom' if direction == 'min' else 'top'
    text_offset = (0, 15) if direction == 'min' else (0, -15)
    
    ax.annotate(
        f'{metric_label} = {best_score:.2f}',
        xy=(best_n, best_score), xytext=text_offset,
        textcoords='offset points', ha='center', va=va_align,
        fontsize=14, color=best_point_color,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=best_point_color, alpha=1)
    )

    # Labels & Title
    ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Performance ({metric_label})', fontsize=12, fontweight='bold')
    if data_type == 'All':
        data_name = 'all data'
    if data_type == 'L':
        data_name = 'low mp data'
    if data_type == 'H':
        data_name = 'high mp data'
    ax.set_title(f'RFE: {data_name}', fontsize=18, pad=15)
    
    # Aesthetics
#    ax.set_ylim(20, 65)
    sns.despine(ax=ax, trim=True, offset=5)
    ax.grid(True, which='major', linestyle='--', alpha=0.6)
    ax.legend(loc='best', frameon=True, framealpha=1., facecolor='white', fontsize=14, title_fontsize=14)
    
    if standalone:
        plt.tight_layout()
        plt.show()

    ax.text(-0.05, 1.02, ann, transform=ax.transAxes, fontsize=14, va='top', ha='left')

    print(f"  [{data_type}] Optimal Feature Set: {best_n} features, Best {metric_label}: {best_score:.4f}")


def compute_metrics(df):
    """Return dict of (rmse, mae) for L, H, and All subsets."""
    results = {}
    for label in ['L', 'H']:
        subset = df[df['MP_label'] == label]
        if len(subset) > 0:
            rmse = np.sqrt(mean_squared_error(subset['MP'], subset['MP_pred']))
            mae  = mean_absolute_error(subset['MP'], subset['MP_pred'])
        else:
            rmse, mae = np.nan, np.nan
        results[label] = (rmse, mae)
    rmse_all = np.sqrt(mean_squared_error(df['MP'], df['MP_pred']))
    mae_all  = mean_absolute_error(df['MP'], df['MP_pred'])
    results['All'] = (rmse_all, mae_all)
    return results


def model_eval_plot(figure_2_results, metric, ax, color_scheme=None, ann=None, show_legend=False):

    metric_idx = {'RMSE': 0, 'MAE': 1}
    metric_labels = {'RMSE': 'RMSE (°C)', 'MAE': 'MAE (°C)'}

    model_keys    = ['L', 'H', 'All']
    subset_keys   = ['L', 'H', 'All']
    subset_labels = {'L': 'Low MP data', 'H': 'High MP data', 'All': 'All data '}
    model_labels  = {'L': 'Low MP\nmodel', 'H': 'High MP\nmodel', 'All': 'All data\nmodel'}

    if color_scheme is None:
        color_scheme = {'L': '#4c72b0', 'H': '#dd8452', 'All': '#55a868'}

    metrics_data = {mk: compute_metrics(figure_2_results[mk]) for mk in model_keys}

    n_groups    = len(model_keys)
    n_bars      = len(subset_keys)
    bar_width   = 0.15
    group_gap   = 0.45
    group_width = n_bars * bar_width + group_gap
    x_centers   = np.arange(n_groups) * group_width

    m_idx = metric_idx[metric]

    for b_idx, sk in enumerate(subset_keys):
        offsets = x_centers + (b_idx - (n_bars - 1) / 2) * bar_width
        values  = [metrics_data[mk][sk][m_idx] for mk in model_keys]
        color   = color_scheme.get(sk, '#333333')

        bars = ax.bar(
            offsets, values,
            width=bar_width * 0.9,
            color=color, alpha=0.85,
            label=subset_labels[sk],
            edgecolor='white', linewidth=0.8
        )
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.15,
                    f'{round(val)}',
                    ha='center', va='bottom', fontsize=12, color=color
                )

    ax.set_ylabel(metric_labels[metric], fontsize=12, fontweight='bold')
    ax.set_xticks(x_centers)
    ax.set_xticklabels([model_labels[mk] for mk in model_keys], fontsize=11)

    ax.set_ylim(0, 180)
    sns.despine(ax=ax, trim=False, offset=5)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    if show_legend:
        ax.legend(title='Evaluated on', frameon=True, framealpha=0.95,
                  facecolor='white', fontsize=10, title_fontsize=10)

    if ann is not None:
        ax.text(-0.08, 1.09, ann, transform=ax.transAxes, fontsize=14, va='top', ha='left')



_PROP_FUNCS = {
    'MP':   None,  # already in df
    'MW':   Descriptors.MolWt,
    'LogP': Descriptors.MolLogP,
    'HBD':  Descriptors.NumHDonors,
    'HBA':  Descriptors.NumHAcceptors,
    'TPSA': Descriptors.TPSA,
    'RingCount': Descriptors.RingCount
}

_LABEL_MAP = {'L': 'Low MP', 'H': 'High MP', 'All': 'All'}

def _ensure_properties(df, smiles_col, props):
    """Compute any missing RDKit properties from SMILES (returns copy if needed)."""
    missing = [p for p in props if p != 'MP' and p not in df.columns]
    if not missing:
        return df
    df = df.copy()
    mols = [Chem.MolFromSmiles(s) for s in df[smiles_col]]
    for prop in missing:
        df[prop] = [_PROP_FUNCS[prop](m) if m else float('nan') for m in mols]
    return df


def property_plot(df, prop, prop_label, ax, x_col, color_scheme,
                  show_outliers, smiles_col='SMILES', ann=None, show_legend=False):
    """
    Boxen plot of a single molecular property grouped by x_col.

    Parameters
    ----------
    prop          : column name of the property to plot
    prop_label    : y-axis label
    x_col         : column used for grouping (e.g. 'MP_label')
    color_scheme  : dict mapping group keys to hex colours
    show_outliers : bool — whether to show outlier points on the boxen plot
    show_legend   : only the last subplot needs the legend
    """
    df = _ensure_properties(df, smiles_col, [prop])

    present     = df[x_col].unique()
    group_order = [k for k in color_scheme if k in present]
    palette     = {k: color_scheme[k] for k in group_order}

    outlier_kws = {} if show_outliers else {'showfliers': False}
    sns.boxenplot(data=df, x=x_col, y=prop, order=group_order,
                  palette=palette, ax=ax, **outlier_kws)

    ax.set_ylabel(prop_label, fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_xticklabels([_LABEL_MAP.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()])
    sns.despine(ax=ax, trim=False, offset=5)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    if show_legend:
        means   = df.groupby(x_col)[prop].mean()
        handles = []
        for grp in group_order:
            if grp not in means.index:
                continue
            display = _LABEL_MAP.get(grp, grp)

            if prop == 'RingCount':
                patch = mpatches.Patch(color=color_scheme[grp],
                                    label=f'{display}  (mean: {round(means[grp],1)})')
            else:
                patch = mpatches.Patch(color=color_scheme[grp],
                        label=f'{display}  (mean: {round(means[grp])})')
            handles.append(patch)
        ax.legend(handles=handles, frameon=True, framealpha=0.95,
                  facecolor='white', fontsize=9, title_fontsize=9)

    if ann is not None:
        ax.text(-0.12, 1.16, ann, transform=ax.transAxes, fontsize=14, va='top', ha='left')



def pca_plot(df, hue_col, subplotspec, fig, color_scheme,
             non_feature_cols=non_feature_cols,
             show_marginals=True, xlim=None, ylim=None, ann=None, show_legend=False):
    """
    PCA joint KDE plot, optionally with PC1 (top) and PC2 (right) marginal KDEs.

    Parameters
    ----------
    subplotspec   : GridSpec cell (e.g. gs[0:2, 0:2]) for the PCA area
    fig           : the parent Figure
    hue_col       : column in df for colouring (e.g. 'MP_label')
    color_scheme  : dict mapping group keys to hex colours
    show_marginals: bool — toggle top/right marginal KDE panels
    """
    features_df = df.drop(columns=non_feature_cols)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)

    pca_model = PCA(n_components=2)
    pcs = pca_model.fit_transform(X_scaled)
    ev  = pca_model.explained_variance_ratio_

    df_plot = pd.DataFrame({'PC1': pcs[:, 0], 'PC2': pcs[:, 1]})
    df_plot[hue_col] = df[hue_col].values

    present     = df_plot[hue_col].unique()
    group_order = [k for k in color_scheme if k in present]

    cut    = 3
    thresh = 0.02
    levels = 12

    if show_marginals:
        # Nested 2×2: top-left=top marginal, bottom-left=joint, bottom-right=right marginal
        inner = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=subplotspec,
            width_ratios=[5, 1], height_ratios=[1, 5],
            hspace=0.05, wspace=0.05
        )
        ax_top   = fig.add_subplot(inner[0, 0])
        ax_joint = fig.add_subplot(inner[1, 0])
        ax_right = fig.add_subplot(inner[1, 1])
        fig.add_subplot(inner[0, 1]).set_visible(False)  # empty corner
    else:
        ax_joint = fig.add_subplot(subplotspec)
        ax_top   = None
        ax_right = None

    for label in group_order:
        subset = df_plot[df_plot[hue_col] == label]
        if len(subset) < 2:
            continue
        color = color_scheme[label]

        # Joint 2D KDE
        sns.kdeplot(data=subset, x='PC1', y='PC2', ax=ax_joint,
                    color=color, fill=True, alpha=0.5,
                    levels=levels, thresh=thresh, cut=cut, common_norm=False)
        sns.kdeplot(data=subset, x='PC1', y='PC2', ax=ax_joint,
                    color=color, fill=False, linewidths=1.5,
                    levels=levels, thresh=thresh, cut=cut, common_norm=False)

        if show_marginals:
            sns.kdeplot(data=subset, x='PC1', ax=ax_top,
                        color=color, fill=True, alpha=0.4, cut=cut, common_norm=False)
            sns.kdeplot(data=subset, y='PC2', ax=ax_right,
                        color=color, fill=True, alpha=0.4, cut=cut, common_norm=False)

    ax_joint.set_xlabel(f'PC1 ({ev[0]:.1%})', fontsize=12, fontweight='bold')
    ax_joint.set_ylabel(f'PC2 ({ev[1]:.1%})', fontsize=12, fontweight='bold')
    if xlim is not None:
        ax_joint.set_xlim(xlim)
    if ylim is not None:
        ax_joint.set_ylim(ylim)
    sns.despine(ax=ax_joint, trim=False, offset=5)
    ax_joint.grid(True, linestyle='--', alpha=0.4)

    if show_marginals:
        ax_top.set_xlim(ax_joint.get_xlim())
        ax_top.set_xlabel('')
        ax_top.set_ylabel('')
        ax_right.set_ylim(ax_joint.get_ylim())  # synced to joint (respects ylim)
        sns.despine(ax=ax_top, left=True, bottom=False, offset=3)
        ax_top.set_yticks([])

        ax_right.set_ylim(ax_joint.get_ylim())
        ax_right.set_ylabel('')
        ax_right.set_xlabel('')
        plt.setp(ax_right.get_yticklabels(), visible=False)
        sns.despine(ax=ax_right, left=False, bottom=True, offset=3)
        ax_right.set_xticks([])

    if show_legend:
        _label_map = {'L': 'Low MP', 'H': 'High MP', 'All': 'All'}
        handles = [mpatches.Patch(color=color_scheme[k], label=_label_map.get(k, k))
                   for k in group_order if k in color_scheme]
        ax_joint.legend(handles=handles, frameon=True, framealpha=0.95,
                        facecolor='white', fontsize=12, title_fontsize=9)

    if ann is not None:
        ann_ax = ax_top if show_marginals else ax_joint
        offset = 1.15 if show_marginals else 1.08
        ann_ax.text(-0.05, offset, ann, transform=ann_ax.transAxes, fontsize=14, va='top', ha='left')



def shap_analysis(model_L, model_H, data_L, data_H, n=10):

    def _compute(model, data):
        X        = data.drop(columns=non_feature_cols, errors='ignore')
        # Align columns to match the model's expected feature order
        expected_features = model.feature_name_
        X = X[expected_features]
        exp      = shap.TreeExplainer(model)
        sv       = exp.shap_values(X)
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:n]
        feats    = X.columns[top_idx].tolist()
        vals     = mean_abs[top_idx]
        return X, sv, feats, vals, exp

    XL, svL, featsL, valsL, expL = _compute(model_L, data_L)
    XH, svH, featsH, valsH, expH = _compute(model_H, data_H)

    setL, setH = set(featsL), set(featsH)

    # ── Colours ──────────────────────────────────────────────────────
    CL, CH           = color_scheme['L'], color_scheme['H']
    CL_edge, CH_edge = color_scheme['L'], color_scheme['H']
    C_BOTH           = color_scheme['Extra']

    def _label_color(feat, own_color):
        if feat in setL and feat in setH:
            return C_BOTH
        return own_color

    rcParams.update({
        'font.family'      : 'sans-serif',
        'font.sans-serif'  : ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size'        : 10,
        'axes.linewidth'   : 0.8,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
    })

    # Shared x limit
    xmax = max(valsL.max(), valsH.max()) * 1.12

    panel_h = max(4, n * 0.8)
    fig = plt.figure(figsize=(16, panel_h), facecolor='white', dpi=150)
    gs  = fig.add_gridspec(1, 2, wspace=0.08, top=0.88, bottom=0.12,
                           left=0.05, right=0.95)
    axL = fig.add_subplot(gs[0, 0])   # left  – model_L / data_L
    axH = fig.add_subplot(gs[0, 1])   # right – model_H / data_H

    bh = 0.62
    # y positions: most important (index 0) at the top → highest y value
    yL = np.arange(len(featsL) - 1, -1, -1)
    yH = np.arange(len(featsH) - 1, -1, -1)
    gap = xmax * 0.025   # small offset from the bar tip

    # ── Left panel – model_L, bars LEFT, labels at LEFT tip ────────────
    axL.barh(yL, valsL, height=bh, color=CL, edgecolor=CL_edge,
             linewidth=0.5, zorder=3)
    axL.set_xlim(0, xmax)
    axL.invert_xaxis()                 # bars extend leftward; x=0 is on the right
    axL.set_ylim(-0.6, len(featsL) - 0.4)
    axL.set_yticks([])                 # no y-tick labels; we annotate manually
    axL.tick_params(axis='x', labelsize=9, length=3)
    axL.set_xlabel('Low MP Model Mean SHAP value', fontsize=11, labelpad=5)
    axL.grid(axis='x', linestyle='--', linewidth=0.45, alpha=0.55, zorder=0)
    axL.spines['top'].set_visible(False)
    axL.spines['left'].set_visible(False)
    axL.spines['right'].set_linewidth(0.7)
    axL.spines['bottom'].set_linewidth(0.7)
    axL.set_title(f'Low MP Model', fontsize=12, 
                  #fontweight='bold',
                  color=CL, pad=9)

    # Feature names at the left tip
    for yi, (feat, val) in zip(yL, zip(featsL, valsL)):
        axL.text(val + gap, yi, feat,
                 ha='right', va='center', fontsize=9, 
                 color=_label_color(feat, CL), zorder=6)

    # ── Right panel – model_H, bars RIGHT, labels at RIGHT tip ──────────
    axH.barh(yH, valsH, height=bh, color=CH, edgecolor=CH_edge,
             linewidth=0.5, zorder=3)
    axH.set_xlim(0, xmax)
    axH.set_ylim(-0.6, len(featsH) - 0.4)
    axH.set_yticks([])
    axH.tick_params(axis='x', labelsize=9, length=3)
    axH.set_xlabel('Hig MP Model Mean SHAP value', fontsize=11, labelpad=5)
    axH.grid(axis='x', linestyle='--', linewidth=0.45, alpha=0.55, zorder=0)
    axH.spines['top'].set_visible(False)
    axH.spines['right'].set_visible(False)
    axH.spines['left'].set_linewidth(0.7)
    axH.spines['bottom'].set_linewidth(0.7)
    axH.set_title(f'High MP Model', fontsize=12,
                  color=CH, pad=9)

    # Feature names at the right tip
    for yi, (feat, val) in zip(yH, zip(featsH, valsH)):
        axH.text(val + gap, yi, feat,
                 ha='left', va='center', fontsize=9,
                 color=_label_color(feat, CH), zorder=6)


    plt.show()
    fig.savefig(figure_output_dir + 'figure_4_shap_comparison.png', dpi=300)


def uncertainty_plot(df, ax, color_scheme, ann=None, show_legend=False):
    """
    Boxen plot of prediction uncertainty split by MP label (L vs H).
    """
    present     = df['MP_label'].unique()
    group_order = [k for k in ['L', 'H'] if k in present]
    palette     = {k: color_scheme[k] for k in group_order}

    sns.boxenplot(data=df, x='MP_label', y='uncertainty', order=group_order,
                  palette=palette, ax=ax, showfliers=False)

    ax.set_ylabel('Uncertainty (°C)', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_xticklabels([_LABEL_MAP.get(t.get_text(), t.get_text())
                        for t in ax.get_xticklabels()])
    sns.despine(ax=ax, trim=False, offset=5)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    if show_legend:
        means   = df.groupby('MP_label')['uncertainty'].mean()
        handles = [mpatches.Patch(color=color_scheme[k],
                                  label=f'{_LABEL_MAP.get(k, k)}  (mean: {round(means[k], 1)})')
                   for k in group_order if k in means.index]
        ax.legend(handles=handles, frameon=True, framealpha=0.95,
                  facecolor='white', fontsize=9, title_fontsize=18)

    if ann is not None:
        ax.text(-0.1, 1.08, ann, transform=ax.transAxes, fontsize=14, va='top', ha='left')


def ae_quantile_plot(df, ax, color_scheme, ann=None, show_legend=False):
    """
    Bar chart of mean AE across four uncertainty quantiles (Q1–Q4).
    Q1 = lowest uncertainty (most confident), Q4 = highest.
    Annotations above each bar show mean uncertainty and mean AE.
    """
    df = df.copy()
    df['_unc_q'] = pd.qcut(df['uncertainty'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    q_order   = ['Q1', 'Q2', 'Q3', 'Q4']
    bar_color = color_scheme['All']

    means_unc = df.groupby('_unc_q', observed=True)['uncertainty'].mean()
    means_ae  = df.groupby('_unc_q', observed=True)['AE'].mean()
    x_pos     = range(len(q_order))

    bars = ax.bar(x_pos, [means_ae[q] for q in q_order],
                  color=bar_color, edgecolor='white', linewidth=0.6)

    # Two-line annotation above each bar
    for x, q, bar in zip(x_pos, q_order, bars):
        top = bar.get_height()
        ax.text(x, top + 0.4,
                f'mean unc = {round(means_unc[q], 1)}\nmean error = {round(means_ae[q], 1)}',
                ha='center', va='bottom', fontsize=9, linespacing=1.5)

    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(q_order)
    ax.set_ylabel('Mean Absolute Error (°C)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Uncertainty Quantile', fontsize=12, fontweight='bold')
    sns.despine(ax=ax, trim=False, offset=5)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    if ann is not None:
        ax.text(-0.1, 1.08, ann, transform=ax.transAxes, fontsize=14, va='top', ha='left')