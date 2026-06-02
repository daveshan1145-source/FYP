# 1. Environment & Plotting Configurations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, kendalltau 

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams['font.size'] = 13          
plt.rcParams['axes.labelsize'] = 16     
plt.rcParams['xtick.labelsize'] = 13    
plt.rcParams['ytick.labelsize'] = 13    
plt.rcParams['legend.fontsize'] = 11    
plt.rcParams['lines.linewidth'] = 2.2   
plt.rcParams['axes.linewidth'] = 1.8   
plt.rcParams['xtick.direction'] = 'in'  
plt.rcParams['ytick.direction'] = 'in'  
plt.rcParams['xtick.major.size'] = 6    
plt.rcParams['ytick.major.size'] = 6    
plt.rcParams['xtick.major.width'] = 1.5   
plt.rcParams['ytick.major.width'] = 1.5   
plt.rcParams['savefig.dpi'] = 300      

UNIFIED_COLOR = '#1f77b4' 
EPSILON = 1e-8

# 2. Path Settings & Helper Functions
model_configs = {
    'Random Forest': {
        'path': os.path.join('..', 'RandonForest', 'rf_cv_predictions_pixel_level.csv'),
        'color': '#2ca02c',  
        'marker': 's'        
    },
    'CatBoost': {
        'path': os.path.join('..', 'CatFinal', 'cv_predictions_pixel_level.csv'),
        'color': '#ff7f0e',  
        'marker': '^'        
    },
    'SVR': {
        'path': os.path.join('..', 'SVMFInal', 'svm_cv_predictions_pixel_level.csv'),
        'color': '#d62728',  
        'marker': 'd'        
    },
    'ANNs': {
        'path': os.path.join('..', 'ANNfinal', 'ann_multitask_predictions.csv'),
        'color': '#1f77b4',  
        'marker': 'o'        
    }
}

def calculate_metrics(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        r2, rho, tau = 0.0, 0.0, 0.0
    else:
        r2 = r2_score(y_true, y_pred)
        rho, _ = spearmanr(y_true, y_pred)
        tau, _ = kendalltau(y_true, y_pred)
        if np.isnan(rho): rho = 0.0
        if np.isnan(tau): tau = 0.0
        
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + EPSILON)) * 100
    return r2, rmse, mae, mape, rho, tau

def plot_custom_scatter(x_data, y_data, x_label, y_label, save_name, r2_val, spearman_val, kendall_val, ax_min, ax_max, has_errorbar=False, y_err=None):
    fig, ax = plt.subplots(figsize=(5.5, 5.5)) 
    
    if has_errorbar and y_err is not None:
        ax.errorbar(
            x_data, y_data, yerr=y_err,
            fmt='o', color=UNIFIED_COLOR, ecolor='#d62728',
            elinewidth=1.2, capsize=2.5, markersize=6.0, alpha=0.8,
            markeredgewidth=0.8, markeredgecolor='white',
            label='Predicted $l_c$ $\pm$ $\sigma_{pred}$'
        )
    else:
        ax.scatter(
            x_data, y_data, color=UNIFIED_COLOR, edgecolors='white',
            s=45, alpha=0.8, linewidths=0.8, label='Noise Alignment'
        )
    
    x_line = np.linspace(ax_min, ax_max, 100)
    ax.plot(x_line, x_line, color='#222222', linestyle='-', linewidth=2.0, label="Ideal line ($y=x$)")
    ax.plot(x_line, x_line * 1.10, color='#666666', linestyle='--', linewidth=1.2, label="$\pm$10% Error bounds")
    ax.plot(x_line, x_line * 0.90, color='#666666', linestyle='--', linewidth=1.2)
    
    ax.set_xlabel(x_label, labelpad=8)
    ax.set_ylabel(y_label, labelpad=8)
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)
    ax.set_aspect('equal', adjustable='box')
    
    leg = ax.legend(frameon=True, loc='upper left', bbox_to_anchor=(0.01, 0.99))
    leg.get_frame().set_edgecolor('#999999')
    leg.get_frame().set_linewidth(1.0)
    ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.8)
    
    stats_str = f"$R^2$: {r2_val:.4f}\n$\\rho_{{Spearman}}$: {spearman_val*100:.1f}%\n$\\tau_{{Kendall}}$: {kendall_val*100:.1f}%"
    ax.text(0.95, 0.05, stats_str, transform=ax.transAxes, fontsize=10, fontname='Times New Roman', 
            verticalalignment='bottom', horizontalalignment='right', 
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#999999', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

# 3. Axis Initialization & Data Verification
lc_global_min, lc_global_max = float('inf'), float('-inf')
sd_global_min, sd_global_max = float('inf'), float('-inf')
loaded_datasets = {}

for model_name, config in model_configs.items():
    file_path = config['path']
    if not os.path.exists(file_path):
        continue
    
    df = pd.read_csv(file_path)
    local_lc_min = min(df['True_lc'].min(), df['Predicted_lc'].min())
    local_lc_max = max(df['True_lc'].max(), df['Predicted_lc'].max())
    if local_lc_min < lc_global_min: lc_global_min = local_lc_min
    if local_lc_max > lc_global_max: lc_global_max = local_lc_max
    
    if 'True_Uncertainty' in df.columns and 'Predicted_sigma' in df.columns:
        local_sd_min = min(df['True_Uncertainty'].min(), df['Predicted_sigma'].min())
        local_sd_max = max(df['True_Uncertainty'].max(), df['Predicted_sigma'].max())
        if local_sd_min < sd_global_min: sd_global_min = local_sd_min
        if local_sd_max > sd_global_max: sd_global_max = local_sd_max
    
    loaded_datasets[model_name] = df

AXIS_LC_MIN = lc_global_min * 0.92
AXIS_LC_MAX = lc_global_max * 1.08
AXIS_SD_MIN = sd_global_min * 0.92 if sd_global_min != float('inf') else 0.0
AXIS_SD_MAX = sd_global_max * 1.08 if sd_global_max != float('-inf') else 10.0

drop_counts = np.arange(0, 6)
sensitivity_matrix = {model: [] for model in loaded_datasets.keys()}

# 4. Multi-Model Pipeline and Outlier Analysis
records_summary = []
worst_fold_records = []
all_model_outliers = [] 

for model_name, df in loaded_datasets.items():
    unique_folds = df['Fold_Group'].unique() if 'Fold_Group' in df.columns else ['Fold_Unknown']
    worst_fold_name = None
    worst_fold_mape = -1.0
    worst_fold_metrics = (0, 0, 0, 0, 0, 0)
    
    for f_name in unique_folds:
        fold_df = df[df['Fold_Group'] == f_name]
        f_true = fold_df['True_lc'].values
        f_pred = fold_df['Predicted_lc'].values
        if len(f_true) == 0: continue
            
        f_r2, f_rmse, f_mae, f_mape, f_rho, f_tau = calculate_metrics(f_true, f_pred)
        if f_mape > worst_fold_mape:
            worst_fold_mape = f_mape
            worst_fold_name = f_name
            worst_fold_metrics = (f_r2, f_rmse, f_mae, f_mape, f_rho, f_tau)
            
    worst_fold_records.append({
        'Model': model_name, 'Worst_Fold': worst_fold_name,
        'R2': worst_fold_metrics[0], 'RMSE': worst_fold_metrics[1], 'MAE': worst_fold_metrics[2], 
        'MAPE(%)': worst_fold_metrics[3], 'Spearman(%)': worst_fold_metrics[4] * 100, 'Kendall(%)': worst_fold_metrics[5] * 100
    })
    
    y_true_lc = df['True_lc'].values
    y_pred_lc = df['Predicted_lc'].values
    has_sd = 'True_Uncertainty' in df.columns and 'Predicted_sigma' in df.columns
    if has_sd:
        y_true_sd = df['True_Uncertainty'].values
        y_pred_sd = df['Predicted_sigma'].values
    
    ape = np.abs(y_true_lc - y_pred_lc) / (np.abs(y_true_lc) + EPSILON)
    
    r2_lc_all, rmse_lc_all, mae_lc_all, mape_lc_all, rho_lc_all, tau_lc_all = calculate_metrics(y_true_lc, y_pred_lc)
    save_lc_all = f"TNR_{model_name.replace(' ', '_')}_All_Samples_Prediction.pdf"
    y_err_in = y_pred_sd if has_sd else None
    plot_custom_scatter(y_true_lc, y_pred_lc, "Experimental $l_c$ ($\mu$m)", "Predicted $l_c$ ($\mu$m)", 
                        save_lc_all, r2_lc_all, rho_lc_all, tau_lc_all, AXIS_LC_MIN, AXIS_LC_MAX, has_errorbar=has_sd, y_err=y_err_in)
    
    records_summary.append({
        'Model': model_name, 'Data_Scope': 'All Samples', 'Metric_Target': 'lc_Property', 'Count': len(y_true_lc), 
        'R2': r2_lc_all, 'RMSE': rmse_lc_all, 'MAE': mae_lc_all, 'MAPE(%)': mape_lc_all, 'Spearman(%)': rho_lc_all * 100, 'Kendall(%)': tau_lc_all * 100
    })
    
    if has_sd:
        r2_sd_all, rmse_sd_all, mae_sd_all, mape_sd_all, rho_sd_all, tau_sd_all = calculate_metrics(y_true_sd, y_pred_sd)
        save_sd_all = f"TNR_{model_name.replace(' ', '_')}_All_Samples_Noise_Evaluation.pdf"
        plot_custom_scatter(y_true_sd, y_pred_sd, "Experimental Uncertainty ($\mu$m)", "Predicted Sigma ($\mu$m)", 
                            save_sd_all, r2_sd_all, rho_sd_all, tau_sd_all, AXIS_SD_MIN, AXIS_SD_MAX, has_errorbar=False)
        records_summary.append({
            'Model': model_name, 'Data_Scope': 'All Samples', 'Metric_Target': 'Uncertainty_Noise', 'Count': len(y_true_sd), 
            'R2': r2_sd_all, 'RMSE': rmse_sd_all, 'MAE': mae_sd_all, 'MAPE(%)': mape_sd_all, 'Spearman(%)': rho_sd_all * 100, 'Kendall(%)': tau_sd_all * 100
        })
    
    threshold_val = np.percentile(ape, 90)
    filtered_mask = ape <= threshold_val
    outlier_mask = ~filtered_mask
    
    df_outliers = df[outlier_mask].copy()
    df_outliers['Calculated_APE(%)'] = ape[outlier_mask] * 100
    df_outliers['Associated_Model'] = model_name
    
    outlier_cols = ['Associated_Model', 'True_lc', 'Predicted_lc', 'Calculated_APE(%)']
    if 'Original_Index' in df.columns: outlier_cols.insert(1, 'Original_Index')
    if 'Fold_Group' in df.columns: outlier_cols.insert(2, 'Fold_Group')
    if has_sd: 
        outlier_cols.extend(['True_Uncertainty', 'Predicted_sigma'])
    
    all_model_outliers.append(df_outliers[outlier_cols])
    
    y_true_lc_f = y_true_lc[filtered_mask]
    y_pred_lc_f = y_pred_lc[filtered_mask]
    r2_lc_filt, rmse_lc_filt, mae_lc_filt, mape_lc_filt, rho_lc_filt, tau_lc_filt = calculate_metrics(y_true_lc_f, y_pred_lc_f)
    save_lc_filt = f"TNR_{model_name.replace(' ', '_')}_Filtered_Top10_Prediction.pdf"
    y_err_f = y_pred_sd[filtered_mask] if has_sd else None
    plot_custom_scatter(y_true_lc_f, y_pred_lc_f, "Experimental $l_c$ ($\mu$m)", "Predicted $l_c$ ($\mu$m)", 
                        save_lc_filt, r2_lc_filt, rho_lc_filt, tau_lc_filt, AXIS_LC_MIN, AXIS_LC_MAX, has_errorbar=has_sd, y_err=y_err_f)
    
    records_summary.append({
        'Model': model_name, 'Data_Scope': 'Top 90% Filtered', 'Metric_Target': 'lc_Property', 'Count': len(y_true_lc_f), 
        'R2': r2_lc_filt, 'RMSE': rmse_lc_filt, 'MAE': mae_lc_filt, 'MAPE(%)': mape_lc_filt, 'Spearman(%)': rho_lc_filt * 100, 'Kendall(%)': tau_lc_filt * 100
    })
    
    if has_sd:
        y_true_sd_f = y_true_sd[filtered_mask]
        y_pred_sd_f = y_pred_sd[filtered_mask]
        r2_sd_filt, rmse_sd_filt, mae_sd_filt, mape_sd_filt, rho_sd_filt, tau_sd_filt = calculate_metrics(y_true_sd_f, y_pred_sd_f)
        save_sd_filt = f"TNR_{model_name.replace(' ', '_')}_Filtered_Top10_Noise_Evaluation.pdf"
        plot_custom_scatter(y_true_sd_f, y_pred_sd_f, "Experimental Uncertainty ($\mu$m)", "Predicted Sigma ($\mu$m)", 
                            save_sd_filt, r2_sd_filt, rho_sd_filt, tau_sd_filt, AXIS_SD_MIN, AXIS_SD_MAX, has_errorbar=False)
        records_summary.append({
            'Model': model_name, 'Data_Scope': 'Top 90% Filtered', 'Metric_Target': 'Uncertainty_Noise', 'Count': len(y_true_sd_f), 
            'R2': r2_sd_filt, 'RMSE': rmse_sd_filt, 'MAE': mae_sd_filt, 'MAPE(%)': mape_sd_filt, 'Spearman(%)': rho_sd_filt * 100, 'Kendall(%)': tau_sd_filt * 100
        })

    temp_df = pd.DataFrame({'true': y_true_lc, 'pred': y_pred_lc, 'ape': ape})
    temp_df_sorted = temp_df.sort_values(by='ape', ascending=False).reset_index(drop=True)
    
    for k in drop_counts:
        filtered_subset = temp_df_sorted.iloc[k:]
        t_subs = filtered_subset['true'].values
        p_subs = filtered_subset['pred'].values
        _, _, _, mape_subs, _, _ = calculate_metrics(t_subs, p_subs)
        sensitivity_matrix[model_name].append(mape_subs)

# 5. Sensitivity Analysis Plotting
plt.rcParams['axes.labelsize'] = 15     
fig, ax = plt.subplots(figsize=(6.5, 5.2))

for model_name, config in model_configs.items():
    if model_name in sensitivity_matrix and len(sensitivity_matrix[model_name]) > 0:
        ax.plot(
            drop_counts, 
            sensitivity_matrix[model_name], 
            label=model_name,
            color=config['color'],
            marker=config['marker'],
            markersize=7.5,
            linewidth=2.0,
            markeredgecolor='white',
            markeredgewidth=0.8
        )

ax.set_xlabel('Number of Removed Anomalous Samples ($APE_{top\_k}$)', labelpad=10)
ax.set_ylabel('Subset Performance Metric: MAPE (%)', labelpad=10)
ax.set_xticks(drop_counts)
ax.set_xlim(-0.3, 5.3)  
ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.8)

leg = ax.legend(frameon=True, loc='upper right')
leg.get_frame().set_edgecolor('#cccccc')
leg.get_frame().set_linewidth(0.8)

plt.tight_layout()
plt.savefig("TNR_Models_Outlier_Removal_Sensitivity_Analysis.pdf", bbox_inches='tight')
plt.close()

# 6. Structured Diagnostics Exports
df_summary = pd.DataFrame(records_summary)
df_worst = pd.DataFrame(worst_fold_records)
df_all_outliers = pd.concat(all_model_outliers, ignore_index=True)

df_summary.to_csv("Model_Property_and_Noise_Performance.csv", index=False)
df_worst.to_csv("Model_Worst_Fold_Metrics.csv", index=False)
df_all_outliers.to_csv("Model_Filtered_Outliers_Diagnostic.csv", index=False)

print("\n=== Pipeline Execution Completed Successfully ===")