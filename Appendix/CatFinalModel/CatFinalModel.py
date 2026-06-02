import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# --- 1. Global Plotting Configurations ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams['font.size'] = 10         
plt.rcParams['axes.labelsize'] = 11    
plt.rcParams['axes.titlesize'] = 12    
plt.rcParams['xtick.labelsize'] = 9    
plt.rcParams['ytick.labelsize'] = 9    
plt.rcParams['legend.fontsize'] = 9    
plt.rcParams['lines.linewidth'] = 1.5  
plt.rcParams['axes.linewidth'] = 1.0   
plt.rcParams['xtick.direction'] = 'in' 
plt.rcParams['ytick.direction'] = 'in' 
plt.rcParams['savefig.dpi'] = 300      

# --- 2. Data Loading & Parameters Extraction ---
file_path = "data.xlsx"
meta_path = "cv_fold_parameters_and_metrics.csv"
optuna_history_path = "optuna_tuning_history.csv"

if not os.path.exists(meta_path):
    raise FileNotFoundError(f"Missing file: {meta_path}")

df = pd.read_excel(file_path)
df_meta = pd.read_csv(meta_path)

best_depth = int(df_meta['Depth'].iloc[0])
best_lr = float(df_meta['Learning_Rate'].iloc[0])
best_l2 = float(df_meta['L2_Leaf_Reg'].iloc[0])
best_iterations = int(df_meta['Optuna_Suggested_Max_Iterations'].iloc[0])

target_mean = 'Critical fragment length (µm)'
target_uncertainty = 'Uncertainty of Critical fragment length (µm)'  

X = df.drop(columns=[target_mean, target_uncertainty])
y_mean = df[target_mean].values
y_unser = df[target_uncertainty].values  
y = np.column_stack((y_mean, y_unser))
cat_features = ['IfSized', 'IfTreated']

# --- 3. Model Training & Evaluation ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

multi_params = {
    'iterations': best_iterations,
    'depth': best_depth,
    'learning_rate': best_lr,
    'l2_leaf_reg': best_l2,
    'loss_function': 'MultiRMSE',
    'random_seed': 42
}

final_multi_model = CatBoostRegressor(**multi_params)
final_multi_model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), early_stopping_rounds=200, verbose=False)

train_preds = final_multi_model.predict(X_train)
val_preds = final_multi_model.predict(X_val)
final_multi_model.save_model("final_9010_multi_model.cbm")

# --- 4. Plotting Loss Convergence ---
evals_result = final_multi_model.get_evals_result()
train_loss = evals_result['learn']['MultiRMSE']
val_loss = evals_result['validation']['MultiRMSE']
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(5.5, 4.2))
plt.plot(epochs, train_loss, color='#1f77b4', linestyle='-', linewidth=1.8, label='Training Loss')
plt.plot(epochs, val_loss, color='#d62728', linestyle='--', linewidth=1.8, label='Validation Loss')
plt.xlabel("Iteration (Trees)")
plt.ylabel("MultiRMSE Loss Value")
plt.title("Model Convergence History", fontweight='bold', pad=12)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(frameon=True, edgecolor='#cccccc', loc='upper right')
plt.tight_layout()
plt.savefig("TNR_FinalModel_Training_Convergence.pdf", bbox_inches='tight')
plt.close()

# --- 5. Plotting Performance Scatter & Error Bars ---
fig, ax = plt.subplots(figsize=(6.5, 5.8))
all_min, all_max = 150, 1200

ax.plot([all_min, all_max], [all_min, all_max], 'k--', alpha=0.8, linewidth=1.5, label='1:1 Identity Line')
x_bound = np.linspace(all_min, all_max, 500)
ax.plot(x_bound, x_bound * 1.10, color='#a6c8e0', linestyle=':', linewidth=1.2, label='±10% Deviation Bounds')
ax.plot(x_bound, x_bound * 0.90, color='#a6c8e0', linestyle=':', linewidth=1.2)

ax.errorbar(
    y_train[:, 0], train_preds[:, 0], yerr=train_preds[:, 1], 
    fmt='o', color='#0f5a9e', elinewidth=0.5, ecolor='#bcd2ee', capsize=0,
    alpha=0.6, ms=5, mec='w', mew=0.4, label='Train (80%)'
)

ax.errorbar(
    y_val[:, 0], val_preds[:, 0], yerr=val_preds[:, 1], 
    fmt='o', color='#d61c8c', elinewidth=1.4, ecolor='#222222', capsize=2.5,
    alpha=0.95, ms=7.5, mec='black', mew=0.8, zorder=15, label='Validation (20%)'
)

r2_lc = r2_score(y_val[:, 0], val_preds[:, 0])
rmse_lc = np.sqrt(mean_squared_error(y_val[:, 0], val_preds[:, 0]))
mae_lc = mean_absolute_error(y_val[:, 0], val_preds[:, 0])
mape_lc = mean_absolute_percentage_error(y_val[:, 0], val_preds[:, 0]) * 100

metric_text = f"$R^2$ = {r2_lc:.3f}\nRMSE = {rmse_lc:.2f} $\\mu$m\nMAE = {mae_lc:.2f} $\\mu$m\nMAPE = {mape_lc:.2f}%"
ax.text(0.05, 0.68, metric_text, transform=ax.transAxes, bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='#cccccc', alpha=0.9), fontsize=9.5)

ax.set_xlabel("Experimental $l_c$ ($\\mu$m)")
ax.set_ylabel("Predicted $l_c$ ($\\mu$m)")
ax.set_title("Final CatBoost Model Prediction Performance", fontweight='bold', pad=12)
ax.set_xlim([all_min - 20, all_max + 20])
ax.set_ylim([all_min - 20, all_max + 20])
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(frameon=True, edgecolor='#cccccc', loc='lower right')
plt.tight_layout()
plt.savefig("TNR_CatBoost_Prediction_WithErrorbar_Bounds.pdf", bbox_inches='tight')
plt.close()

# --- 6. SHAP Interpretability Computations ---
shap_params = {
    'iterations': 1000, 'depth': best_depth, 'learning_rate': best_lr,
    'l2_leaf_reg': best_l2, 'loss_function': 'RMSE', 'random_seed': 42, 'verbose': 0
}

model_for_shap_lc = CatBoostRegressor(**shap_params).fit(X_train, y_train[:, 0], cat_features=cat_features)
explainer_lc = shap.TreeExplainer(model_for_shap_lc)
shap_values_lc = explainer_lc(X_train, check_additivity=False)

model_for_shap_sigma = CatBoostRegressor(**shap_params).fit(X_train, y_train[:, 1], cat_features=cat_features)
explainer_sigma = shap.TreeExplainer(model_for_shap_sigma)
shap_values_sigma = explainer_sigma(X_train, check_additivity=False)

# --- 7. Plotting SHAP Metrics for Target (lc) ---
plt.figure(figsize=(6, 4))
shap.plots.bar(shap_values_lc, show=False)
plt.title("SHAP Feature Importance Bar Plot ($l_c$)", fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig("TNR_SHAP_lc_Bar.pdf", bbox_inches='tight')
plt.close()

plt.figure(figsize=(6, 4.5))
shap.plots.beeswarm(shap_values_lc, show=False)
plt.title("SHAP Beeswarm Plot for Main Target ($l_c$)", fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig("TNR_SHAP_lc_Beeswarm.pdf", bbox_inches='tight')
plt.close()

top_6_features_lc = [shap_values_lc.feature_names[i] for i in np.argsort(np.abs(shap_values_lc.values).mean(0))[::-1][:6]]
fig_dep_lc, axes_dep_lc = plt.subplots(2, 3, figsize=(14, 8))
axes_flat_lc = axes_dep_lc.flatten()
for idx, feature_name in enumerate(top_6_features_lc):
    ax = axes_flat_lc[idx]
    shap.plots.scatter(shap_values_lc[:, feature_name], color=shap_values_lc[:, feature_name], ax=ax, show=False)
    ax.set_title(f"({chr(97 + idx)}) {feature_name}", fontsize=10, fontweight='bold', loc='left')
    ax.grid(True, linestyle=':', alpha=0.5)
for j in range(len(top_6_features_lc), 6): fig_dep_lc.delaxes(axes_flat_lc[j])
plt.suptitle("Combined SHAP Dependence Plots for Top 6 Features ($l_c$)", fontsize=12, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("TNR_SHAP_lc_Combined_Dependence_Top6.pdf", bbox_inches='tight')
plt.close()

# --- 8. Plotting SHAP Metrics for Uncertainty (sigma) ---
plt.figure(figsize=(6, 4))
shap.plots.bar(shap_values_sigma, show=False)
plt.title("SHAP Feature Importance Bar Plot ($\sigma_{pred}$)", fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig("TNR_SHAP_Uncertainty_Bar.pdf", bbox_inches='tight')
plt.close()

plt.figure(figsize=(6, 4.5))
shap.plots.beeswarm(shap_values_sigma, show=False)
plt.title("SHAP Beeswarm Plot for Predicted Uncertainty ($\sigma_{pred}$)", fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig("TNR_SHAP_Uncertainty_Beeswarm.pdf", bbox_inches='tight')
plt.close()

top_6_features_sigma = [shap_values_sigma.feature_names[i] for i in np.argsort(np.abs(shap_values_sigma.values).mean(0))[::-1][:6]]
fig_dep_sig, axes_dep_sig = plt.subplots(2, 3, figsize=(14, 8))
axes_flat_sig = axes_dep_sig.flatten()
for idx, feature_name in enumerate(top_6_features_sigma):
    ax = axes_flat_sig[idx]
    shap.plots.scatter(shap_values_sigma[:, feature_name], color=shap_values_sigma[:, feature_name], ax=ax, show=False)
    ax.set_title(f"({chr(97 + idx)}) {feature_name}", fontsize=10, fontweight='bold', loc='left')
    ax.grid(True, linestyle=':', alpha=0.5)
for j in range(len(top_6_features_sigma), 6): fig_dep_sig.delaxes(axes_flat_sig[j])
plt.suptitle("Combined SHAP Dependence Plots for Top 6 Features ($\sigma_{pred}$)", fontsize=12, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("TNR_SHAP_Uncertainty_Combined_Dependence_Top6.pdf", bbox_inches='tight')
plt.close()

# --- 9. Plotting Optuna Slice History ---
if os.path.exists(optuna_history_path):
    df_history = pd.read_csv(optuna_history_path)
    df_valid = df_history[df_history['state'] == 'COMPLETE'].copy()
    param_cols = {
        'params_iterations': 'Iterations', 'params_depth': 'Tree Depth',
        'params_learning_rate': 'Learning Rate', 'params_l2_leaf_reg': 'L2 Leaf Reg'
    }
    available_params = {k: v for k, v in param_cols.items() if k in df_valid.columns}
    
    if available_params:
        fig_opt, axes_opt = plt.subplots(2, 2, figsize=(11, 8.5))
        axes_opt_flat = axes_opt.flatten()
        best_idx = df_valid['value'].idxmin()
        best_val = df_valid.loc[best_idx, 'value']
        
        for idx, (p_col, p_name) in enumerate(available_params.items()):
            ax = axes_opt_flat[idx]
            ax.scatter(df_valid[p_col].values, df_valid['value'].values, color='#1f77b4', alpha=0.6, edgecolors='none', s=35, label='Evaluated Trial')
            ax.scatter(df_valid.loc[best_idx, p_col], best_val, color='#d62728', marker='*', s=130, edgecolor='black', linewidths=0.8, label='Best Parameter', zorder=5)
            if 'learning_rate' in p_col: ax.set_xscale('log')
            ax.set_xlabel(p_name, fontsize=10)
            ax.set_ylabel("Multi-Task Combined Loss", fontsize=10)
            ax.set_title(f"({chr(97 + idx)}) Objective Value vs. {p_name}", fontsize=10, fontweight='bold', loc='left')
            ax.grid(True, linestyle=':', alpha=0.5)
            if idx == 0: ax.legend(frameon=True, edgecolor='#cccccc', loc='upper right')
            
        for j in range(len(available_params), 4): fig_opt.delaxes(axes_opt_flat[j])
        plt.suptitle("Optuna Hyperparameter Response Slice Plots (Multi-Task Loss)", fontsize=12, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig("TNR_Optuna_Hyperparameter_Slice_Plots.pdf", bbox_inches='tight')
        plt.close()

# --- 10. Clean Metric Logging Outputs ---
print("\n" + "="*65)
print(" EXTRACTED BEST HYPERPARAMETERS")
print("-"*65)
print(f"  Iterations : {best_iterations:<10} |  Tree Depth  : {best_depth}")
print(f"  Learn Rate : {best_lr:<10} |  L2 Leaf Reg : {best_l2}")
print("="*65)

print("\n" + "="*65)
print("FINAL MULTI-TASK MODEL METRICS (10% HOLDOUT VALIDATION)")
print("-"*65)
print(f"  R² (Mean) : {r2_lc:.4f}")
print(f"  RMSE      : {rmse_lc:.2f} μm")
print(f"  MAE       : {mae_lc:.2f} μm")
print(f"  MAPE      : {mape_lc:.2f}%")
print("="*65)