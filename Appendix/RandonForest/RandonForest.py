import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

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

# --- 2. Data Loading & Feature Engineering ---
file_path = "data.xlsx"
df = pd.read_excel(file_path)

target_mean = 'Critical fragment length (µm)'
target_uncertainty = 'Uncertainty of Critical fragment length (µm)'  

X_raw = df.drop(columns=[target_mean, target_uncertainty])
y_mean = df[target_mean].values
y_unser = df[target_uncertainty].values  

X = pd.get_dummies(X_raw, drop_first=True)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# --- 3. Hyperparameter Optimization via Optuna ---
def objective_mean(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'max_depth': trial.suggest_int('max_depth', 4, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42,
        'n_jobs': -1
    }
    rmses = []
    for tr, va in kf.split(X):
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(**params).fit(X.iloc[tr], y_mean[tr])
        rmses.append(np.sqrt(mean_squared_error(y_mean[va], model.predict(X.iloc[va]))))
    return np.mean(rmses)

study_mean = optuna.create_study(direction='minimize')
study_mean.optimize(objective_mean, n_trials=100, n_jobs=20)
best_params_mean = study_mean.best_params
trials_mean_df = study_mean.trials_dataframe()
trials_mean_df.to_csv("rf_agent_A_tuning_history.csv", index=False)

def objective_unser(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'max_depth': trial.suggest_int('max_depth', 4, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42,
        'n_jobs': -1
    }
    rmses = []
    for tr, va in kf.split(X):
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(**params).fit(X.iloc[tr], y_unser[tr])
        rmses.append(np.sqrt(mean_squared_error(y_unser[va], model.predict(X.iloc[va]))))
    return np.mean(rmses)

study_unser = optuna.create_study(direction='minimize')
study_unser.optimize(objective_unser, n_trials=50)
best_params_unser = study_unser.best_params
trials_unser_df = study_unser.trials_dataframe()
trials_unser_df.to_csv("rf_agent_B_tuning_history.csv", index=False)

# --- 4. 10-Fold Cross-Validation Evaluation ---
all_true_mean, all_pred_mean, all_pred_std = [], [], []
cv_records = []
fold_metrics_summary = [] 

agent_A_params = best_params_mean.copy()
agent_A_params.update({'random_state': 42, 'n_jobs': -1})

agent_B_params = best_params_unser.copy()
agent_B_params.update({'random_state': 42, 'n_jobs': -1})

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    from sklearn.ensemble import RandomForestRegressor
    
    model_mean = RandomForestRegressor(**agent_A_params).fit(X_train, y_mean[train_idx])
    pred_mean = model_mean.predict(X_test)
    
    model_unser = RandomForestRegressor(**agent_B_params).fit(X_train, y_unser[train_idx])
    pred_std = np.maximum(model_unser.predict(X_test), 0)

    fold_r2 = r2_score(y_mean[test_idx], pred_mean)
    fold_rmse = np.sqrt(mean_squared_error(y_mean[test_idx], pred_mean))
    fold_mae = mean_absolute_error(y_mean[test_idx], pred_mean)
    fold_mape = np.mean(np.abs(y_mean[test_idx] - pred_mean) / (np.abs(y_mean[test_idx]) + 1e-8)) * 100

    all_true_mean.extend(y_mean[test_idx])
    all_pred_mean.extend(pred_mean)
    all_pred_std.extend(pred_std)

    fold_metrics_summary.append({
        'Fold': f"Fold_{fold}",
        'AgentA_n_estimators': agent_A_params['n_estimators'],
        'AgentA_max_depth': agent_A_params['max_depth'],
        'AgentB_n_estimators': agent_B_params['n_estimators'],
        'AgentB_max_depth': agent_B_params['max_depth'],
        'Mean_Fold_R2': fold_r2,
        'Mean_Fold_RMSE': fold_rmse,
        'Mean_Fold_MAE': fold_mae,
        'Mean_Fold_MAPE(%)': fold_mape
    })

    for i, idx in enumerate(test_idx):
        cv_records.append({
            'Original_Index': idx,
            'Fold_Group': f"Fold_{fold}",
            'True_lc': y_mean[idx],
            'True_Uncertainty': y_unser[idx],
            'Predicted_lc': pred_mean[i],
            'Predicted_sigma': pred_std[i]
        })

pd.DataFrame(fold_metrics_summary).to_csv("rf_dual_agent_fold_metrics.csv", index=False)
pd.DataFrame(cv_records).to_csv("rf_cv_predictions_pixel_level.csv", index=False)

# --- 5. Final Full-Dataset Training & Model Export ---
from sklearn.ensemble import RandomForestRegressor
final_model_mean = RandomForestRegressor(**agent_A_params).fit(X, y_mean)
final_model_unser = RandomForestRegressor(**agent_B_params).fit(X, y_unser)

joblib.dump(final_model_mean, "final_rf_model_mean.pkl")
joblib.dump(final_model_unser, "final_rf_model_unser.pkl")

# --- 6. Plotting Cross-Validation Predictions ---
fig_pred, ax_pred = plt.subplots(figsize=(5, 5))
ax_pred.errorbar(
    all_true_mean, all_pred_mean, yerr=all_pred_std,
    fmt='o', color='#1f77b4', ecolor='#d62728',
    elinewidth=0.7, capsize=1.5, markersize=4.5, alpha=0.75,
    label='Predicted $l_c$ $\pm$ $\sigma_{pred\_experimental}$'
)

mn = min(min(all_true_mean), min(all_pred_mean)) * 0.92
mx = max(max(all_true_mean), max(all_pred_mean)) * 1.08
x_line = np.linspace(mn, mx, 100)

ax_pred.plot(x_line, x_line, color='#333333', linestyle='-', linewidth=1.2, label="Ideal line ($y=x$)")
ax_pred.plot(x_line, x_line * 1.10, color='#7f7f7f', linestyle=':', linewidth=1.0, label="$\pm$10% Error bounds")
ax_pred.plot(x_line, x_line * 0.90, color='#7f7f7f', linestyle=':', linewidth=1.0)

ax_pred.set_xlabel("Experimental $l_c$ ($\mu$m)")
ax_pred.set_ylabel("Predicted $l_c$ ($\mu$m)")
ax_pred.set_title("10-Fold Cross-Validation: RF Dual-Agent Prediction", fontsize=11, fontweight='bold')
ax_pred.set_xlim(mn, mx)
ax_pred.set_ylim(mn, mx)
ax_pred.set_aspect('equal', adjustable='box')
ax_pred.legend(frameon=True, edgecolor='#cccccc', loc='upper left')
ax_pred.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("TNR_RF_DualTask_Prediction.pdf", bbox_inches='tight')
plt.close()

# --- 7. Plotting SHAP Explanations for Agent A ---
explainer_mean = shap.TreeExplainer(final_model_mean)
shap_values_mean = explainer_mean(X)

plt.figure()
shap.plots.beeswarm(shap_values_mean, show=False)
ax_m = plt.gca()
for label in (ax_m.get_xticklabels() + ax_m.get_yticklabels()):
    label.set_fontname('Times New Roman')
ax_m.set_xlabel("SHAP value (impact on $l_c$ Mean)", fontname='Times New Roman', fontsize=11)
ax_m.set_title("SHAP Beeswarm Plot for RF ($l_c$ Mean)", fontname='Times New Roman', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("TNR_RF_SHAP_lc_mean_beeswarm.pdf", bbox_inches='tight')
plt.close()

# --- 8. Plotting SHAP Explanations for Agent B ---
explainer_unser = shap.TreeExplainer(final_model_unser)
shap_values_unser = explainer_unser(X)

plt.figure()
shap.plots.beeswarm(shap_values_unser, show=False)
ax_u = plt.gca()
for label in (ax_u.get_xticklabels() + ax_u.get_yticklabels()):
    label.set_fontname('Times New Roman')
ax_u.set_xlabel("SHAP value (impact on Uncertainty $\sigma_{pred}$)", fontname='Times New Roman', fontsize=11)
ax_u.set_title("SHAP Beeswarm Plot for RF Uncertainty ($\sigma_{pred}$)", fontname='Times New Roman', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("TNR_RF_SHAP_lc_uncertainty_beeswarm.pdf", bbox_inches='tight')
plt.close()

# --- 9. Plotting Optuna Optimization Metrics ---
completed_trials = trials_mean_df[trials_mean_df['state'] == 'COMPLETE']
trial_numbers = completed_trials['number'].values
trial_values = completed_trials['value'].values
best_values = np.minimum.accumulate(trial_values)

fig_opt, ax_opt = plt.subplots(figsize=(5.5, 4))
ax_opt.scatter(trial_numbers, trial_values, color='#1f77b4', alpha=0.6, edgecolors='none', s=25, label='Trial objective value')
ax_opt.plot(trial_numbers, best_values, color='#d62728', linewidth=1.8, drawstyle='steps-post', label='Best objective value')
ax_opt.set_xlabel("Number of Trials")
ax_opt.set_ylabel("Cross-Validation RMSE")
ax_opt.set_title("Optuna Hyperparameter Optimization History (Agent A)", fontsize=11, fontweight='bold')
ax_opt.grid(True, linestyle=':', alpha=0.5)
ax_opt.legend(frameon=True, edgecolor='#cccccc', loc='upper right')
plt.tight_layout()
plt.savefig("TNR_RF_Optuna_Optimization_History.pdf", bbox_inches='tight')
plt.close()

# --- 10. Summary Performance Metrics Printout ---
mean_r2_list = [f['Mean_Fold_R2'] for f in fold_metrics_summary]
mean_rmse_list = [f['Mean_Fold_RMSE'] for f in fold_metrics_summary]
mean_mae_list = [f['Mean_Fold_MAE'] for f in fold_metrics_summary]
mean_mape_list = [f['Mean_Fold_MAPE(%)'] for f in fold_metrics_summary]

print("\n" + "="*65)
print(" FINAL DUAL-AGENT 10-FOLD CV PERFORMANCE")
print("-"*65)
print(f"  Average R²   : {np.mean(mean_r2_list):.4f} (±{np.std(mean_r2_list):.4f})")
print(f"  Average RMSE : {np.mean(mean_rmse_list):.4f} (±{np.std(mean_rmse_list):.4f})")
print(f"  Average MAE  : {np.mean(mean_mae_list):.4f} (±{np.std(mean_mae_list):.4f})")
print(f"  Average MAPE : {np.mean(mean_mape_list):.2f}% (±{np.std(mean_mape_list):.2f}%)")
print("="*65)