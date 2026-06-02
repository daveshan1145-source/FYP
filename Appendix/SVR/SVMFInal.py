# 1. Environment & Plotting Configurations
import os
import joblib
import optuna
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR

optuna.logging.set_verbosity(optuna.logging.WARNING)

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

# 2. Data Loading & Feature Setup
file_path = "data.xlsx"
df = pd.read_excel(file_path)

target_mean = 'Critical fragment length (µm)'
target_uncertainty = 'Uncertainty of Critical fragment length (µm)'  

X_raw = df.drop(columns=[target_mean, target_uncertainty])
y_mean = df[target_mean].values
y_unser = df[target_uncertainty].values  

X = pd.get_dummies(X_raw, drop_first=True)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 3. Hyperparameter Tuning (Optuna)
def objective_mean(trial):
    params = {
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
        'C': trial.suggest_float('C', 1.0, 2000.0, log=True),
        'epsilon': trial.suggest_float('epsilon', 0.01, 2.0, log=True),
    }
    
    if params['kernel'] in ['rbf', 'poly']:
        params['gamma'] = trial.suggest_categorical('gamma_mean', ['scale', 'auto']) 
    if params['kernel'] == 'poly':
        params['degree'] = trial.suggest_int('degree_mean', 2, 4)
        
    rmses = []
    for tr, va in kf.split(X):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X.iloc[tr])
        X_va = scaler.transform(X.iloc[va])
        model = SVR(**params).fit(X_tr, y_mean[tr])
        rmses.append(np.sqrt(mean_squared_error(y_mean[va], model.predict(X_va))))
    return np.mean(rmses)

study_mean = optuna.create_study(direction='minimize')
study_mean.optimize(objective_mean, n_trials=200)

best_params_mean = study_mean.best_params.copy()
if 'gamma_mean' in best_params_mean: best_params_mean['gamma'] = best_params_mean.pop('gamma_mean')
if 'degree_mean' in best_params_mean: best_params_mean['degree'] = best_params_mean.pop('degree_mean')

trials_mean_df = study_mean.trials_dataframe()
trials_mean_df.to_csv("svm_agent_A_tuning_history.csv", index=False)

def objective_unser(trial):
    params = {
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
        'C': trial.suggest_float('C', 1.0, 2000.0, log=True),
        'epsilon': trial.suggest_float('epsilon', 0.01, 1.0, log=True), 
    }
    
    if params['kernel'] in ['rbf', 'poly']:
        params['gamma'] = trial.suggest_categorical('gamma_unser', ['scale', 'auto'])
    if params['kernel'] == 'poly':
        params['degree'] = trial.suggest_int('degree_unser', 2, 4)
        
    rmses = []
    for tr, va in kf.split(X):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X.iloc[tr])
        X_va = scaler.transform(X.iloc[va])
        model = SVR(**params).fit(X_tr, y_unser[tr])
        rmses.append(np.sqrt(mean_squared_error(y_unser[va], model.predict(X_va))))
    return np.mean(rmses)

study_unser = optuna.create_study(direction='minimize')
study_unser.optimize(objective_unser, n_trials=100)

best_params_unser = study_unser.best_params.copy()
if 'gamma_unser' in best_params_unser: best_params_unser['gamma'] = best_params_unser.pop('gamma_unser')
if 'degree_unser' in best_params_unser: best_params_unser['degree'] = best_params_unser.pop('degree_unser')

trials_unser_df = study_unser.trials_dataframe()
trials_unser_df.to_csv("svm_agent_B_tuning_history.csv", index=False)

# 4. Out-of-Fold Cross Validation Evaluation
all_true_mean, all_pred_mean, all_pred_std = [], [], []
cv_records = []
fold_metrics_summary = [] 

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model_mean = SVR(**best_params_mean).fit(X_train_s, y_mean[train_idx])
    pred_mean = model_mean.predict(X_test_s)
    
    model_unser = SVR(**best_params_unser).fit(X_train_s, y_unser[train_idx])
    pred_std = np.maximum(model_unser.predict(X_test_s), 0)

    fold_r2 = r2_score(y_mean[test_idx], pred_mean)
    fold_rmse = np.sqrt(mean_squared_error(y_mean[test_idx], pred_mean))
    fold_mae = mean_absolute_error(y_mean[test_idx], pred_mean)
    fold_mape = np.mean(np.abs(y_mean[test_idx] - pred_mean) / (np.abs(y_mean[test_idx]) + 1e-8)) * 100

    all_true_mean.extend(y_mean[test_idx])
    all_pred_mean.extend(pred_mean)
    all_pred_std.extend(pred_std)

    fold_metrics_summary.append({
        'Fold': f"Fold_{fold}",
        'AgentA_C': best_params_mean['C'],
        'AgentA_Epsilon': best_params_mean['epsilon'],
        'AgentB_C': best_params_unser['C'],
        'AgentB_Epsilon': best_params_unser['epsilon'],
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

pd.DataFrame(fold_metrics_summary).to_csv("svm_dual_agent_fold_metrics.csv", index=False)
pd.DataFrame(cv_records).to_csv("svm_cv_predictions_pixel_level.csv", index=False)

# 5. Final Model Training & Exporting
final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X)

final_model_mean = SVR(**best_params_mean).fit(X_scaled, y_mean)
final_model_unser = SVR(**best_params_unser).fit(X_scaled, y_unser)

joblib.dump(final_model_mean, "final_svm_model_mean.pkl")
joblib.dump(final_model_unser, "final_svm_model_unser.pkl")
joblib.dump(final_scaler, "final_svm_scaler.pkl")

# 6. Performance & Interpretation Plotting
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
ax_pred.set_title("10-Fold Cross-Validation: SVM Dual-Agent Prediction", fontsize=11, fontweight='bold')
ax_pred.set_xlim(mn, mx)
ax_pred.set_ylim(mn, mx)
ax_pred.set_aspect('equal', adjustable='box')
ax_pred.legend(frameon=True, edgecolor='#cccccc', loc='upper left')
ax_pred.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("TNR_SVM_DualTask_Prediction.pdf", bbox_inches='tight')
plt.close()

background = shap.sample(X_scaled, min(50, X_scaled.shape[0]))

explainer_mean = shap.KernelExplainer(final_model_mean.predict, background)
shap_values_mean = explainer_mean.shap_values(X_scaled, nsamples=100)
plt.figure()
shap.summary_plot(shap_values_mean, X, show=False)
ax_m = plt.gca()
for label in (ax_m.get_xticklabels() + ax_m.get_yticklabels()):
    label.set_fontname('Times New Roman')
ax_m.set_xlabel("SHAP value (impact on $l_c$ Mean)", fontname='Times New Roman', fontsize=11)
ax_m.set_title("SHAP Beeswarm Plot for SVM ($l_c$ Mean)", fontname='Times New Roman', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("TNR_SVM_SHAP_lc_mean_beeswarm.pdf", bbox_inches='tight')
plt.close()

explainer_unser = shap.KernelExplainer(final_model_unser.predict, background)
shap_values_unser = explainer_unser.shap_values(X_scaled, nsamples=100)
plt.figure()
shap.summary_plot(shap_values_unser, X, show=False)
ax_u = plt.gca()
for label in (ax_u.get_xticklabels() + ax_u.get_yticklabels()):
    label.set_fontname('Times New Roman')
ax_u.set_xlabel("SHAP value (impact on Uncertainty $\sigma_{pred}$)", fontname='Times New Roman', fontsize=11)
ax_u.set_title("SHAP Beeswarm Plot for SVM Uncertainty ($\sigma_{pred}$)", fontname='Times New Roman', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("TNR_SVM_SHAP_lc_uncertainty_beeswarm.pdf", bbox_inches='tight')
plt.close()

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
plt.savefig("TNR_SVM_Optuna_Optimization_History.pdf", bbox_inches='tight')
plt.close()