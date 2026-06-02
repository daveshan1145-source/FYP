# 1. Environment & Plotting Configurations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
df = pd.read_excel("data.xlsx")
target_mean = 'Critical fragment length (µm)'
target_uncertainty = 'Uncertainty of Critical fragment length (µm)'  

X = df.drop(columns=[target_mean, target_uncertainty])
y_mean = df[target_mean].values
y_unser = df[target_uncertainty].values  
y = np.column_stack((y_mean, y_unser))

cat_features = ['IfSized', 'IfTreated']
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 3. Hyperparameter Tuning (Optuna)
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 3500, 15000),
        'depth': trial.suggest_int('depth', 4, 8),              
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3.0, 7.0),
        'loss_function': 'MultiRMSE',
        'random_seed': 42,
        'verbose': 0
    }
    
    cv_mean_rmse = []
    cv_unser_rmse = []
    
    for train_idx, test_idx in kf.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_va = y[train_idx], y[test_idx]
        
        model_cv = CatBoostRegressor(**params)
        model_cv.fit(X_tr, y_tr, cat_features=cat_features, eval_set=(X_va, y_va), early_stopping_rounds=150, verbose=0)
        
        preds = model_cv.predict(X_va)
        cv_mean_rmse.append(np.sqrt(mean_squared_error(y_va[:, 0], preds[:, 0])))
        cv_unser_rmse.append(np.sqrt(mean_squared_error(y_va[:, 1], preds[:, 1])))
        
    return np.mean(cv_mean_rmse) + np.mean(cv_unser_rmse) * 0.5

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)  
best_params = study.best_params

trials_df = study.trials_dataframe()
trials_df.to_csv("optuna_tuning_history.csv", index=False)

# 4. Out-of-Fold Cross Validation Evaluation
best_params['loss_function'] = 'MultiRMSE'
best_params['random_seed'] = 42

r2_list, rmse_list, mae_list, mape_list = [], [], [], []
all_true_mean, all_pred_mean, all_pred_std = [], [], []
cv_records = []
fold_metrics_summary = []  

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = CatBoostRegressor(**best_params)
    model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test), early_stopping_rounds=200, verbose=0)

    preds = model.predict(X_test)
    pred_mean = preds[:, 0]
    pred_std = np.maximum(preds[:, 1], 0)  
    true_mean = y_test[:, 0]
    true_unser = y_test[:, 1]

    fold_r2 = r2_score(true_mean, pred_mean)
    fold_rmse = np.sqrt(mean_squared_error(true_mean, pred_mean))
    fold_mae = mean_absolute_error(true_mean, pred_mean)
    fold_mape = np.mean(np.abs(true_mean - pred_mean) / (np.abs(true_mean) + 1e-8)) * 100

    r2_list.append(fold_r2)
    rmse_list.append(fold_rmse)
    mae_list.append(fold_mae)
    mape_list.append(fold_mape)

    all_true_mean.extend(true_mean)
    all_pred_mean.extend(pred_mean)
    all_pred_std.extend(pred_std)

    fold_metrics_summary.append({
        'Fold': f"Fold_{fold}",
        'Actual_Best_Iteration': model.get_best_iteration(),  
        'Fold_R2': fold_r2, 'Fold_RMSE': fold_rmse, 'Fold_MAE': fold_mae
    })

    for i, idx in enumerate(test_idx):
        cv_records.append({
            'Original_Index': idx, 'Fold_Group': f"Fold_{fold}",
            'True_lc': true_mean[i], 'True_Uncertainty': true_unser[i],
            'Predicted_lc': pred_mean[i], 'Predicted_sigma': pred_std[i]
        })

pd.DataFrame(fold_metrics_summary).to_csv("cv_fold_parameters_and_metrics.csv", index=False)
pd.DataFrame(cv_records).to_csv("cv_predictions_pixel_level.csv", index=False)

# 5. Final Model Training & Exporting
final_model = CatBoostRegressor(**best_params)
final_model.fit(X, y, cat_features=cat_features, verbose=0)
final_model.save_model("final_multi_rmse_model.cbm")

shap_params = best_params.copy()
shap_params['iterations'] = 1000 
shap_params['verbose'] = 0

shap_params['loss_function'] = 'RMSE'
model_for_mean = CatBoostRegressor(**shap_params)
model_for_mean.fit(X, y_mean, cat_features=cat_features) 
model_for_mean.save_model("shap_model_mean.cbm") 

model_for_std = CatBoostRegressor(**shap_params)
model_for_std.fit(X, y_unser, cat_features=cat_features) 
model_for_std.save_model("shap_model_unser.cbm") 

# 6. Performance & Interpretation Plotting
fig_pred, ax_pred = plt.subplots(figsize=(5, 5))
ax_pred.errorbar(
    all_true_mean, all_pred_mean, yerr=all_pred_std,
    fmt='o', color='#1f77b4', ecolor='#d62728', elinewidth=0.7, capsize=1.5, markersize=4.5, alpha=0.75,
    label='Predicted $l_c$ $\pm$ $\sigma_{pred}$'
)
mn = min(min(all_true_mean), min(all_pred_mean)) * 0.92
mx = max(max(all_true_mean), max(all_pred_mean)) * 1.08
x_line = np.linspace(mn, mx, 100)
ax_pred.plot(x_line, x_line, color='#333333', linestyle='-', linewidth=1.2, label="y=x")
ax_pred.plot(x_line, x_line * 1.10, color='#7f7f7f', linestyle=':', linewidth=1.0, label="$\pm$10% Bounds")
ax_pred.plot(x_line, x_line * 0.90, color='#7f7f7f', linestyle=':', linewidth=1.0)
ax_pred.set_xlabel("Experimental $l_c$ ($\mu$m)")
ax_pred.set_ylabel("Predicted $l_c$ ($\mu$m)")
ax_pred.set_xlim(mn, mx)
ax_pred.set_ylim(mn, mx)
ax_pred.set_aspect('equal', adjustable='box')
ax_pred.legend(frameon=True, edgecolor='#cccccc', loc='upper left')
ax_pred.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("TNR_lc_Prediction_with_10percent_bounds.pdf", bbox_inches='tight')

explainer_mean = shap.TreeExplainer(model_for_mean)
shap_values_mean = explainer_mean(X, check_additivity=False)
plt.figure()
shap.plots.beeswarm(shap_values_mean, show=False)
plt.tight_layout()
plt.savefig("TNR_SHAP_lc_mean.pdf", bbox_inches='tight')

explainer_std = shap.TreeExplainer(model_for_std)
shap_values_std = explainer_std(X, check_additivity=False)
plt.figure()
shap.plots.beeswarm(shap_values_std, show=False)
plt.tight_layout()
plt.savefig("TNR_SHAP_lc_uncertainty.pdf", bbox_inches='tight')

completed_trials = trials_df[trials_df['state'] == 'COMPLETE']
trial_numbers = completed_trials['number'].values
trial_values = completed_trials['value'].values
best_values = np.minimum.accumulate(trial_values)

fig_opt, ax_opt = plt.subplots(figsize=(5.5, 4))
ax_opt.scatter(trial_numbers, trial_values, color='#1f77b4', alpha=0.6, edgecolors='none', s=25, label='Trial Value')
ax_opt.plot(trial_numbers, best_values, color='#d62728', linewidth=1.8, drawstyle='steps-post', label='Best Value')
ax_opt.set_xlabel("Number of Trials")
ax_opt.set_ylabel("Multi-Task Combined Loss")
ax_opt.grid(True, linestyle=':', alpha=0.5)
ax_opt.legend(frameon=True, edgecolor='#cccccc', loc='upper right')
plt.tight_layout()
plt.savefig("TNR_Optuna_Optimization_History.pdf", bbox_inches='tight')