import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from joblib import Parallel, delayed

# --- 1. Global Plotting Configurations ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

# --- 2. Data Loading & Preparation ---
file_path = "data.xlsx"
df = pd.read_excel(file_path)
target_mean = 'Critical fragment length (µm)'
target_uncertainty = 'Uncertainty of Critical fragment length (µm)'
X = df.drop(columns=[target_mean, target_uncertainty])
y_mean, y_unser = df[target_mean].values, df[target_uncertainty].values
y = np.column_stack((y_mean, y_unser))
cat_features = ['IfSized', 'IfTreated']

# --- 3. Best Parameter Extraction ---
optuna_history = pd.read_csv("optuna_tuning_history.csv")
best_trial = optuna_history.loc[optuna_history['value'].idxmin()]
best_params = {
    'iterations': int(best_trial['params_iterations']),
    'depth': int(best_trial['params_depth']),
    'learning_rate': float(best_trial['params_learning_rate']),
    'l2_leaf_reg': float(best_trial['params_l2_leaf_reg']),
    'loss_function': 'MultiRMSE', 'verbose': 0, 'random_seed': 42
}

# --- 4. Task Flattening for Parallel Processing ---
n_splits, n_repeats, seeds = 5, 20, list(range(10))
tasks = []
for seed in seeds:
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for fold_id, (train_idx, test_idx) in enumerate(rkf.split(X), 1):
        tasks.append({'seed': seed, 'fold_id': fold_id, 'train_idx': train_idx, 'test_idx': test_idx})

# --- 5. Fold Worker Function ---
def run_single_fold_worker(task, X_data, y_data, base_params, cat_feats):
    t_idx, v_idx = task['train_idx'], task['test_idx']
    X_tr, X_te = X_data.iloc[t_idx], X_data.iloc[v_idx]
    y_tr, y_te = y_data[t_idx], y_data[v_idx]
    local_params = base_params.copy()
    local_params['thread_count'] = 1
    model = CatBoostRegressor(**local_params).fit(X_tr, y_tr, cat_features=cat_feats, eval_set=(X_te, y_te), early_stopping_rounds=200, verbose=False)
    preds = model.predict(X_te)
    r2 = r2_score(y_te[:, 0], preds[:, 0])
    rmse = np.sqrt(mean_squared_error(y_te[:, 0], preds[:, 0]))
    mae = mean_absolute_error(y_te[:, 0], preds[:, 0])
    mape = np.mean(np.abs((y_te[:, 0] - preds[:, 0]) / (y_te[:, 0] + 1e-8))) * 100
    return {'Seed': task['seed'], 'Fold_ID': task['fold_id'], 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE_percent': mape}

# --- 6. Parallel Execution ---
records = Parallel(n_jobs=-1, backend='loky')(delayed(run_single_fold_worker)(task, X, y, best_params, cat_features) for task in tasks)
df_results = pd.DataFrame(records)
df_results.to_csv("repeated_cv_results_10seeds.csv", index=False)

# --- 7. Metric Calculation ---
all_r2, all_rmse, all_mae, all_mape = df_results['R2'], df_results['RMSE'], df_results['MAE'], df_results['MAPE_percent']
summary_df = pd.DataFrame({'Metric': ['R2', 'RMSE', 'MAE', 'MAPE%'], 'Mean': [all_r2.mean(), all_rmse.mean(), all_mae.mean(), all_mape.mean()], 'Std': [all_r2.std(), all_rmse.std(), all_mae.std(), all_mape.std()]})
summary_df.to_csv("repeated_cv_summary_statistics_10seeds.csv", index=False)

# --- 8. Metric Distribution Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
metrics_data = [all_r2, all_rmse, all_mae, all_mape]
metrics_names, colors = ['$R^2$', 'RMSE', 'MAE', 'MAPE (%)'], ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for i, (data, name, color) in enumerate(zip(metrics_data, metrics_names, colors)):
    ax = axes[i]
    ax.boxplot(data, vert=True, patch_artist=True, showmeans=True, boxprops=dict(facecolor=color, alpha=0.6), medianprops=dict(color='black', linewidth=2))
    ax.set_title(f'Distribution of {name}', fontsize=12, pad=10)
    ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("metric_stability_2x2_subplots.pdf", bbox_inches='tight', dpi=300)

# --- 9. R2 Histogram ---
fig_hist, ax_hist = plt.subplots(figsize=(5.5, 4.2))
ax_hist.hist(all_r2, bins=30, color='#1f77b4', alpha=0.7, edgecolor='white')
ax_hist.set_xlabel("$R^2$")
ax_hist.set_ylabel("Frequency")
ax_hist.set_title("$R^2$ Empirical Distribution", pad=12)
ax_hist.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("R2_histogram_professional.pdf", bbox_inches='tight', dpi=300)
plt.show()