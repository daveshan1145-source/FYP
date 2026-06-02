import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# Silencing TensorFlow logs and Optuna default printouts
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Global matplotlib configuration for academic journals (Times New Roman)
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

# Data ingestion and preprocessing
df = pd.read_excel("data.xlsx")
target_mean = 'Critical fragment length (µm)'
target_uncertainty = 'Uncertainty of Critical fragment length (µm)'  

X_raw = df.drop(columns=[target_mean, target_uncertainty])
y_mean = df[target_mean].values
y_unser = df[target_uncertainty].values  

# Categorical encoding and CV setup
X = pd.get_dummies(X_raw, columns=['IfSized', 'IfTreated'], drop_first=True)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Multi-Task ANN builder function
def build_multi_task_ann(input_dim, shared_layers, lr):
    inputs = Input(shape=(input_dim,))
    x = inputs
    for units in shared_layers:
        x = Dense(units, activation='relu')(x)
        
    out_mean = Dense(1, name="output_mean")(x)
    out_unser = Dense(1, name="output_unser")(x)
    
    model = Model(inputs=inputs, outputs=[out_mean, out_unser])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={"output_mean": "mse", "output_unser": "mse"},
        loss_weights={"output_mean": 1.0, "output_unser": 0.2}
    )
    return model

# Objective function for hyperparameter optimization
def objective_shared(trial):
    n_layers = trial.suggest_int('n_shared_layers', 1, 5)
    layers = [trial.suggest_int(f'shared_units_L{i}', 16, 128, step=16) for i in range(n_layers)]
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    rmses = []
    for tr, va in kf.split(X):
        scaler_X = StandardScaler()
        scaler_yA = StandardScaler()
        scaler_yB = StandardScaler()
        
        X_tr = scaler_X.fit_transform(X.iloc[tr])
        X_va = scaler_X.transform(X.iloc[va])
        
        y_tr_M = scaler_yA.fit_transform(y_mean[tr].reshape(-1, 1)).flatten()
        y_tr_U = scaler_yB.fit_transform(y_unser[tr].reshape(-1, 1)).flatten()
        
        model = build_multi_task_ann(X_tr.shape[1], layers, lr)
        early_stop = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
        
        model.fit(
            X_tr, {"output_mean": y_tr_M, "output_unser": y_tr_U},
            validation_split=0.2, epochs=400, batch_size=batch_size, callbacks=[early_stop], verbose=0
        )
        
        pred_M_s, _ = model.predict(X_va, verbose=0)
        pred_M = scaler_yA.inverse_transform(pred_M_s.reshape(-1, 1)).flatten()
        rmses.append(np.sqrt(mean_squared_error(y_mean[va], pred_M)))
        
    return np.mean(rmses)

# Execute parameter tuning
study_shared = optuna.create_study(direction='minimize')
study_shared.optimize(objective_shared, n_trials=100)
best_params = study_shared.best_params

trials_df = study_shared.trials_dataframe()
trials_df.to_csv("ann_multitask_tuning_history.csv", index=False)

# Validation loop utilizing optimized hyper-parameters
all_true_mean, all_pred_mean, all_pred_std = [], [], []
fold_metrics_summary = [] 
cv_records = []
shared_layers = [best_params[f'shared_units_L{i}'] for i in range(best_params['n_shared_layers'])]

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    
    scaler_X = StandardScaler()
    scaler_yA = StandardScaler()
    scaler_yB = StandardScaler()
    
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    y_train_M_s = scaler_yA.fit_transform(y_mean[train_idx].reshape(-1, 1)).flatten()
    y_train_U_s = scaler_yB.fit_transform(y_unser[train_idx].reshape(-1, 1)).flatten()

    model = build_multi_task_ann(X_train_s.shape[1], shared_layers, best_params['lr'])
    es = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    
    model.fit(
        X_train_s, {"output_mean": y_train_M_s, "output_unser": y_train_U_s},
        validation_split=0.1, epochs=1000, batch_size=best_params['batch_size'], callbacks=[es], verbose=0
    )
    
    pred_M_s, pred_U_s = model.predict(X_test_s, verbose=0)
    pred_mean = scaler_yA.inverse_transform(pred_M_s.reshape(-1, 1)).flatten()
    pred_std = scaler_yB.inverse_transform(pred_U_s.reshape(-1, 1)).flatten()
    pred_std = np.maximum(pred_std, 0) 

    all_true_mean.extend(y_mean[test_idx])
    all_pred_mean.extend(pred_mean)
    all_pred_std.extend(pred_std)

    fold_metrics_summary.append({
        'Fold': f"Fold_{fold}",
        'Mean_Fold_R2': r2_score(y_mean[test_idx], pred_mean),
        'Mean_Fold_RMSE': np.sqrt(mean_squared_error(y_mean[test_idx], pred_mean)),
        'Mean_Fold_MAE': mean_absolute_error(y_mean[test_idx], pred_mean)
    })

    for i, idx in enumerate(test_idx):
        cv_records.append({
            'Original_Index': idx, 'Fold_Group': f"Fold_{fold}",
            'True_lc': y_mean[idx], 'True_Uncertainty': y_unser[idx],
            'Predicted_lc': pred_mean[i], 'Predicted_sigma': pred_std[i]
        })

pd.DataFrame(fold_metrics_summary).to_csv("ann_multitask_fold_metrics.csv", index=False)
pd.DataFrame(cv_records).to_csv("ann_multitask_predictions.csv", index=False)

# Train and dump final model artifacts using complete dataset
final_scaler_X = StandardScaler()
final_scaler_yA = StandardScaler()
final_scaler_yB = StandardScaler()

X_scaled = final_scaler_X.fit_transform(X)
y_mean_scaled = final_scaler_yA.fit_transform(y_mean.reshape(-1, 1)).flatten()
y_unser_scaled = final_scaler_yB.fit_transform(y_unser.reshape(-1, 1)).flatten()

final_model = build_multi_task_ann(X_scaled.shape[1], shared_layers, best_params['lr'])
final_model.fit(X_scaled, {"output_mean": y_mean_scaled, "output_unser": y_unser_scaled}, epochs=1000, batch_size=best_params['batch_size'], verbose=0)

final_model.save("final_ann_multitask_model.h5")
joblib.dump(final_scaler_X, "final_ann_shared_scaler_X.pkl")
joblib.dump(final_scaler_yA, "final_ann_shared_scaler_yA.pkl")
joblib.dump(final_scaler_yB, "final_ann_shared_scaler_yB.pkl")

# Generate regression performance evaluation plot
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
ax_pred.plot(x_line, x_line * 0.90, color='#7f7f7f', linestyle=':')
ax_pred.set_xlabel("Experimental $l_c$ ($\mu$m)")
ax_pred.set_ylabel("Predicted $l_c$ ($\mu$m)")
ax_pred.set_xlim(mn, mx)
ax_pred.set_ylim(mn, mx)
ax_pred.set_aspect('equal', adjustable='box')
ax_pred.legend(frameon=True, edgecolor='#cccccc', loc='upper left')
ax_pred.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("TNR_ANN_MultiTask_Prediction.pdf", bbox_inches='tight')

# Generate convergence optimization trajectory plot
completed_trials = trials_df[trials_df['state'] == 'COMPLETE']
trial_numbers = completed_trials['number'].values
trial_values = completed_trials['value'].values
best_values = np.minimum.accumulate(trial_values)

fig_opt, ax_opt = plt.subplots(figsize=(5.5, 4))
ax_opt.scatter(trial_numbers, trial_values, color='#1f77b4', alpha=0.6, edgecolors='none', s=25, label='Trial Value')
ax_opt.plot(trial_numbers, best_values, color='#d62728', linewidth=1.8, drawstyle='steps-post', label='Best Value')
ax_opt.set_xlabel("Number of Trials")
ax_opt.set_ylabel("Cross-Validation RMSE")
ax_opt.grid(True, linestyle=':', alpha=0.5)
ax_opt.legend(frameon=True, edgecolor='#cccccc', loc='upper right')
plt.tight_layout()
plt.savefig("TNR_ANN_MultiTask_Optuna_History.pdf", bbox_inches='tight')