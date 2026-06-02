# 1. Environment & Plotting Configurations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

# 2. Data Loading & Feature Setup
file_path = "data.xlsx"
df = pd.read_excel(file_path)

target_col = df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col].values 

cat_cols = ["IfSized", "IfTreated"]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="if_binary"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ]
)

cv = KFold(n_splits=10, shuffle=True, random_state=42)

# 4. Out-of-Fold Cross Validation Evaluation
pipe_lr = Pipeline([
    ("preprocess", preprocess), 
    ("model", LinearRegression())
])

oof_lr = cross_val_predict(pipe_lr, X, y, cv=cv, n_jobs=-1)

r2 = r2_score(y, oof_lr)
rmse = np.sqrt(mean_squared_error(y, oof_lr))
mae = mean_absolute_error(y, oof_lr)
epsilon = 1e-8
mape = np.mean(np.abs((y - oof_lr) / (y + epsilon))) * 100

df_output = X.copy()
df_output["Experimental $l_c$ ($\mu$m)"] = y
df_output["Predicted $l_c$ ($\mu$m) by Linear_Regression"] = oof_lr
df_output.to_csv("linear_regression_cv_predictions.csv", index=False)

df_metrics = pd.DataFrame({
    'Metric': ['R2', 'RMSE', 'MAE', 'MAPE_percent'],
    'Value': [r2, rmse, mae, mape]
})
df_metrics.to_csv("linear_regression_metrics_summary.csv", index=False)

# 6. Performance & Interpretation Plotting
fig, ax = plt.subplots(figsize=(6, 5))
min_v = min(y.min(), oof_lr.min())
max_v = max(y.max(), oof_lr.max())
ax.scatter(y, oof_lr, alpha=0.6, color="#1f77b4", edgecolors='w', s=40, label="Data Points")
ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.5, label="Perfect Fit Line")
ax.set_xlabel("Experimental $l_c$ ($\mu$m)", fontsize=11)
ax.set_ylabel("Predicted $l_c$ ($\mu$m) by Linear Regression", fontsize=11)
ax.set_xlim(min_v, max_v)
ax.set_ylim(min_v, max_v)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig("linear_regression_oof_performance.pdf", bbox_inches='tight', dpi=300)
plt.close()

residuals = y - oof_lr
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(oof_lr, residuals, alpha=0.6, color="#1f77b4", edgecolors="w", s=40, label="Residuals")
ax.axhline(y=0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
slope, intercept = np.polyfit(oof_lr, residuals, 1)
x_trend = np.linspace(oof_lr.min(), oof_lr.max(), 100)
y_trend = slope * x_trend + intercept
ax.plot(x_trend, y_trend, color="red", linewidth=2, label=f"Trend (slope: {slope:.2e})")
ax.set_xlabel("Predicted $l_c$ ($\mu$m)", fontsize=11)
ax.set_ylabel("Residuals (Experimental - Predicted) ($\mu$m)", fontsize=11)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig("linear_regression_residual_plot.pdf", bbox_inches="tight", dpi=300)
plt.close()

std_residuals = residuals / np.std(residuals)
core_mask = np.abs(std_residuals) <= 2
coef = np.polyfit(oof_lr[core_mask], std_residuals[core_mask], 1)
poly1d_fn = np.poly1d(coef)
fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(oof_lr, std_residuals, alpha=0.6, color="#1f77b4", edgecolors="w", s=40, label="Data Points")
x_trend_core = np.linspace(min(oof_lr[core_mask]), max(oof_lr[core_mask]), 100)
ax.plot(x_trend_core, poly1d_fn(x_trend_core), color="green", linestyle="-", linewidth=2, label=f"Core Trend (slope: {coef[0]:.2e})")
ax.axhline(0, color='red', linestyle='--')
ax.axhline(2, color='gray', linestyle=':')
ax.axhline(-2, color='gray', linestyle=':')
ax.set_xlabel("Predicted $l_c$ ($\mu$m)")
ax.set_ylabel("Standardized Residuals")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper right", frameon=True, edgecolor="none")
plt.tight_layout()
plt.savefig("linear_regression_standardized_residuals.pdf", bbox_inches='tight', dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(5,5))
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title("Normal Q-Q Plot")
plt.tight_layout()
plt.savefig("linear_regression_qqplot.pdf", bbox_inches='tight', dpi=300)
plt.close()