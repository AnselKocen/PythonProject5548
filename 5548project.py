import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# === 全局浅色主题 ===
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_theme(style="whitegrid")

# === Step 1: 读取数据
df = pd.read_csv("boston.csv")  # 请确保路径正确
target_col = "MV"
X = df.drop(columns=[target_col])
y = df[target_col]

# === Step 2: 拆分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# === Step 3: 设置模型与参数网格
model_configs = {
    "enet": {
        "estimator": ElasticNet(max_iter=10000, random_state=42),
        "param_grid": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0],
            "model__l1_ratio": [0.1, 0.5, 0.9]
        }
    },
    "extra": {
        "estimator": ExtraTreesRegressor(random_state=42),
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 10],
            "model__max_features": ["sqrt", "log2"]
        }
    },
    "xgb": {
        "estimator": XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1),
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.01, 0.1]
        }
    }
}

# === Step 4: 创建保存路径
model_dir = Path("boston_model_results")
model_dir.mkdir(parents=True, exist_ok=True)
fig_dir = Path("boston_model_results/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

# === Step 5: 训练模型、调参、评估与可视化
results = {}

for model_key, config in model_configs.items():
    print(f"\n [{model_key.upper()}] Hyperparameter tuning...")

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", config["estimator"])
    ])

    grid = GridSearchCV(pipe, config["param_grid"], scoring="neg_mean_squared_error", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    # 保存模型与参数
    joblib.dump(grid.best_estimator_, model_dir / f"{model_key}_final_model.pkl")
    with open(model_dir / f"{model_key}_best_params.txt", "w") as f:
        f.write(str(grid.best_params_))
    print('-----------------------------------------------------')
    print(f"Best hyper-parameter: {grid.best_params_}")

    # 预测与评估
    y_test_pred = grid.predict(X_test)
    y_train_pred = grid.predict(X_train)

    r2 = r2_score(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    results[model_key] = {"r2": r2, "mse": mse, "rmse": rmse}
    print(f" R²: {r2:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f}")

    # === 残差图（白底版） ===
    train_residuals = y_train_pred - y_train
    test_residuals = y_test_pred - y_test

    plt.figure(figsize=(8, 6))
    plt.scatter(
        y_train_pred, train_residuals,
        c='deepskyblue', marker='o', s=35, alpha=0.7, label='Training data'
    )
    plt.scatter(
        y_test_pred, test_residuals,
        c='orange', marker='s', s=35, alpha=0.7, label='Test data'
    )

    # 参考线（灰色）
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    plt.xlabel(f'{model_key.upper()} Predicted values')
    plt.ylabel(f'{model_key.upper()} Residuals')
    plt.title(f'{model_key.upper()} Residual Plot (Train vs Test)')
    plt.legend(loc='upper left')
    plt.xlim([-10, 50])
    plt.tight_layout()

    plt.savefig(fig_dir / f"{model_key}_residuals_scatter.png", dpi=300)
    plt.close()

    # === ElasticNet 变量选择输出
    if model_key == "enet":
        enet_model = grid.best_estimator_.named_steps["model"]
        nonzero_mask = enet_model.coef_ != 0
        selected_features = X.columns[nonzero_mask]
        with open(model_dir / "enet_selected_features.txt", "w") as f:
            for feat in selected_features:
                f.write(f"{feat}\n")

    # === ExtraTrees / XGBoost 特征重要性输出（白底版） ===
    if model_key in ["extra", "xgb"]:
        model = grid.best_estimator_.named_steps["model"]
        importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        plt.figure(figsize=(8, 6))
        sns.barplot(
            data=importance_df.head(10),
            x="importance", y="feature",
            palette="viridis"
        )
        plt.title(f"{model_key.upper()} Feature Importance (Top 10)")
        plt.tight_layout()

        plt.savefig(fig_dir / f"{model_key}_feature_importance.png", dpi=300)
        plt.close()

        importance_df.to_csv(model_dir / f"{model_key}_feature_importance.csv", index=False)

# === Step 6: 保存特征信息
BASE_DIR = Path(__file__).resolve().parent
X.columns.to_series().to_csv(BASE_DIR / "figures" / "feature_cols.txt", index=False, header=False)
X.mean().to_csv(BASE_DIR / "figures" / "feature_means.csv", header=["mean"])

# === Step 7: 输出结果表
results_df = pd.DataFrame(results).T
results_df.to_csv(model_dir / "model_comparison.csv", index=True)
print("\n all save to CSV：model_comparison.csv")
