import streamlit as st
import pandas as pd
from PIL import Image
import json
from pathlib import Path
import joblib
# =======================================
# Path setup
# =======================================
BASE_DIR = Path(__file__).resolve().parent  # 当前脚本所在目录
fig_dir = BASE_DIR / "figures"              # figures 子目录
# =======================================
# Page setup
# =======================================
st.set_page_config(
    page_title="Collateral Modeling Dashboard",
    page_icon="🏦",
    layout="wide"
)
st.markdown(
    """
    <style>
    img {
        max-width: 70% !important;
        height: auto;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    # 🏦 Collateral Modeling Dashboard
    A compact, four-tab view for your modeling outputs.  
    Just update the file paths inside each tab.
    """
)

# =======================================
# Tabs
# =======================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 Residuals",
    "📖 Feature Introduction",
    "🧪 Feature Selection / Importance",
    "🔧 Best Hyperparameters",
    "🏠 Collateral Price Predictions"
])

# -------- Tab 1: Residuals
with tab1:
    st.subheader("📉 Residuals")
    st.caption("A quick look at residual distribution / fit diagnostics.")

    # Select model for residual plot
    model_choice = st.radio(
        "Select model:",
        ["ElasticNet", "ExtraTrees", "XGBoost"],
        horizontal=True,
        key="residual_radio_tab1"  # ✅ 唯一 key
    )

    residual_paths = {
        "ElasticNet": fig_dir / "enet_residuals_scatter.png",
        "ExtraTrees": fig_dir / "extra_residuals_scatter.png",
        "XGBoost": fig_dir / "xgb_residuals_scatter.png"
    }

    residual_img_path = residual_paths[model_choice]

    try:
        img = Image.open(residual_img_path)
        st.image(img, use_container_width=True, caption=f"Residuals Plot - {model_choice}")
    except Exception as e:
        st.warning(f"Could not display residual image for {model_choice}: {e}")

    st.markdown(
        """
        **Notes**
        - Check for heteroskedasticity patterns
        - Look for non-linearity or outliers
        - Validate residual symmetry & spread
        """
    )

##########################################################################
with tab2:
    st.subheader("📖 Feature Introduction")
    st.markdown("""
    **Features Input**  
    1. CRIM 🚨 — per capita crime rate by town  
    2. ZN 🏡 — proportion of residential land zoned for lots over 25,000 sq.ft.  
    3. INDUS 🏭 — proportion of non-retail business acres per town  
    4. CHAS 🌊 — Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)  
    5. NOX 💨 — nitric oxides concentration (parts per 10 million)  
    6. RM 🛏️ — average number of rooms per dwelling  
    7. AGE 🏚️ — proportion of owner-occupied units built prior to 1940  
    8. DIS 📍 — weighted distances to five Boston employment centres  
    9. RAD 🛣️ — index of accessibility to radial highways  
    10. TAX 💰 — full-value property-tax rate per $10,000  
    11. PTRATIO 🏫 — pupil-teacher ratio by town  
    12. B 🧑🏿 — 1000(Bk - 0.63)^2, where Bk is the proportion of blacks by town  
    13. LSTAT 📉 — % lower status of the population  

    **Target Variable**  
    14. MEDV 💵 — Median value of owner-occupied homes in $1000's
    """)

# -------- Tab 3: Feature Selection / Importance
with tab3:
    st.subheader("🧪 Feature Selection / Importance")
    st.caption("Highlighting the drivers behind model performance.")

    feat_choice = st.radio(
        "Select model:",
        ["ElasticNet", "ExtraTrees", "XGBoost"],
        horizontal=True,
        key="feature_radio_tab2"
    )

    feat_files = {
        "ExtraTrees": fig_dir / "extra_feature_importance.png",
        "XGBoost": fig_dir / "xgb_feature_importance.png"
    }

    try:
        if feat_choice == "ElasticNet":
            # 直接文字展示选中的特征
            selected_features = [
                "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
                "DIS", "RAD", "TAX", "PT", "B", "LSTAT"
            ]
            st.markdown(
                "### Selected Features (ElasticNet)\n"
                + ", ".join(f"`{feat}`" for feat in selected_features)
            )
        else:
            # 保持读图
            selected_file = feat_files[feat_choice]
            img = Image.open(selected_file)
            st.image(img, use_container_width=True, caption=f"Feature Importance - {feat_choice}")

    except Exception as e:
        st.warning(f"Could not display feature info for {feat_choice}: {e}")

    st.markdown(
        """
        **Tips**
        - Confirm top contributors align with domain intuition  
        - Watch for leakage features  
        - Revisit feature selection if importance is too concentrated
        """
    )



# -------- Tab 4: Best Hyperparameters
with tab4:
    st.subheader("🔧 Best Hyperparameters")
    st.caption("Winning configuration from your tuning process.")

    params_file = "best_params_all.txt"  # <-- Change your params TXT file here
    try:
        with open(params_file, "r", encoding="utf-8") as f:
            txt_content = f.read()
        # 更美观的代码高亮显示
        st.code(txt_content, language="yaml")
    except Exception as e:
        st.warning(f"Could not read params file: {e}")



# -------- Tab 5: Collateral Price Predictions
with tab5:
    st.subheader("🏠 Collateral Price Predictions")
    st.info(
        "Default values are the mean of each feature in the training dataset.Enter your feature values to get predictions from all 3 models.")

    # 模型路径
    model_paths = {
        "ElasticNet": fig_dir / "enet_final_model.pkl",
        "ExtraTrees": fig_dir / "extra_final_model.pkl",
        "XGBoost": fig_dir / "xgb_final_model.pkl"
    }

    # 加载模型
    models = {}
    for name, path in model_paths.items():
        try:
            models[name] = joblib.load(path)
        except Exception as e:
            st.warning(f"Could not load {name} model: {e}")

    # ===== 从训练时保存的文件读取特征列 =====
    try:
        feature_cols = pd.read_csv(fig_dir / "feature_cols.txt", header=None)[0].tolist()
    except FileNotFoundError:
        st.error("❌ feature_cols.txt not found. Please run training script first.")
        feature_cols = []

    # ===== 从训练时保存的文件读取特征均值 =====
    try:
        feature_means = pd.read_csv(fig_dir / "feature_means.csv", index_col=0)["mean"].to_dict()
    except FileNotFoundError:
        feature_means = {feat: 0.0 for feat in feature_cols}

    st.markdown("### 📋 Input Feature Values")
    user_input = {}
    cols = st.columns(3)
    for i, feat in enumerate(feature_cols):
        col = cols[i % 3]
        default_val = feature_means.get(feat, 0.0)
        user_input[feat] = col.number_input(
            feat, value=float(default_val), step=0.1
        )

    # 按钮预测
    if st.button("🔮 Predict Price"):
        # 按训练时的列顺序生成 DataFrame
        input_df = pd.DataFrame([[user_input[feat] for feat in feature_cols]], columns=feature_cols)

        results = {}
        for name, model in models.items():
            try:
                pred = model.predict(input_df)[0]
                results[name] = pred
            except Exception as e:
                results[name] = f"Error: {e}"

        st.markdown("### 📊 Prediction Results")
        res_df = pd.DataFrame(list(results.items()), columns=["Model", "Predicted Price"])
        st.dataframe(
            res_df.style.set_properties(**{"font-size": "20px"}),  # 调整字体
            use_container_width=True,
            hide_index=True  # ✅ 隐藏索引
        )




# Footer
st.markdown("---")
st.caption("© Your Team • This dashboard only formats and displays your static outputs.")
# streamlit run C:\Users\10526\PycharmProjects\PythonProject\main_5548.py