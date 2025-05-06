# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import plotly.express as px
import plotly.graph_objects as go

# --- 页面配置 & 样式 ---
st.set_page_config(page_title="智能应力预测系统", layout="wide")
st.markdown("""
<style>
.section-card {
    background: rgba(255,255,255,0.95);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}
.stSidebar .stButton>button {
    width: 100%;
    border-radius: 8px !important;
    background-color: #4a90e2 !important;
    color: white !important;
    padding: 0.6rem 1rem !important;
}
.stSidebar .stDownloadButton>button {
    width: 100%;
    border-radius: 8px !important;
    background-color: #27ae60 !important;
    color: white !important;
    padding: 0.5rem 1rem !important;
}
</style>
""", unsafe_allow_html=True)

# --- 主标题 & 文件上传 ---
with st.container():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center;'>智能应力预测系统</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "上传数据集（支持 Excel 或 CSV）",
        type=["csv", "xlsx"],
        key="data_uploader"
    )
    st.markdown('</div>', unsafe_allow_html=True)

if not uploaded_file:
    st.stop()

# --- 数据加载 ---
try:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"数据加载失败：{e}")
    st.stop()

# 去除多余空列
df.dropna(axis=1, how='all', inplace=True)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# --- 数据预览 ---
with st.container():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("数据预览")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 侧边栏：列映射 + 模型配置 ---
st.sidebar.header("模型配置")

# 1. 列映射
all_cols = df.columns.tolist()
target_col = st.sidebar.selectbox("目标列 (Label)", options=all_cols)
feature_cols = st.sidebar.multiselect(
    "特征列 (Features)",
    options=[c for c in all_cols if c != target_col],
    default=[c for c in all_cols if c != target_col]
)
if not feature_cols:
    st.sidebar.error("请至少选择一个特征列。")
    st.stop()

# 2. 预测模式
prediction_mode = st.sidebar.radio("预测模式", ["测试集预测", "全量数据预测"])

# 3. 模型选择
model_choice = st.sidebar.selectbox("选择模型算法", ["随机森林", "XGBoost", "SVM", "KNN"])

# 4. 超参数设置
with st.sidebar.expander("超参数设置", expanded=False):
    if model_choice == "随机森林":
        rf_n_estimators      = st.slider("树数量 (n_estimators)", 50, 500, 100)
        rf_max_depth         = st.slider("最大深度 (max_depth)", 0, 30, 0)
        rf_min_samples_split = st.slider("最小分裂样本数 (min_samples_split)", 2, 20, 2)
        rf_min_samples_leaf  = st.slider("最小叶节点样本数 (min_samples_leaf)", 1, 10, 1)
    elif model_choice == "XGBoost":
        xgb_n_estimators = st.slider("树数量 (n_estimators)", 50, 500, 100)
        xgb_max_depth    = st.slider("最大深度 (max_depth)", 1, 15, 6)
        xgb_learning_rate= st.slider("学习率 (learning_rate)", 0.01, 0.5, 0.1, step=0.01)
        xgb_reg_alpha    = st.slider("L1 正则 (reg_alpha)", 0.0, 1.0, 0.0, step=0.1)
        xgb_reg_lambda   = st.slider("L2 正则 (reg_lambda)", 0.0, 2.0, 1.0, step=0.1)
    elif model_choice == "SVM":
        svm_C       = st.slider("C (惩罚项)", 0.1, 10.0, 1.0, step=0.1)
        svm_epsilon = st.slider("ε (epsilon)", 0.01, 1.0, 0.1, step=0.01)
        svm_kernel  = st.selectbox("核函数 (kernel)", ["rbf", "linear", "poly", "sigmoid"])
    else:  # KNN
        knn_n_neighbors = st.slider("邻居数 (n_neighbors)", 1, 20, 5)
        knn_weights     = st.selectbox("权重方式 (weights)", ["uniform", "distance"])
        knn_algorithm   = st.selectbox("算法 (algorithm)", ["auto", "ball_tree", "kd_tree", "brute"])

# 5. 开始训练按钮
if st.sidebar.button("▶️ 开始训练"):
    # 特征与标签
    X = df[feature_cols]
    y = df[target_col]

    # 划分
    if prediction_mode == "测试集预测":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y

    # 构建模型
    if model_choice == "随机森林":
        model = RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth or None,
            min_samples_split=rf_min_samples_split,
            min_samples_leaf=rf_min_samples_leaf,
            random_state=42
        )
    elif model_choice == "XGBoost":
        model = XGBRegressor(
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            reg_alpha=xgb_reg_alpha,
            reg_lambda=xgb_reg_lambda,
            use_label_encoder=False,
            verbosity=0,
            random_state=42
        )
    elif model_choice == "SVM":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(C=svm_C, epsilon=svm_epsilon, kernel=svm_kernel))
        ])
    else:  # KNN
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(
                n_neighbors=knn_n_neighbors,
                weights=knn_weights,
                algorithm=knn_algorithm))
        ])

    # 训练与预测
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 评估
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    # 存储会话状态
    st.session_state.update({
        "model_trained": True,
        "model": model,
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test,   "y_test": y_test,
        "y_pred": y_pred,
        "mse": mse, "r2": r2
    })

# --- 右侧结果展示 ---
if st.session_state.get("model_trained"):
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("模型评估指标")
        col1, col2 = st.columns(2)
        col1.metric("均方误差 MSE", f"{st.session_state['mse']:.4f}")
        col2.metric("决定系数 R²", f"{st.session_state['r2']:.4f}")

        # 图表选择
        charts = ["预测值 vs 真实值", "残差分布", "学习曲线"]
        # 特征重要性可选
        if hasattr(st.session_state["model"], "feature_importances_"):
            charts.append("特征重要性")
        chart_type = st.radio("选择图表", charts, horizontal=True)

        # 准备图表
        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]

        if chart_type == "预测值 vs 真实值":
            fig = px.scatter(
                x=y_test, y=y_pred,
                labels={"x": "真实值", "y": "预测值"},
                title="预测值 vs 真实值",
                template="plotly_white")
            # 加参考线
            minv = float(np.min([y_test.min(), y_pred.min()]))
            maxv = float(np.max([y_test.max(), y_pred.max()]))
            fig.add_shape(
                type="line", x0=minv, y0=minv, x1=maxv, y1=maxv,
                line=dict(color="red", dash="dash"))
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "残差分布":
            res = y_test - y_pred
            fig = px.histogram(
                x=res, nbins=30,
                labels={"x": "残差", "y": "频数"},
                title="残差分布",
                template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "学习曲线":
            # 仅在测试集模式下可用
            if prediction_mode == "测试集预测":
                train_sizes, train_scores, test_scores = learning_curve(
                    st.session_state["model"],
                    st.session_state["X_train"],
                    st.session_state["y_train"],
                    cv=5, scoring="neg_mean_squared_error",
                    train_sizes=np.linspace(0.1, 1.0, 5)
                )
                train_err = -np.mean(train_scores, axis=1)
                test_err  = -np.mean(test_scores,  axis=1)
                lc = go.Figure()
                lc.add_trace(go.Scatter(
                    x=train_sizes, y=train_err,
                    mode="lines+markers", name="训练误差"))
                lc.add_trace(go.Scatter(
                    x=train_sizes, y=test_err,
                    mode="lines+markers", name="验证误差"))
                lc.update_layout(
                    title="学习曲线",
                    xaxis_title="训练样本数",
                    yaxis_title="均方误差",
                    template="plotly_white")
                st.plotly_chart(lc, use_container_width=True)
            else:
                st.warning("学习曲线仅在“测试集预测”模式下可用。")

        else:  # 特征重要性
            importances = st.session_state["model"].feature_importances_
            feats = st.session_state["X_train"].columns
            idx = np.argsort(importances)
            fig = go.Figure(go.Bar(
                x=importances[idx], y=feats[idx], orientation="h"))
            fig.update_layout(
                title="特征重要性",
                xaxis_title="重要性得分",
                yaxis_title="特征",
                template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        # 导出结果为 Excel，保证中文列名不乱码
        result_df = pd.DataFrame({
            "真实值": st.session_state["y_test"].values,
            "预测值": st.session_state["y_pred"]
        })
        # 如果有残差列
        result_df["残差"] = result_df["真实值"] - result_df["预测值"]
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            result_df.to_excel(writer, index=False, sheet_name="结果")
        output.seek(0)
        st.download_button(
            label="📥 下载结果 (Excel)",
            data=output,
            file_name="prediction_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.markdown('</div>', unsafe_allow_html=True)
