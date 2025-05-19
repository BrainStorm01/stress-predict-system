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

from openai import OpenAI  
import openai               

# 页面配置
st.set_page_config(
    page_title="智能应力预测系统", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 辅助函数
def call_chatgpt_api(question: str, result_df: pd.DataFrame) -> str:
    """
    基于前10行预测结果，调用 OpenAI v1 接口返回分析回答，
    并捕获常见的速率限制错误。
    """
    # 检查 Secret
    if "OPENAI_API_KEY" not in st.secrets:
        return "⚠️ 未配置 OPENAI_API_KEY，无法使用智能问答。"
    # 初始化客户端
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    # 构造 prompt
    sample = result_df.head(10).to_string(index=False)
    prompt = (
        "下面是模型的部分预测结果（前10行）：\n"
        f"{sample}\n\n请基于这些结果回答：{question}"
    )

    try:
        # 发起请求
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是智能应力预测系统的分析助手。"},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.2
        )
        return resp.choices[0].message.content

    except openai.RateLimitError:
        # 速率限制
        return "⚠️ 请求过于频繁，已达到速率限制，请稍后再试。"
    except Exception as e:
        # 捕获其它可能的错误
        return f"❌ 调用问答接口失败：{e}"

# 全局CSS样式
st.markdown("""
<style>
    /* 全局字体和颜色设置 */
    *:not(code, pre) {
        font-family: 'PingFang SC', 'Microsoft YaHei', 'Helvetica Neue', sans-serif !important;
    }
    
    /* 主题色调 */
    :root {
        --primary-color: #2E7BEE;
        --secondary-color: #36D399;
        --accent-color: #F6C549;
        --background-color: #F9FAFC;
        --card-bg-color: #FFFFFF;
        --text-color: #1D273B;
        --error-color: #E14B59;
        --warning-color: #F5A962;
    }
    
    /* 页面背景 */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: var(--background-color);
    }
    
    /* 卡片样式 */
    .card {
        background: var(--card-bg-color);
        border-radius: 12px;
        padding: 1.8rem;
        margin: 1.2rem 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .card:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }
    
    /* 标题样式 */
    h1 {
        color: var(--text-color);
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        margin-bottom: 1.5rem !important;
        text-align: center;
    }
    h2, h3, .subheader {
        color: var(--text-color);
        font-weight: 600 !important;
    }
    
    /* 按钮样式 */
    button[kind="primary"], .stButton>button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.2rem !important;
        font-weight: 500 !important;
        border: none !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"]:hover, .stButton>button:hover {
        background-color: #1A66D6 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
        transform: translateY(-1px);
    }
    
    /* 侧边栏样式 */
    .css-1d391kg, .css-12oz5g7 {
        background-color: #F2F5F9;
        border-right: 1px solid rgba(0,0,0,0.05);
    }
    .stSidebar .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .stSidebar .stDownloadButton>button {
        width: 100%;
        background-color: var(--secondary-color) !important;
        color: white !important;
        border-radius: 8px !important;
    }
    
    /* 下载按钮样式 */
    .stDownloadButton>button {
        background-color: var(--secondary-color) !important;
        color: white !important;
    }
    
    /* 文件上传器样式 */
    .stFileUploader > div > button {
        background-color: var(--primary-color);
        color: white;
    }
    .stFileUploader > div {
        border: 2px dashed rgba(0,0,0,0.1);
        border-radius: 10px;
        padding: 2rem;
        background: rgba(0,0,0,0.02);
    }
    
    /* 指标样式 */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4ecfb 100%);
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid rgba(0,0,0,0.05);
    }
    [data-testid="stMetricLabel"] {
        font-weight: 500;
    }
    [data-testid="stMetricValue"] {
        font-weight: 700;
        color: var(--primary-color);
    }
    
    /* 图表容器样式 */
    .chart-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    
    /* 表格样式 */
    [data-testid="stTable"] {
        background: white;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* 提示样式 */
    .banner {
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .success-banner {
        background-color: rgba(54, 211, 153, 0.1);
        border-left: 4px solid var(--secondary-color);
        color: var(--text-color);
    }
    .warning-banner {
        background-color: rgba(245, 169, 98, 0.1);
        border-left: 4px solid var(--warning-color);
        color: var(--text-color);
    }
    
    /* 输入框样式 */
    input[type="text"] {
        border-radius: 6px !important;
        border: 1px solid rgba(0,0,0,0.1) !important;
        padding: 0.6rem 1rem !important;
    }
    
    /* 滑块样式 */
    .stSlider div[data-baseweb="slider"] {
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    .stSlider [data-testid="stTickBarMax"], .stSlider [data-testid="stTickBarMin"] {
        color: var(--text-color);
    }
    
    /* 单选按钮样式 */
    .stRadio [role="radiogroup"] {
        background-color: white;
        border-radius: 8px;
        padding: 0.5rem;
        border: 1px solid rgba(0,0,0,0.05);
        display: flex;
        justify-content: center;
    }
    
    /* 分割线 */
    hr {
        margin: 2rem 0;
        border-top: 1px solid rgba(0,0,0,0.1);
    }
    
    /* 动画效果 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-in {
        animation: fadeIn 0.5s ease forwards;
    }
    
    /* 响应式调整 */
    @media (max-width: 992px) {
        .card {
            padding: 1.2rem;
        }
        h1 {
            font-size: 2rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# 主标题和欢迎区域
st.markdown('<div class="card animate-in">', unsafe_allow_html=True)

# 创建标题行
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("<h1>智能应力预测系统 <span style='font-size:1.5rem'>📊</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#5f6c7b;margin-top:-1rem;margin-bottom:1.5rem;'>上传数据集，使用机器学习模型进行精确预测</p>", unsafe_allow_html=True)

# 文件上传区
upload_container = st.container()
with upload_container:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "上传数据集（支持 Excel 或 CSV）",
            type=["csv", "xlsx"],
            key="data_uploader"
        )

# 添加说明信息（仅当未上传文件时显示）
if not uploaded_file:
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0;color:#5f6c7b;">
        <p>👆 请先上传数据文件</p>
        <ul style="list-style-type:none;padding:0;margin-top:1.5rem;">
            <li style="display:inline-block;margin:0 1rem;"><span style="color:#2E7BEE;font-weight:500;">✓</span> 支持CSV和Excel格式</li>
            <li style="display:inline-block;margin:0 1rem;"><span style="color:#2E7BEE;font-weight:500;">✓</span> 自动识别特征和目标列</li>
            <li style="display:inline-block;margin:0 1rem;"><span style="color:#2E7BEE;font-weight:500;">✓</span> 多种机器学习模型</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
st.markdown('</div>', unsafe_allow_html=True)

# 如果未上传文件，停止执行
if not uploaded_file:
    st.stop()

# 数据加载
try:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
        
    # 去除多余空列
    df.dropna(axis=1, how='all', inplace=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
except Exception as e:
    st.error(f"数据加载失败：{e}")
    st.stop()

# 数据预览
st.markdown('<div class="card animate-in">', unsafe_allow_html=True)
st.subheader("📋 数据预览")

# 添加数据统计信息
col1, col2, col3 = st.columns(3)
col1.metric("数据行数", f"{len(df)}")
col2.metric("数据列数", f"{df.shape[1]}")
col3.metric("数值型列数", f"{df.select_dtypes(include=['number']).shape[1]}")

# 显示数据表
st.dataframe(df.head(10), use_container_width=True)

# 添加数据预处理提示
if df.isnull().values.any():
    st.markdown("""
    <div class="banner warning-banner">
        <p>⚠️ 数据中包含缺失值，可能影响模型训练效果。建议在上传前进行缺失值处理。</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# 侧边栏模型配置
st.sidebar.markdown("""
<h3 style="margin-bottom:1.5rem;color:#1D273B;font-weight:600;"><span style="color:#2E7BEE">⚙️</span> 模型配置</h3>
""", unsafe_allow_html=True)

# 创建分隔部分的函数
def sidebar_section(title, icon="🔹"):
    st.sidebar.markdown(f"""
    <div style="margin-top:1.5rem;margin-bottom:0.8rem;padding-bottom:0.5rem;border-bottom:1px solid rgba(0,0,0,0.1);">
        <span style="color:#2E7BEE;font-weight:600;">{icon}</span>
        <span style="font-weight:600;font-size:1.05rem;color:#1D273B;margin-left:0.5rem;">{title}</span>
    </div>
    """, unsafe_allow_html=True)

# 1. 列映射
sidebar_section("数据配置", "📊")

# 列映射
all_cols = df.columns.tolist()
target_col = st.sidebar.selectbox(
    "目标列 (Label)", 
    options=all_cols,
    help="选择要预测的目标变量列"
)

feature_cols = st.sidebar.multiselect(
    "特征列 (Features)",
    options=[c for c in all_cols if c != target_col],
    default=[c for c in all_cols if c != target_col],
    help="选择用于训练模型的特征列"
)

if not feature_cols:
    st.sidebar.error("请至少选择一个特征列。")
    st.stop()

# 2. 预测模式
sidebar_section("预测配置", "🎯")
prediction_mode = st.sidebar.radio(
    "预测模式", 
    ["测试集预测", "全量数据预测"],
    help="测试集预测会自动分割训练集和测试集，全量数据预测将使用所有数据训练和预测"
)

# 3. 模型选择
sidebar_section("模型选择", "🧠")
model_choice = st.sidebar.selectbox(
    "选择算法", 
    ["随机森林", "XGBoost", "SVM", "KNN"],
    help="选择适合您数据的机器学习算法"
)

# 4. 调参方式
sidebar_section("训练设置", "⚡")
tune_mode = st.sidebar.radio(
    "调参方式",
    ["手动调参", "自动调参"],
    horizontal=True,
    key="tune_mode",
    help="手动调参允许您自定义参数，自动调参将系统自动寻找最优参数"
)

if tune_mode == "自动调参":
    search_strategy = st.sidebar.selectbox(
        "自动调参策略",
        ["网格搜索(GridSearchCV)", "随机搜索(RandomizedSearchCV)", "贝叶斯优化(Optuna)"],
        key="search_strategy",
        help="选择不同的超参数搜索策略，贝叶斯优化通常效率最高"
    )

# 5. 超参数设置
with st.sidebar.expander("超参数设置", expanded=False):
    if tune_mode == "手动调参":
        if model_choice == "随机森林":
            st.markdown('<div style="margin-bottom:1rem;padding-bottom:0.5rem;border-bottom:1px solid rgba(0,0,0,0.05);">', unsafe_allow_html=True)
            st.markdown("🌲 **随机森林参数**", unsafe_allow_html=True)
            rf_n_estimators = st.slider("树数量 (n_estimators)", 50, 500, 100, help="森林中决策树的数量，通常越多越好，但会增加计算量")
            rf_max_depth = st.slider("最大深度 (max_depth)", 0, 30, 0, help="树的最大深度，0表示不限制")
            rf_min_samples_split = st.slider("最小分裂样本数", 2, 20, 2, help="分裂内部节点所需的最小样本数")
            rf_min_samples_leaf = st.slider("最小叶节点样本数", 1, 10, 1, help="叶节点上所需的最小样本数")
            st.markdown('</div>', unsafe_allow_html=True)

        elif model_choice == "XGBoost":
            st.markdown('<div style="margin-bottom:1rem;padding-bottom:0.5rem;border-bottom:1px solid rgba(0,0,0,0.05);">', unsafe_allow_html=True)
            st.markdown("🚀 **XGBoost参数**", unsafe_allow_html=True)
            xgb_n_estimators = st.slider("树数量 (n_estimators)", 50, 500, 100, help="提升迭代的次数")
            xgb_max_depth = st.slider("最大深度 (max_depth)", 1, 15, 6, help="树的最大深度")
            xgb_learning_rate = st.slider("学习率 (learning_rate)", 0.01, 0.5, 0.1, step=0.01, help="每次迭代对权重的贡献率，较小的值使训练更稳健")
            xgb_reg_alpha = st.slider("L1 正则 (reg_alpha)", 0.0, 1.0, 0.0, step=0.1, help="L1正则化项，控制模型复杂度")
            xgb_reg_lambda = st.slider("L2 正则 (reg_lambda)", 0.0, 2.0, 1.0, step=0.1, help="L2正则化项，控制模型复杂度")
            st.markdown('</div>', unsafe_allow_html=True)

        elif model_choice == "SVM":
            st.markdown('<div style="margin-bottom:1rem;padding-bottom:0.5rem;border-bottom:1px solid rgba(0,0,0,0.05);">', unsafe_allow_html=True)
            st.markdown("🔄 **SVM参数**", unsafe_allow_html=True)
            svm_C = st.slider("C (惩罚项)", 0.1, 10.0, 1.0, step=0.1, help="错误项的惩罚系数，较大的值表示更严格的正则化")
            svm_epsilon = st.slider("ε (epsilon)", 0.01, 1.0, 0.1, step=0.01, help="ε-SVR中的ε参数，定义了不受惩罚的区域")
            svm_kernel = st.selectbox("核函数 (kernel)", ["rbf", "linear", "poly", "sigmoid"], help="指定算法中使用的核函数类型")
            st.markdown('</div>', unsafe_allow_html=True)

        else:  # KNN
            st.markdown('<div style="margin-bottom:1rem;padding-bottom:0.5rem;border-bottom:1px solid rgba(0,0,0,0.05);">', unsafe_allow_html=True)
            st.markdown("🔍 **KNN参数**", unsafe_allow_html=True)
            knn_n_neighbors = st.slider("邻居数 (n_neighbors)", 1, 20, 5, help="k值，用于决定预测时考虑的最近邻居数")
            knn_weights = st.selectbox("权重方式 (weights)", ["uniform", "distance"], help="预测时的权重计算方式")
            knn_algorithm = st.selectbox("算法 (algorithm)", ["auto", "ball_tree", "kd_tree", "brute"], help="计算最近邻居的算法")
            st.markdown('</div>', unsafe_allow_html=True)

# 6. 训练按钮
train_button = st.sidebar.button(
    "▶️ 开始训练模型",
    key="train_button",
    help="点击开始训练模型"
)

# 模型训练过程
if train_button:
    # 特征与标签
    X = df[feature_cols]
    y = df[target_col]

    # 划分数据集
    if prediction_mode == "测试集预测":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y

    # 自动调参训练分支
    if tune_mode == "自动调参":
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        import importlib
        if importlib.util.find_spec("optuna") is None:
            import subprocess
            subprocess.run(["pip", "install", "optuna"])
        import optuna

        param_grid = {}
        if model_choice == "随机森林":
            base_model = RandomForestRegressor(random_state=42)
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        elif model_choice == "XGBoost":
            base_model = XGBRegressor(random_state=42, use_label_encoder=False, verbosity=0)
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.1, 0.2],
                "reg_alpha": [0, 0.5, 1.0],
                "reg_lambda": [0, 1.0, 2.0]
            }
        elif model_choice == "SVM":
            base_model = SVR()
            param_grid = {
                "C": [0.1, 1, 10],
                "epsilon": [0.01, 0.1, 0.5],
                "kernel": ["rbf", "linear"]
            }
        elif model_choice == "KNN":
            base_model = KNeighborsRegressor()
            param_grid = {
                "n_neighbors": [3, 5, 10],
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree"]
            }
        
        # 选择搜索策略
        if search_strategy == "网格搜索(GridSearchCV)":
            search = GridSearchCV(
                base_model, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1
            )
            with st.spinner("网格搜索自动调参中..."):
                search.fit(X_train, y_train)
            st.success(f"最优参数：{search.best_params_}")
            model = search.best_estimator_
        elif search_strategy == "随机搜索(RandomizedSearchCV)":
            search = RandomizedSearchCV(
                base_model, param_grid, n_iter=10, cv=3, scoring="neg_mean_squared_error", n_jobs=-1, random_state=42
            )
            with st.spinner("随机搜索自动调参中..."):
                search.fit(X_train, y_train)
            st.success(f"最优参数：{search.best_params_}")
            model = search.best_estimator_
        else:  # Optuna
            def objective(trial):
                if model_choice == "随机森林":
                    model = RandomForestRegressor(
                        n_estimators=trial.suggest_int("n_estimators", 50, 200),
                        max_depth=trial.suggest_int("max_depth", 3, 20),
                        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 4),
                        random_state=42
                    )
                elif model_choice == "XGBoost":
                    model = XGBRegressor(
                        n_estimators=trial.suggest_int("n_estimators", 50, 200),
                        max_depth=trial.suggest_int("max_depth", 3, 10),
                        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
                        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 2.0),
                        use_label_encoder=False, verbosity=0, random_state=42
                    )
                elif model_choice == "SVM":
                    model = SVR(
                        C=trial.suggest_float("C", 0.1, 10.0),
                        epsilon=trial.suggest_float("epsilon", 0.01, 0.5),
                        kernel=trial.suggest_categorical("kernel", ["rbf", "linear"])
                    )
                elif model_choice == "KNN":
                    model = KNeighborsRegressor(
                        n_neighbors=trial.suggest_int("n_neighbors", 3, 10),
                        weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
                        algorithm=trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree"])
                    )
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                return mean_squared_error(y_test, pred)
                
            study = optuna.create_study(direction="minimize")
            with st.spinner("贝叶斯优化(Optuna)自动调参中..."):
                study.optimize(objective, n_trials=20)
            best_params = study.best_params
            st.success(f"Optuna 贝叶斯优化完成，最佳参数：{best_params}")
            
            # 使用最优参数初始化模型
            if model_choice == "随机森林":
                model = RandomForestRegressor(**best_params, random_state=42)
            elif model_choice == "XGBoost":
                model = XGBRegressor(**best_params, use_label_encoder=False, verbosity=0, random_state=42)
            elif model_choice == "SVM":
                model = SVR(**best_params)
            elif model_choice == "KNN":
                model = KNeighborsRegressor(**best_params)

    else:
        # 手动调参分支
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
    with st.spinner("模型训练中..."):
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
        "mse": mse, "r2": r2,
        "prediction_mode": prediction_mode
    })

# 结果展示部分
if st.session_state.get("model_trained"):
    st.markdown('<div class="card animate-in">', unsafe_allow_html=True)
    st.subheader("📈 模型评估与结果")
    
    # 模型评估指标
    col1, col2 = st.columns(2)
    col1.metric("均方误差 (MSE)", f"{st.session_state['mse']:.4f}", 
                help="均方误差，值越小表示模型预测越准确")
    col2.metric("决定系数 (R²)", f"{st.session_state['r2']:.4f}", 
               help="决定系数，值越接近1表示模型拟合效果越好")
    
    # 修改图表选择部分
    st.markdown("""
    <div style="margin:1.5rem 0 1rem 0;">
        <div style="font-weight:600;margin-bottom:0.5rem;color:#1D273B;">选择可视化图表</div>
    </div>
    """, unsafe_allow_html=True)
    
    charts = ["预测值 vs 真实值", "残差分布", "学习曲线"]
    if hasattr(st.session_state["model"], "feature_importances_"):
        charts.append("特征重要性")
    
    chart_type = st.radio(
        "选择图表类型",  # 添加标签
        charts, 
        horizontal=True, 
        key="chart_type",
        label_visibility="collapsed"  # 隐藏标签但保持可访问性
    )
    
    # 图表容器
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # 准备数据
    y_test = st.session_state["y_test"]
    y_pred = st.session_state["y_pred"]
    
    # 根据选择渲染不同图表
    if chart_type == "预测值 vs 真实值":
        fig = px.scatter(
            x=y_test, y=y_pred,
            labels={"x": "真实值", "y": "预测值"},
            title="预测值 vs 真实值对比",
            template="plotly_white"
        )
        # 添加参考线
        min_val = float(np.min([y_test.min(), y_pred.min()]))
        max_val = float(np.max([y_test.max(), y_pred.max()]))
        fig.add_shape(
            type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
            line=dict(color="#2E7BEE", dash="dash", width=2)
        )
        # 自定义样式
        fig.update_traces(
            marker=dict(
                size=8,
                color="#36D399",
                line=dict(width=1, color="#FFFFFF")
            )
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 添加解释
        st.markdown("""
        <div style="font-size:0.9rem;color:#5f6c7b;margin-top:0.5rem;">
            点越接近参考线，表示预测越准确。散点分布越集中，表示模型预测稳定性越好。
        </div>
        """, unsafe_allow_html=True)
        
    elif chart_type == "残差分布":
        try:
            # 计算残差
            residuals = y_test - y_pred
            
            # 创建基础图表
            fig = go.Figure()
            
            # 添加直方图
            fig.add_trace(go.Histogram(
                x=residuals,
                nbinsx=30,
                name="残差分布",
                marker_color="#2E7BEE",
                opacity=0.7
            ))
            
            # 计算正态分布曲线
            mean = np.mean(residuals)
            std = np.std(residuals)
            x_norm = np.linspace(min(residuals), max(residuals), 100)
            y_norm = np.exp(-0.5 * ((x_norm - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
            y_norm = y_norm * len(residuals) * (max(residuals) - min(residuals)) / 30
            
            # 添加正态分布曲线
            fig.add_trace(go.Scatter(
                x=x_norm,
                y=y_norm,
                mode="lines",
                name="正态分布拟合",
                line=dict(color="#F6C549", width=2)
            ))
            
            # 添加零线 - 使用固定值而不是依赖fig.data
            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=0,
                y1=max(y_norm) * 1.1,  # 使用正态分布曲线的最大值，并增加10%的余量
                line=dict(color="#E14B59", dash="dash", width=2)
            )
            
            # 更新布局
            fig.update_layout(
                title="残差分布",
                xaxis_title="残差",
                yaxis_title="频数",
                template="plotly_white",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=50, b=20),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )
            
            # 显示图表
            st.plotly_chart(fig, use_container_width=True)
            
            # 添加分析
            st.markdown(f"""
            <div style="font-size:0.9rem;color:#5f6c7b;margin-top:0.5rem;">
                残差平均值：<span style="font-weight:500">{mean:.4f}</span>, 
                标准差：<span style="font-weight:500">{std:.4f}</span>。
                理想状态下，残差应该呈正态分布，并且均值接近于0。
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"生成残差分布图时出错：{str(e)}")
            st.info("请确保数据已正确加载并完成模型训练。")
        
    elif chart_type == "学习曲线":
        # 仅在测试集模式下有效
        if st.session_state.get("prediction_mode") == "测试集预测":
            # 计算学习曲线
            with st.spinner("计算学习曲线..."):
                train_sizes, train_scores, test_scores = learning_curve(
                    st.session_state["model"],
                    st.session_state["X_train"],
                    st.session_state["y_train"],
                    cv=5, scoring="neg_mean_squared_error",
                    train_sizes=np.linspace(0.1, 1.0, 5)
                )
            
            # 转换为错误值
            train_errors = -np.mean(train_scores, axis=1)
            test_errors = -np.mean(test_scores, axis=1)
            train_errors_std = np.std(train_scores, axis=1)
            test_errors_std = np.std(test_scores, axis=1)
            
            # 创建学习曲线
            fig = go.Figure()
            
            # 添加训练误差
            fig.add_trace(go.Scatter(
                x=train_sizes, y=train_errors,
                mode="lines+markers",
                name="训练误差",
                line=dict(color="#2E7BEE", width=2),
                marker=dict(size=8)
            ))
            
            # 添加验证误差
            fig.add_trace(go.Scatter(
                x=train_sizes, y=test_errors,
                mode="lines+markers",
                name="验证误差",
                line=dict(color="#F6C549", width=2),
                marker=dict(size=8)
            ))
            
            # 添加训练误差范围
            fig.add_trace(go.Scatter(
                x=np.concatenate([train_sizes, train_sizes[::-1]]),
                y=np.concatenate([train_errors - train_errors_std, 
                                 (train_errors + train_errors_std)[::-1]]),
                fill="toself",
                fillcolor="rgba(46, 123, 238, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False
            ))
            
            # 添加验证误差范围
            fig.add_trace(go.Scatter(
                x=np.concatenate([train_sizes, train_sizes[::-1]]),
                y=np.concatenate([test_errors - test_errors_std, 
                                 (test_errors + test_errors_std)[::-1]]),
                fill="toself",
                fillcolor="rgba(246, 197, 73, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False
            ))
            
            # 更新布局
            fig.update_layout(
                title="学习曲线 (误差随训练样本数的变化)",
                xaxis_title="训练样本数",
                yaxis_title="均方误差 (MSE)",
                template="plotly_white",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=70, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 添加说明
            gap = test_errors[-1] - train_errors[-1]
            if gap > 0.2 * test_errors[-1]:
                conclusion = "模型可能存在过拟合，可以尝试增加正则化或减少模型复杂度。"
            elif train_errors[-1] > 0.7 * test_errors[-1]:
                conclusion = "模型可能存在欠拟合，可以尝试增加模型复杂度或特征工程。"
            else:
                conclusion = "模型拟合程度良好，训练和验证误差较为接近。"
            
            st.markdown(f"""
            <div style="font-size:0.9rem;color:#5f6c7b;margin-top:0.5rem;">
                <p><strong>分析：</strong> {conclusion}</p>
                <p>训练样本增加时，如果验证误差仍在下降，增加更多数据可能会进一步提高模型性能。</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.info("学习曲线仅在'测试集预测'模式下可用。请切换到测试集预测模式后重新训练模型。")
            
    else:  # 特征重要性
        importances = st.session_state["model"].feature_importances_
        features = st.session_state["X_train"].columns.tolist()
        
        # 排序
        importance_df = pd.DataFrame({
            "特征": features,
            "重要性": importances
        })
        importance_df = importance_df.sort_values("重要性", ascending=True)
        
        # 创建水平条形图
        fig = px.bar(
            importance_df, x="重要性", y="特征",
            orientation="h",
            title="特征重要性排名",
            template="plotly_white",
            color="重要性",
            color_continuous_scale=["#c4ddff", "#2E7BEE"]
        )
        
        # 更新布局
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 添加分析
        top_features = importance_df.iloc[-3:]["特征"].tolist()
        st.markdown(f"""
        <div style="font-size:0.9rem;color:#5f6c7b;margin-top:0.5rem;">
            <p>最重要的三个特征是：<span style="font-weight:500">{", ".join(top_features)}</span>。
            这些特征对模型预测结果影响最大，可以重点关注。</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 预测结果表格与下载
    st.markdown("""
    <div style="margin:2rem 0 1rem 0;">
        <div style="font-weight:600;margin-bottom:0.5rem;color:#1D273B;">预测结果</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 创建结果数据框
    result_df = pd.DataFrame({
        "真实值": st.session_state["y_test"].values,
        "预测值": st.session_state["y_pred"],
        "残差": st.session_state["y_test"].values - st.session_state["y_pred"],
        "相对误差 (%)": abs((st.session_state["y_test"].values - st.session_state["y_pred"]) / st.session_state["y_test"].values) * 100
    })
    
    # 显示前10行结果
    st.dataframe(result_df.head(10).style.format({
        "真实值": "{:.4f}",
        "预测值": "{:.4f}",
        "残差": "{:.4f}",
        "相对误差 (%)": "{:.2f}%"
    }), use_container_width=True)
    
    # 导出结果按钮
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False, sheet_name="预测结果")
    output.seek(0)
    
    st.download_button(
        label="📥 下载完整预测结果 (Excel)",
        data=output,
        file_name="prediction_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # 添加智能问答功能
    st.markdown("""
    <div style="margin:2rem 0 1rem 0;">
        <div style="font-weight:600;margin-bottom:0.5rem;color:#1D273B;">🤖 智能问答</div>
    </div>
    """, unsafe_allow_html=True)
    
    question = st.text_input(
        "输入你的问题，例如：'哪几个样本误差最大？'", 
        key="qa_input",
        placeholder="输入你想问的问题，AI助手会基于预测结果进行分析..."
    )
    
    if st.button("提交问题", key="qa_button"):
        if not question.strip():
            st.warning("⚠️ 请输入一个问题后再提交。")
        else:
            # 调用ChatGPT
            with st.spinner("AI 助手正在思考，请稍候..."):
                answer = call_chatgpt_api(question, result_df)
            
            # 显示回答
            st.markdown("""
            <div style="background-color:#f8f9fa;border-radius:8px;padding:1rem;margin-top:1rem;border-left:4px solid #2E7BEE;">
                <div style="font-weight:500;margin-bottom:0.5rem;">AI 助手回答:</div>
                <div style="color:#1D273B;">
            """, unsafe_allow_html=True)
            
            st.markdown(answer)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
