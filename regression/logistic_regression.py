# File: pages/2_Logistic_Regression.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Safe import of utils
try:
    from utils import apply_app_styling
except ImportError:
    st.warning("Utils module not found. Using default settings.")
    def apply_app_styling():
        pass

st.set_page_config(
    page_title="Logistic Regression",
    page_icon="➕",
    layout="wide",
)
apply_app_styling()

st.header("➕ Logistic Regression")

with st.expander("📘 What is logistic regression?", expanded=True):
    st.markdown("""
    **Logistic Regression** is a classification algorithm used to predict a categorical outcome variable from one or more predictor variables. The outcome is often binary, meaning there are only two possible outcomes, such as "yes/no," "pass/fail," or "positive/negative."

    Unlike linear regression, which predicts a continuous value, logistic regression models the **probability** that an instance belongs to a certain class. This probability is then mapped to a categorical outcome.

    The model uses the logistic function (also known as the sigmoid function) to transform the output into a probability between 0 and 1. The general form of the model is:
    """)
    st.latex(r''' P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \dots + \beta_p X_p)}} ''')
    st.markdown("""
    Where $P(Y=1|X)$ is the probability of the outcome being 1 given the input variables $X$.

    **Key Use Cases**:
    - **Disease Prediction**: Predicting if a patient has a disease (yes/no) based on symptoms.
    - **Quality Control**: Classifying a product as defective or non-defective based on manufacturing parameters.
    - **Customer Churn**: Predicting if a customer will cancel their subscription (churn/not churn) based on their usage data.

    This page allows you to perform logistic regression on a dataset with a binary outcome variable.
    """)

# ================================
# 1. DATA UPLOAD AND PREVIEW
# ================================
with st.expander("📤 Upload Your CSV File", expanded=True):
    uploaded_file = st.file_uploader(" ", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    with st.expander("📖 Data Preview", expanded=False):
        st.dataframe(df, use_container_width=True)

    # ================================
    # 2. ANALYSIS CONTROLS
    # ================================
    st.header("⚙️ Controls")
    with st.expander("🔧 Configure Analysis Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if not categorical_cols:
                st.error("❌ A categorical column is required for the outcome variable (Y-axis).")
                st.stop()
            
            y_axis_log = st.selectbox("Choose your Y-axis (Outcome - must be categorical)", categorical_cols)
            unique_y = df[y_axis_log].nunique()
            if unique_y != 2:
                st.warning(f"⚠️ The selected column has {unique_y} unique values. Logistic regression works best with a binary outcome.")
            
        with col2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.error("❌ At least one numeric column is required for the feature variable(s).")
                st.stop()
            
            x_axis_log = st.multiselect("Choose your X-axis (Feature(s) - must be numeric)", numeric_cols, default=numeric_cols[0] if numeric_cols else None)
            if not x_axis_log:
                st.warning("Please select at least one feature column.")
                st.stop()

    # ================================
    # 3. LOGISTIC REGRESSION ANALYSIS AND PLOTS
    # ================================
    st.header("📈 Logistic Regression Analysis")
    
    # Prepare data
    log_df = df.dropna(subset=[y_axis_log] + x_axis_log)
    if log_df.empty:
        st.error("❌ No valid data found in selected columns for logistic regression.")
        st.stop()
    
    X = log_df[x_axis_log]
    y_raw = log_df[y_axis_log]
    
    # Map categorical labels to 0 and 1
    labels = y_raw.unique()
    if len(labels) == 2:
        label_map = {labels[0]: 0, labels[1]: 1}
        y = y_raw.map(label_map)
        st.info(f"Using: **'{labels[0]}'** as 0 and **'{labels[1]}'** as 1.")
    else:
        st.error("❌ The selected outcome variable is not binary. Please choose a column with exactly two unique values.")
        st.stop()
    
    try:
        model = LogisticRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        # Metrics
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])

        st.subheader("📊 Key Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{acc:.4f}")
        with col2:
            st.metric("Precision", f"{prec:.4f}")
        with col3:
            st.metric("Recall", f"{rec:.4f}")

        st.subheader("📋 Confusion Matrix")
        st.dataframe(pd.DataFrame(cm, index=[f'Actual {labels[0]}', f'Actual {labels[1]}'], columns=[f'Predicted {labels[0]}', f'Predicted {labels[1]}']), use_container_width=True)

        st.subheader("📉 Probability Plot")
        if len(x_axis_log) == 1:
            fig_log = go.Figure()
            feature_name = x_axis_log[0]
            
            # Plot data points with jitter
            fig_log.add_trace(go.Scatter(
                x=log_df[feature_name],
                y=y_raw.map(lambda x: 1 if x == labels[1] else 0),
                mode='markers',
                name='Data',
                marker=dict(color='rgba(0, 100, 255, 0.5)', size=8, line=dict(width=1, color='blue')),
                jitter=0.2,
            ))

            # Plot predicted probabilities
            x_range = np.linspace(log_df[feature_name].min(), log_df[feature_name].max(), 300).reshape(-1, 1)
            y_prob_line = model.predict_proba(x_range)[:, 1]
            fig_log.add_trace(go.Scatter(
                x=x_range.flatten(),
                y=y_prob_line,
                mode='lines',
                name='Predicted Probability',
                line=dict(color='red', width=2)
            ))
            
            fig_log.update_layout(
                title=f"Logistic Regression Fit: {y_axis_log} vs {feature_name}",
                xaxis_title=feature_name,
                yaxis_title=f"Probability of '{labels[1]}'",
                yaxis_range=[-0.1, 1.1]
            )
            st.plotly_chart(fig_log, use_container_width=True)
        else:
            st.warning("Probability plot is only supported for a single feature variable.")

    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")