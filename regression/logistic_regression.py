# File: pages/2_Logistic_Regression.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, roc_curve, auc
)

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
    **Logistic Regression** is a classification algorithm used to predict a categorical outcome variable from one or more predictor variables.

    Unlike linear regression, which predicts a continuous value, logistic regression models the **probability** that an instance belongs to a certain class. This probability is then mapped to a categorical outcome.

    The model uses the logistic function (also known as the sigmoid function) to transform the output into a probability between 0 and 1.
    """)
    st.latex(r''' P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \dots + \beta_p X_p)}} ''')
    st.markdown("""
    **Key Use Cases**:
    - Disease Prediction
    - Quality Control
    - Customer Churn

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
            labels = sorted(df[y_axis_log].dropna().unique())

            if unique_y > 2:
                selected_classes = st.multiselect("Select two classes for binary classification", options=labels, default=labels[:2])
                if len(selected_classes) != 2:
                    st.warning("Please select exactly two classes.")
                    st.stop()
                df = df[df[y_axis_log].isin(selected_classes)]
                labels = sorted(selected_classes)

        with col2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.error("❌ At least one numeric column is required for the feature variable(s).")
                st.stop()

            x_axis_log = st.multiselect("Choose your X-axis (Feature(s) - must be numeric)", numeric_cols, default=numeric_cols[0] if numeric_cols else None)
            if not x_axis_log:
                st.warning("Please select at least one feature column.")
                st.stop()

        use_class_weight = st.checkbox("Balance classes automatically (use class_weight='balanced')", value=True)
        use_train_test_split = st.checkbox("Use train/test split (70/30)", value=True)

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
    label_map = {labels[0]: 0, labels[1]: 1}
    y = y_raw.map(label_map)

    st.write("### Class Distribution")
    st.write(y.value_counts().rename(index={0: labels[0], 1: labels[1]}))

    # Split data
    if use_train_test_split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    else:
        X_train = X_test = X
        y_train = y_test = y

    try:
        model = LogisticRegression(class_weight='balanced' if use_class_weight else None)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        st.subheader("📊 Key Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{acc:.4f}")
        with col2:
            st.metric("Precision", f"{prec:.4f}")
        with col3:
            st.metric("Recall", f"{rec:.4f}")

        st.subheader("📋 Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                    xticklabels=[f'Pred {labels[0]}', f'Pred {labels[1]}'],
                    yticklabels=[f'Actual {labels[0]}', f'Actual {labels[1]}'])
        st.pyplot(fig_cm)

        st.subheader("📉 Probability Plot")
        if len(x_axis_log) == 1:
            feature_name = x_axis_log[0]
            x_jittered = log_df[feature_name] + np.random.normal(0, 0.1, size=len(log_df))
            fig_log = go.Figure()
            fig_log.add_trace(go.Scatter(
                x=x_jittered,
                y=y,
                mode='markers',
                name='Data',
                marker=dict(color='rgba(0, 100, 255, 0.5)', size=8, line=dict(width=1, color='blue'))
            ))

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
            st.info("Probability plot only available for one feature variable.")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        st.subheader("📉 ROC Curve")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        fig_roc.update_layout(
            title=f'ROC Curve (AUC = {roc_auc:.2f})',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
