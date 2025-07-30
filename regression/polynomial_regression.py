import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def run():
    st.header("📉 Polynomial Regression")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if not uploaded_file:
        st.info("Please upload a CSV file to proceed.")
        return

    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("The uploaded file must contain at least two numeric columns.")
        return

    x_axis = st.selectbox("Select X-axis", numeric_cols)
    y_axis = st.selectbox("Select Y-axis", numeric_cols)
    degree = st.slider("Select degree of polynomial", min_value=2, max_value=6, value=2)

    if x_axis and y_axis:
        x = df[[x_axis]].to_numpy()
        y = df[y_axis].to_numpy()

        # Fit polynomial regression
        poly_features = PolynomialFeatures(degree=degree)
        x_poly = poly_features.fit_transform(x)

        model = LinearRegression()
        model.fit(x_poly, y)
        y_pred = model.predict(x_poly)
        r_squared = r2_score(y, y_pred)

        # Sort for plotting
        sort_idx = np.argsort(x.flatten())
        x_sorted = x[sort_idx].flatten()
        y_sorted = y_pred[sort_idx]

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x.flatten(), y=y, mode='markers', name='Data'))
        fig.add_trace(go.Scatter(x=x_sorted, y=y_sorted, mode='lines', name=f'Polynomial Fit (deg {degree})'))

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**R² = {r_squared:.4f}**")
        st.markdown(f"**Polynomial Coefficients:** `{model.coef_}`")
