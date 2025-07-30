import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from utils import apply_app_styling, units_list

def run(df):
    if df is not None:
        st.subheader("📖 Data Preview:")
        st.dataframe(df)

        units = st.selectbox(
            "Select Units for Analytes",
            options=units_list, 
            index=0
        )

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_columns) < 2:
            st.error("❌ Not enough numeric columns found for X and Y selection.")
        else:
            # Keep the selections persistent
            x_axis = st.selectbox("Select the X-axis (e.g., Standard Concentration)", numeric_columns, index=0)
            y_axis = st.selectbox("Select the Y-axis (e.g., Measured Value)", numeric_columns, index=1)

            identifier_column = st.selectbox("Optional: Select an identifier column (e.g., Sample ID)", [None] + df.columns.tolist())

            st.session_state.x_axis = x_axis
            st.session_state.y_axis = y_axis
            st.session_state.identifier_column = identifier_column

            clean_df = df[[x_axis, y_axis] + ([identifier_column] if identifier_column else [])].dropna()

            if clean_df.empty:
                st.error("❌ The selected columns contain no valid numeric data.")
            else:
                try:
                    x = clean_df[x_axis].to_numpy()
                    y = clean_df[y_axis].to_numpy()

                    # Linear fit
                    slope, intercept = np.polyfit(x, y, 1)
                    fitted_values = slope * x + intercept
                    residuals = y - fitted_values
                    r_squared = 1 - (np.sum(residuals**2) / np.sum((y - np.mean(y))**2))

                    # Optional hover text if identifier column is selected
                    hover_text = (
                        clean_df[identifier_column].astype(str) + "<br>"
                        + x_axis + ": " + clean_df[x_axis].astype(str) + "<br>"
                        + y_axis + ": " + clean_df[y_axis].astype(str)
                    ) if identifier_column else None

                    # Plotting the data and linear fit
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=y,
                        mode='markers',
                        name='Data Points',
                        marker=dict(color='blue'),
                        text=hover_text,
                        hoverinfo='text' if identifier_column else 'x+y'
                    ))
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=fitted_values,
                        mode='lines',
                        name=f"Fit: y = {slope:.2f}x + {intercept:.2f}<br>R² = {r_squared:.4f}",
                        line=dict(color='red')
                    ))
                    fig.update_layout(
                        title="Standard Curve (Linear Fit)",
                        xaxis_title=f"{x_axis} ({units})",
                        yaxis_title=f"{y_axis} ({units})",
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )
                    st.plotly_chart(fig)

                    # Interpretation based on R² value
                    if r_squared >= 0.99:
                        interpretation = "Excellent linearity — the response is highly consistent with the standard concentrations."
                    elif r_squared >= 0.95:
                        interpretation = "Good linearity — results are acceptable, but further verification may be considered."
                    elif r_squared >= 0.90:
                        interpretation = "Moderate linearity — further investigation may be needed for accuracy at extreme points."
                    else:
                        interpretation = "Poor linearity — data may not be reliable for quantitative analysis."

                    st.markdown(f"🧠 **Interpretation:** \n{interpretation}")

                    # Show fitted values and residuals in a new dataframe
                    results_df = clean_df.copy()
                    results_df["Fitted Value"] = fitted_values
                    results_df["Residuals"] = residuals
                    st.dataframe(results_df)

                except np.linalg.LinAlgError:
                    st.error("❌ Linear fitting failed due to numerical instability in the data.")
                except TypeError:
                    st.error("❌ Ensure X and Y axis selections are numeric and 1D.")
