import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def cusum_analysis(data, target, k, h):
    deviations = data - target
    s_pos, s_neg = [0], [0]

    for x in deviations:
        s_pos.append(max(0, s_pos[-1] + x - k))
        s_neg.append(min(0, s_neg[-1] + x + k))

    s_pos, s_neg = s_pos[1:], s_neg[1:]
    out_of_control = [abs(p) > h or abs(n) > h for p, n in zip(s_pos, s_neg)]
    return s_pos, s_neg, out_of_control

def run():
    st.header("📈 CUSUM Test for Shift Detection")

    with st.expander("📘 What is the CUSUM Test?", expanded=False):
        st.markdown("""
        The **Cumulative Sum (CUSUM)** test detects small and persistent shifts in process mean over time.

        **Formulas:**
        - CUSUM⁺ = max(0, previous⁺ + (x - target) - k)
        - CUSUM⁻ = min(0, previous⁻ + (x - target) + k)

        **Parameters:**
        - 🎯 **Target:** The expected process mean
        - ⚙️ **k:** Reference value (half the desired shift)
        - 📉 **h:** Decision threshold for triggering a signal
        """)

    with st.expander("📘 Instructions"):
        st.markdown("""
        1. Upload your CSV file.
        2. Select a material from column 3.
        3. Choose a numeric column to analyze.
        4. Enter the **target mean**, **reference value (k)**, and **decision threshold (h)**.
        5. Click **Run CUSUM Test**.
        """)

    # File uploader
    # --- File Upload ---
    with st.expander("📤 Upload Your CSV File", expanded=True):
        st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
        uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"])
    

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Data Preview")
        st.dataframe(df)

        if df.shape[1] < 3:
            st.error("❌ The uploaded file must have at least 3 columns (Material must be in column 3).")
            return

        material_col = df.columns[3]  
        materials = df[material_col].dropna().unique().tolist()

        selected_material = st.selectbox("🧪 Select Material", sorted(materials))
        filtered_df = df[df[material_col] == selected_material]

        numeric_cols = filtered_df.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            st.warning("⚠️ No numeric columns found for selected material.")
            return

        selected_col = st.selectbox("📈 Select column for CUSUM analysis", numeric_cols)
        target = st.number_input("🎯 Target mean", value=0.0)
        k = st.number_input("⚙️ Reference value (k)", value=0.5, min_value=0.0, step=0.1)
        h = st.number_input("📉 Decision interval (h)", value=5.0, min_value=0.1, step=0.1)

        if st.button("Run CUSUM Test"):
            try:
                series = filtered_df[selected_col].dropna().reset_index(drop=True)
                cusum_pos, cusum_neg, out_flags = cusum_analysis(series, target, k, h)

                fig = go.Figure()
                fig.add_trace(go.Scatter(y=cusum_pos, mode='lines+markers', name='CUSUM +', line=dict(color='blue')))
                fig.add_trace(go.Scatter(y=cusum_neg, mode='lines+markers', name='CUSUM -', line=dict(color='red')))
                fig.add_hline(y=h, line_dash="dash", line_color="green", annotation_text="+h threshold")
                fig.add_hline(y=-h, line_dash="dash", line_color="green", annotation_text="-h threshold")
                fig.update_layout(title=f"CUSUM Chart for {selected_col} — {selected_material}",
                                  xaxis_title="Observation Index",
                                  yaxis_title="Cumulative Deviation",
                                  height=500)

                st.plotly_chart(fig, use_container_width=True)

                out_indices = [i for i, flag in enumerate(out_flags) if flag]
                if out_indices:
                    st.warning(f"⚠️ CUSUM exceeded the decision threshold at indices: {out_indices}")
                else:
                    st.success("✅ All values are within threshold. No shift detected.")

                with st.expander("🧠 Interpretation"):
                    if out_indices:
                        st.markdown(f"""
                        - 📌 The CUSUM chart indicates a **potential process shift** for **{selected_material}**.
                        - ✅ Investigate process events or changes around indices: `{out_indices}`.
                        - 🔍 This may suggest a calibration issue, instrument drift, or an external influence.
                        """)
                    else:
                        st.markdown(f"""
                        - ✅ The process for **{selected_material}** appears **in control**.
                        - 📉 CUSUM did not detect any consistent drift or signal beyond the thresholds.
                        """)

            except Exception as e:
                st.error(f"❌ Error: {e}")
