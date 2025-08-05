import streamlit as st
from my_component import statistical_test_selector

st.set_page_config(
    page_title="Statistical Test Selector",
    layout="wide"
)

st.title("Statistical Test Selector")
st.write("---")

st.markdown("""
This page embeds a custom React component that helps you choose the right statistical test for your data.
""")

# Call the function from your component wrapper to render the React app
statistical_test_selector()