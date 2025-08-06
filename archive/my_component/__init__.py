import os
import streamlit as st
import streamlit.components.v1 as components

# The path to the frontend build folder
parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "frontend/build")

_component = components.declare_component(
    "statistical_test_selector",
    path=build_dir
)

def statistical_test_selector():
    """
    Renders the statistical test selector component.
    """
    _component(key="statistical_test_selector_key")