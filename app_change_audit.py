"""Standalone entry point for the Part IV workplace demo:  streamlit run app_change_audit.py"""
import streamlit as st
from change_audit.ui import render

st.set_page_config(page_title="Change Validation Audit Copilot", layout="wide")
render()
