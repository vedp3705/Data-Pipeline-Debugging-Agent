import streamlit as st
import requests

st.title("Data Pipeline Debugging Agent")

API_URL = st.text_input("Enter API URL:", "http://127.0.0.1:8000/diagnose")

if st.button("Run Diagnosis"):
    try:
        resp = requests.get(API_URL)
        st.write(resp.json()["diagnosis"])
    except Exception as e:
        st.error(f"Error contacting API: {e}")
