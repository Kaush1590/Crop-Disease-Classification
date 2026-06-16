import streamlit as st
from pathlib import Path

pages_path = Path(__file__).parent / "pages"

prediction_page = st.Page(page=pages_path / "prediction.py", title="Prediction", icon="🔍")
dashbord_page = st.Page(page=pages_path / "dashboard.py", title="Dashboard", icon="🗃️")

run = st.navigation([prediction_page, dashbord_page], expanded=False)
run.run()