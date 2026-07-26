import warnings
import pandas as pd
import streamlit as st

# (Adjust to 'engine.core' if you renamed your file to core.py)
from engine.core2 import DataCleanerEngine 

# Import your two grouped UI components
from components import ui_cleaning, ui_analysis

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pandas.io.formats.style")

# ==========================================
# 1. PAGE SETUP
# ==========================================
st.set_page_config(page_title="PrepML Studio", page_icon="🧹", layout="wide")

if "engine" not in st.session_state:
    st.session_state.engine = None

# ==========================================
# 2. SIDEBAR
# ==========================================
with st.sidebar:
    st.title("🧹 PrepML Studio")
    st.caption("A no-code data cleaning and preprocessing tool for ML pipelines.")
    st.divider()
    
    # Toggle between Upload and URL
    input_method = st.radio("Choose Data Source:", ["Upload File", "Paste URL"])
    
    if "loaded_source" not in st.session_state:
        st.session_state.loaded_source = None

    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

        # Load whenever there's a new file selected (different from what's already loaded),
        # rather than only when no engine exists yet — so swapping files just works.
        if uploaded_file is not None and st.session_state.loaded_source != ("file", uploaded_file.name, uploaded_file.size):
            try:
                st.session_state.engine = DataCleanerEngine(pd.read_csv(uploaded_file), file_name=uploaded_file.name)
                st.session_state.loaded_source = ("file", uploaded_file.name, uploaded_file.size)
                st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                
    elif input_method == "Paste URL":
        csv_url = st.text_input("🔗 Public CSV URL:", placeholder="https://example.com/data.csv")
        
        # We use a button so it doesn't try to download on every single keystroke
        if st.button("Fetch Data", type="primary") and csv_url:
            try:
                # Extract a dummy filename from the URL, or default to "url_data.csv"
                extracted_name = csv_url.split("/")[-1] if "/" in csv_url else "url_data.csv"
                
                # pd.read_csv handles the URL request automatically!
                st.session_state.engine = DataCleanerEngine(pd.read_csv(csv_url), file_name=extracted_name)
                st.session_state.loaded_source = ("url", csv_url)
                st.success("Data loaded successfully from URL!")
            except Exception as e:
                st.error(f"Failed to load URL. Make sure it points directly to a raw CSV file. Error: {e}")
            
    if st.session_state.engine is not None:
        st.divider()
        if st.button("↺ Reset All Changes", use_container_width=True):
            st.session_state.engine.reset_data()
            st.rerun()
        if st.button("🗑️ Clear Dataset", use_container_width=True):
            st.session_state.engine = None
            st.session_state.loaded_source = None
            st.rerun()

# ==========================================
# 3. DASHBOARD TABS
# ==========================================
if st.session_state.engine is None:
    st.info("Please upload a CSV file in the sidebar to begin.")
    st.stop()

engine: DataCleanerEngine = st.session_state.engine
df = engine.clean_df

tab_profile, tab_prune, tab_clean, tab_scale, tab_target, tab_ml, tab_export = st.tabs([
    "Data Health", "Smart Pruner", "Cleaning Studio", "Scaling Studio", "Target Dependency", "ML Quickstart", "Export Code"
])

# ==========================================
# 4. RENDER UI COMPONENTS
# ==========================================
# Pass the relevant tabs to the Cleaning UI file
ui_cleaning.render(engine, tab_profile, tab_prune, tab_clean, tab_scale)

# Pass the relevant tabs to the Analysis UI file
ui_analysis.render(engine, tab_target, tab_ml, tab_export)