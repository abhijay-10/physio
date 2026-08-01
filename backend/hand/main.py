import streamlit as st
import importlib.util
import os
import sys

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Physio Hand Master UI", layout="wide")

st.title("🖐️ Physio Hand Posture Master Dashboard")
st.markdown("Select a specific posture detection module to begin.")

# ==========================================
# FUNCTION TO RUN SUB-SCRIPTS
# ==========================================
def run_module(folder_name):
    # Construct path to the specific stream.py
    script_path = os.path.join(folder_name, "stream.py")
    
    if os.path.exists(script_path):
        # We use 'exec' to run the script within the current context
        # This allows the sub-script to use the existing Streamlit session
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()
            # Clear the current UI to make room for the module
            st.empty() 
            exec(code, globals())
    else:
        st.error(f"❌ Could not find {script_path}")

# ==========================================
# DASHBOARD GRID (7 BOXES)
# ==========================================
# We create a 4-column layout for the grid
col1, col2, col3, col4 = st.columns(4)

# Define the modules based on your folder names
modules = [
    {"name": "Bilateral Hand", "folder": "bilateralhand", "icon": "👐"},
    {"name": "Fan Lateral", "folder": "fanlateral", "icon": "🖐️"},
    {"name": "Lateral Hand", "folder": "lateralhand", "icon": "✋"},
    {"name": "Oblique Hand", "folder": "obliquehand", "icon": "🖖"},
    {"name": "PA Hand", "folder": "pa_hand", "icon": "🤚"},
    {"name": "PA 3-Finger", "folder": "pa3finger", "icon": "✌️"},
    {"name": "Oblique Thumb", "folder": "obliquethumb", "icon": "👍"}
]

# Initialize session state to track which module is active
if "active_module" not in st.session_state:
    st.session_state.active_module = None

# If no module is selected, show the grid
if st.session_state.active_module is None:
    for i, mod in enumerate(modules):
        # Determine which column to place the box in
        current_col = [col1, col2, col3, col4][i % 4]
        
        with current_col:
            with st.container(border=True):
                st.subheader(f"{mod['icon']} {mod['name']}")
                if st.button(f"Open {mod['name']}", key=mod['folder']):
                    st.session_state.active_module = mod['folder']
                    st.rerun()

# If a module is selected, run its code
else:
    if st.button("⬅️ Back to Master Dashboard"):
        st.session_state.active_module = None
        st.rerun()
    
    st.divider()
    run_module(st.session_state.active_module)