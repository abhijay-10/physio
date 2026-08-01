import streamlit as st
import os
import sys

# ==========================================
# PAGE SETUP & DASHBOARD SKIN
# ==========================================
st.set_page_config(
    page_title="Axoris Physio - Elbow Diagnostic Suite", 
    layout="wide"
)

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; height: 3.2em; font-weight: bold; }
    .stSelectbox label { font-weight: bold; font-size: 1.1em; }
    h1 { color: #2E86C1; }
    </style>
    """, unsafe_allow_html=True)

# ✅ FIXED: Track by unique relative subfolder. Each contains its own 'stream.py' and image.
ELBOW_MODULES = [
    {
        "name": "Desk Baseline (Straight Elbow)", 
        "folder": "straight", 
        "img": "straight_elbow.png",
        "desc": "Baseline analysis with full forearm flat down on the surface plate."
    },
    {
        "name": "Lateral Elbow (90° Profile)", 
        "folder": "elbow90", 
        "img": "elbow90.png",
        "desc": "Clinical lateral view enforcing a strict 90-degree flexion window."
    },
    {
        "name": "AP Partial Flexion (Distal Humerus)", 
        "folder": "humerus", 
        "img": "humerus.png",
        "desc": "Used when full extension is limited; places the humerus flat on the receptor."
    },
    {
        "name": "AP Acute Flexion (Jones View)", 
        "folder": "humerusjones", 
        "img": "humerus_jones.png",
        "desc": "Advanced acute flexion focusing cleanly on the distal humerus structures."
    },
    {
        "name": "PA Axial (Olecranon Focus)", 
        "folder": "olecaran", 
        "img": "olecaran.png",
        "desc": "PA axial alignment targeting the olecranon process and articular margins free of superimposition."
    }
]

# Track current module execution context by its subfolder name
if "current_module" not in st.session_state:
    st.session_state.current_module = None

# ==========================================
# SIDEBAR CONTROLS & SELECTION
# ==========================================
st.sidebar.title("🦾 Axoris Elbow Control Center")
st.sidebar.write("Select a diagnostic module below to open the webcam stream inline.")
st.sidebar.divider()

for mod in ELBOW_MODULES:
    button_label = f"📁 {mod['name']}"
    if st.session_state.current_module == mod['folder']:
        button_label = f"▶️ {mod['name']} (Running)"
        
    # ✅ FIXED: Using unique folder strings for navigation keys to prevent Streamlit key collisions
    if st.sidebar.button(button_label, key=f"nav_{mod['folder']}"):
        st.session_state.current_module = mod['folder']
        st.rerun()

st.sidebar.divider()
if st.session_state.current_module:
    if st.sidebar.button("🛑 Stop Camera & Clear", type="primary"):
        st.session_state.current_module = None
        st.rerun()

# ==========================================
# ROBUST AUTO-RESOLVING RUNTIME PIPELINE
# ==========================================
def execute_sub_script(folder_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_folder_path = os.path.join(base_dir, folder_name)
    target_path = os.path.join(target_folder_path, "stream.py")

    if os.path.exists(target_path):
        with open(target_path, "r", encoding="utf-8") as file:
            script_code = file.read()
            
            # Preserve state to revert context smoothly
            old_cwd = os.getcwd()
            old_sys_path = list(sys.path)
            
            try:
                # ✅ FIXED: Hot-swap operational directories so internal sub-scripts can resolve 
                # their own relative media or .task files without paths breaking.
                os.chdir(target_folder_path)
                sys.path.insert(0, target_folder_path)
                
                isolated_globals = globals().copy()
                exec(script_code, isolated_globals)
            except Exception as error:
                st.error(f"❌ Execution Failure inside '{folder_name}/stream.py': {error}")
            finally:
                # Always restore back to base directory context execution state safely
                os.chdir(old_cwd)
                sys.path = old_sys_path
    else:
        st.error(f"📂 **File Not Found Error:** Unable to locate your target stream file script at `{target_path}`.")

# ==========================================
# DUAL-COLUMN MAIN DASHBOARD RENDERER
# ==========================================
base_dir = os.path.dirname(os.path.abspath(__file__))

if st.session_state.current_module is None:
    st.title("🦾 Elbow Diagnostic Suite Landing Center")
    st.write("Please click a module in the sidebar panel to launch its computer vision tracking algorithm.")
    st.divider()
    
    grid_cols = st.columns(3)
    for index, mod in enumerate(ELBOW_MODULES):
        with grid_cols[index % 3]:
            with st.container(border=True):
                st.subheader(mod['name'])
                st.write(mod['desc'])
                
                # ✅ FIXED: Images resolved directly relative to their own nested asset folder location
                img_path = os.path.join(base_dir, mod['folder'], mod['img'])
                
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                
                if st.button("Launch Positioner", key=f"grid_launch_{mod['folder']}"):
                    st.session_state.current_module = mod['folder']
                    st.rerun()
else:
    active_mod_info = next(m for m in ELBOW_MODULES if m['folder'] == st.session_state.current_module)
    layout_col1, layout_col2 = st.columns([1.8, 1.2])
    
    with layout_col2:
        st.subheader("📋 Reference Clinical Guide")
        # ✅ FIXED: Reference images path resolution pointing to subfolders
        ref_img_path = os.path.join(base_dir, active_mod_info['folder'], active_mod_info['img'])
        
        if os.path.exists(ref_img_path):
            st.image(ref_img_path, caption=f"Target Profile: {active_mod_info['name']}", use_container_width=True)
            
        with st.container(border=True):
            st.write(f"**Description:** {active_mod_info['desc']}")
            st.info("💡 **Instructions:** Adjust your arm until the validation panel turns green.")

    with layout_col1:
        execute_sub_script(st.session_state.current_module)