import streamlit as st
import os

# ==========================================
# 1. PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(page_title="Axoris Physio Master AI", layout="wide")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; font-weight: bold; }
    .stMetric { background-color: #1e2129; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. MODULE DIRECTORY MAPPING
# ==========================================
CHEST_MODULES = [
    {"name": "Back Pose", "folder": "chest/back_pose", "icon": "🧘"},
    {"name": "Front Pose", "folder": "chest/lordotic_front_pose", "icon": "👤"},
    {"name": "Sleep Front", "folder": "chest/sleep_front", "icon": "🛌"},
    {"name": "Back Front", "folder": "chest/sleep_back", "icon": "🔄"},
    {"name": "Sitting Front", "folder": "chest/sitting_front_pose", "icon": "🪑"}
]

HAND_MODULES = [
    {"name": "Bilateral Hand", "folder": "hand/bilateralhand", "icon": "👐"},
    {"name": "Fan Lateral", "folder": "hand/fanlateral", "icon": "🖐️"},
    {"name": "Lateral Hand", "folder": "hand/lateralhand", "icon": "✋"},
    {"name": "Oblique Hand", "folder": "hand/obliquehand", "icon": "🖖"},
    {"name": "PA Hand", "folder": "hand/pa_hand", "icon": "🤚"},
    {"name": "PA 3-Finger", "folder": "hand/pa3finger", "icon": "✌️"},
    {"name": "Oblique Thumb", "folder": "hand/obliquethumb", "icon": "👍"}
]

SPINE_MODULES = [
    {"name": "Lateral Spine Scan", "folder": "spine", "icon": "🦴"}
]

# Enforced precise nested folder paths to match Windows os.path requirements
ELBOW_MODULES = [
    {"name": "Straight Desk Baseline", "folder": os.path.join("elbow", "straight"), "icon": "📏"},
    {"name": "Lateral 90°", "folder": os.path.join("elbow", "elbow90"), "icon": "💪"},
    {"name": "Humerus AP Partial", "folder": os.path.join("elbow", "humerus"), "icon": "📐"},
    {"name": "Humerus Jones AP Acute", "folder": os.path.join("elbow", "humerusjones"), "icon": "💥"},
    {"name": "PA Axial Olecranon", "folder": os.path.join("elbow", "olecaran"), "icon": "🦾"}
]

KNEE_MODULES = [
    {"name": "Hungsten", "folder": os.path.join("knee", "hungsten"), "icon": "🦵"},
    {"name": "PA Knee", "folder": os.path.join("knee", "pa_knee"), "icon": "🦿"}
]

FOOT_MODULES = [
    {"name": "Back Leg Posture", "folder": os.path.join("foot", "foot_angle"), "icon": "🦶"},
    {"name": "Front Leg Posture", "folder": os.path.join("foot", "foot_ap"), "icon": "🦵"}
]

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🚀 Axoris Physio Master")
st.sidebar.divider()

category = st.sidebar.selectbox(
    "📁 Select Diagnostic Category", 
    ["Dashboard Home", "Chest Radiology", "Hand Postures", "Spine Vertebrae", "Elbow Joint Profile", "Knee Diagnostics", "Foot Analytics"]
)

if "active_mod" not in st.session_state:
    st.session_state.active_mod = None


# ==========================================
# 4. ABSOLUTE PATH RESOLUTION EXECUTOR
# ==========================================
def run_module(folder_path):
    # Get the exact absolute directory path of your master orchestrator script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Normalize the folder path variable context cleanly
    clean_folder = os.path.normpath(folder_path)
    
    # Choose correct script name target mapping definitions
    script_name = "spine_app.py" if clean_folder == "spine" else "stream.py"
    
    # Force system to construct an absolute path right from your project root folder
    script_path = os.path.join(base_dir, clean_folder, script_name)
    
    if os.path.exists(script_path):
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()
            try:
                module_globals = globals().copy()
                exec(code, module_globals)
                
                # Execution lifecycle trigger checkpoints hooks
                if "run_spine_analysis" in module_globals:
                    module_globals["run_spine_analysis"]()
                elif "run_chest_analysis" in module_globals:
                    module_globals["run_chest_analysis"]()
                elif "run_hand_analysis" in module_globals:
                    module_globals["run_hand_analysis"]()
                elif "run_elbow_analysis" in module_globals:
                    module_globals["run_elbow_analysis"]()
                elif "run_knee_analysis" in module_globals:
                    module_globals["run_knee_analysis"]()
                elif "run_foot_analysis" in module_globals:
                    module_globals["run_foot_analysis"]()
                    
            except Exception as e:
                st.error(f"❌ Error in execution environment loop: {e}")
    else:
        # Debug screen to show you exactly what is wrong on your local computer storage layout
        st.error(f"📂 **Target Resource Missing Error**")
        st.write(f"The system is looking for the file at: `{script_path}`")
        st.write("Please verify that this file physically exists at that location and has no typos in its name.")

# ==========================================
# 5. DASHBOARD VIEWS
# ==========================================

if st.session_state.active_mod:
    if st.sidebar.button("⬅️ Back to Menu"):
        st.session_state.active_mod = None
        st.rerun()

if category == "Dashboard Home":
    st.title("🖐Welcome to Axoris Physio Master Hub")
    st.write("Select a category in the sidebar to view available diagnostic modules.")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active Chest Modules", len(CHEST_MODULES))
    m2.metric("Active Hand Modules", len(HAND_MODULES))
    m3.metric("Active Spine Modules", len(SPINE_MODULES))
    m4.metric("Active Elbow Modules", len(ELBOW_MODULES)) # Renders new count metric
    
    m5, m6 = st.columns(2)
    m5.metric("Active Knee Modules", len(KNEE_MODULES))
    m6.metric("Active Foot Modules", len(FOOT_MODULES))
    
    st.divider()
    st.image("https://via.placeholder.com/800x200?text=System+Ready+For+Diagnostic+Scan", use_container_width=True)

elif category == "Chest Radiology":
    if not st.session_state.active_mod:
        st.header("🫁 Chest Diagnostic Suite")
        cols = st.columns(3)
        for i, mod in enumerate(CHEST_MODULES):
            with cols[i % 3]:
                with st.container(border=True):
                    st.write(f"### {mod['icon']} {mod['name']}")
                    if st.button("Launch Scanner", key=f"btn_{mod['folder']}"):
                        st.session_state.active_mod = mod['folder']
                        st.rerun()
    else:
        run_module(st.session_state.active_mod)

elif category == "Hand Postures":
    if not st.session_state.active_mod:
        st.header("🖐️ Hand Posture Suite")
        cols = st.columns(4)
        for i, mod in enumerate(HAND_MODULES):
            with cols[i % 4]:
                with st.container(border=True):
                    st.write(f"### {mod['icon']} {mod['name']}")
                    if st.button("Launch Scanner", key=f"btn_{mod['folder']}"):
                        st.session_state.active_mod = mod['folder']
                        st.rerun()
    else:
        run_module(st.session_state.active_mod)

elif category == "Spine Vertebrae":
    if not st.session_state.active_mod:
        st.header("🦴 Spine Diagnostic Suite")
        cols = st.columns(3)
        for i, mod in enumerate(SPINE_MODULES):
            with cols[i % 3]:
                with st.container(border=True):
                    st.write(f"### {mod['icon']} {mod['name']}")
                    if st.button("Launch Scanner", key=f"btn_{mod['folder']}"):
                        st.session_state.active_mod = mod['folder']
                        st.rerun()
    else:
        run_module(st.session_state.active_mod)

# ✅ NEW VIEW PANEL INTERFACE GENERATION Layer
elif category == "Elbow Joint Profile":
    if not st.session_state.active_mod:
        st.header("🦾 Elbow Diagnostic Suite")
        cols = st.columns(3)
        for i, mod in enumerate(ELBOW_MODULES):
            with cols[i % 3]:
                with st.container(border=True):
                    st.write(f"### {mod['icon']} {mod['name']}")
                    if st.button("Launch Scanner", key=f"btn_{mod['folder']}"):
                        st.session_state.active_mod = mod['folder']
                        st.rerun()
    else:
        run_module(st.session_state.active_mod)

elif category == "Knee Diagnostics":
    if not st.session_state.active_mod:
        st.header("🦵 Knee Diagnostic Suite")
        cols = st.columns(3)
        for i, mod in enumerate(KNEE_MODULES):
            with cols[i % 3]:
                with st.container(border=True):
                    st.write(f"### {mod['icon']} {mod['name']}")
                    if st.button("Launch Scanner", key=f"btn_{mod['folder']}"):
                        st.session_state.active_mod = mod['folder']
                        st.rerun()
    else:
        run_module(st.session_state.active_mod)

elif category == "Foot Analytics":
    if not st.session_state.active_mod:
        st.header("🦶 Foot Diagnostic Suite")
        cols = st.columns(3)
        for i, mod in enumerate(FOOT_MODULES):
            with cols[i % 3]:
                with st.container(border=True):
                    st.write(f"### {mod['icon']} {mod['name']}")
                    if st.button("Launch Scanner", key=f"btn_{mod['folder']}"):
                        st.session_state.active_mod = mod['folder']
                        st.rerun()
    else:
        run_module(st.session_state.active_mod)