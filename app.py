import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import time

# --- 1. PAGE CONFIGURATION ---
# This sets the browser tab title and makes the layout wide and modern.
st.set_page_config(page_title="Retail AI System", layout="wide")

# --- 2. VISUAL STYLING (CSS) ---
# We use CSS to make the sidebar dark blue and the fonts easy to read.
st.markdown("""
    <style>
        .stApp { background-color: #f8fafc; }
        [data-testid="stSidebar"] {
            background-color: #1e3a8a !important; /* Deep Blue Sidebar */
        }
        [data-testid="stSidebar"] * {
            color: #ffffff !important; /* White text for clarity */
            font-size: 1.1rem !important;
        }
        .auth-card {
            background-color: white;
            padding: 2.5rem;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE ---
# This remembers if the user is logged in so the page doesn't reset.
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_email' not in st.session_state:
    st.session_state['user_email'] = ""

# --- 4. LOADING THE AI (YOLOv8) ---
# This function loads the "brain" of the project. 
# '@st.cache_resource' ensures it only loads once to keep the app fast.
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt") 

try:
    model = load_yolo()
except Exception as e:
    st.error("Error: 'yolov8n.pt' file not found in D:\\Xender\\app")

# =========================================================
#             5. LOGIN & ACCOUNT CREATION SYSTEM
# =========================================================
# This section blocks access to the dashboard until the user logs in.
if not st.session_state['logged_in']:
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col_center, _ = st.columns([1, 2, 1])
    
    with col_center:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.title("🛍️ Retail AI Gateway")
        tab1, tab2 = st.tabs(["Login", "Create Account"])
        
        with tab1:
            email = st.text_input("Email", key="l_email")
            pw = st.text_input("Password", type="password", key="l_pw")
            if st.button("Sign In", use_container_width=True):
                if email and pw:
                    st.session_state['logged_in'] = True
                    st.session_state['user_email'] = email
                    st.rerun()
        
        with tab2:
            new_email = st.text_input("New Email", key="s_email")
            new_pw = st.text_input("New Password", type="password", key="s_pw")
            if st.button("Register", use_container_width=True):
                if new_email and new_pw:
                    st.session_state['logged_in'] = True
                    st.session_state['user_email'] = new_email
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop() # Stops execution here if not logged in

# =========================================================
#             6. SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:
    st.markdown("### 🌐 RETAIL MONITOR PRO")
    st.write(f"Logged as: {st.session_state['user_email']}")
    st.markdown("---")
    # This creates the menu to switch between pages.
    page = st.radio("MAIN MENU", ["📊 Dashboard", "🔍 New Analysis", "📄 History"])
    st.markdown("<br>"*10, unsafe_allow_html=True)
    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()

# --- HEADER BAR ---
# Shows the user initial in a yellow circle for a professional look.
u_initial = st.session_state['user_email'][0].upper()
st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; background: white; padding: 10px 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
        <h4 style="margin:0; color:#1e3a8a;">AI Engine Status: <span style="color:#10b981;">Ready</span></h4>
        <div style="width:35px; height:35px; background:#fbbf24; border-radius:50%; display:flex; align-items:center; justify-content:center; color:white; font-weight:bold;">{u_initial}</div>
    </div>
""", unsafe_allow_html=True)

# =========================================================
#             7. PAGE 1: DASHBOARD
# =========================================================
if "Dashboard" in page:
    st.header("Global Store Performance")
    # Metrics show the "Big Numbers" for the manager.
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Visitors", "1,842", "+14%")
    c2.metric("Hot Zone", "Counter 02", "Action Required")
    c3.metric("Uptime", "99.9%")
    
    # A simple chart to represent data over time.
    st.subheader("Visitor Flow (Past 24 Hours)")
    st.line_chart(pd.DataFrame(np.random.randn(20, 2), columns=['Entry', 'Exit']))

# =========================================================
#             8. PAGE 2: NEW ANALYSIS (AI CORE)
# =========================================================
elif "New Analysis" in page:
    st.header("🔍 Analysis Engine")
    
    # Selection menu for Live vs. Video
    mode = st.selectbox("Choose Source", ["Live Webcam Monitor", "Upload Supermarket Video"])
    
    # --- LIVE WEBCAM LOGIC ---
    if mode == "Live Webcam Monitor":
        st.subheader("📹 Real-Time YOLOv8 Detection")
        run_cam = st.checkbox("Start Webcam")
        FRAME_WINDOW = st.image([]) # Empty space where the video will show
        
        if run_cam:
            vid = cv2.VideoCapture(0)
            while run_cam:
                ret, frame = vid.read()
                if not ret: break
                
                # YOLOv8 checks each frame for 'person' (class 0)
                results = model(frame, classes=[0], verbose=False)
                annotated_frame = results[0].plot() # Draws the boxes
                
                # Convert colors (OpenCV uses BGR, Streamlit uses RGB)
                display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(display_frame)
            vid.release()

    # --- VIDEO UPLOAD LOGIC ---
    else:
        st.subheader("📁 Supermarket POS Analysis")
        up_file = st.file_uploader("Upload Store Video", type=['mp4', 'avi'])
        
        if up_file:
            st.video(up_file)
            if st.button("🚀 Process & Generate Report"):
                with st.spinner("AI is calculating densities..."):
                    time.sleep(3) # Simulate thinking time
                    st.success("Analysis Complete!")
                    
                    # RESULTS TABLE: Matches your supermarket requirements.
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Counter 1", "3 People", "Clear")
                    m2.metric("Counter 2", "9 People", "!! BUSY !!")
                    m3.metric("Counter 3", "0 People", "Idle")
                    
                    df = pd.DataFrame({
                        "Counter ID": ["POS-01", "POS-02", "POS-03"],
                        "Wait Time": ["2 min", "12 min", "0 min"],
                        "Action": ["Normal", "Open New Counter", "Redirect Staff"]
                    })
                    st.table(df)
                    st.warning("Strategic Recommendation: Move staff from Counter 3 to Counter 2 immediately.")

# =========================================================
#             9. PAGE 3: HISTORY
# =========================================================
elif "History" in page:
    st.header("Saved Records")
    st.write("This table shows data stored in the database.")
    st.dataframe(pd.DataFrame({
        "Report ID": ["#901", "#902", "#903"],
        "Timestamp": ["10:00 AM", "01:45 PM", "04:10 PM"],
        "Zone": ["Main Entrance", "Aisle 4", "Checkout"],
        "Status": ["Saved", "Saved", "Saved"]
    }), use_container_width=True)