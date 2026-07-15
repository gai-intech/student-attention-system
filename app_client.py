import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set page configuration with premium dark/glassmorphic styling
st.set_page_config(
    page_title="Student Attention System Client",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium stylesheet overrides for modern UI aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 36px;
        font-weight: 800;
        color: #1B8A3A;
    }
    .metric-title {
        font-size: 14px;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
""", unsafe_allow_html=True)

# Remote API configuration
API_HOST = "13.60.62.49"
API_PORT = "8000"
API_URL = f"http://{API_HOST}:{API_PORT}"

st.sidebar.title("🎓 Model Connection")
st.sidebar.markdown("---")
server_ip = st.sidebar.text_input("AWS Server IP", API_HOST)
server_port = st.sidebar.text_input("Port", API_PORT)
st.sidebar.markdown("---")

# Health Check connection verification
health_url = f"http://{server_ip}:{server_port}/health"
try:
    health_resp = requests.get(health_url, timeout=5)
    if health_resp.status_code == 200 and health_resp.json().get("status") == "ok":
        st.sidebar.success("● Connected to AWS Server")
    else:
        st.sidebar.error("❌ Connection failed")
except Exception:
    st.sidebar.error("❌ Offline (Could not connect)")

st.title("🎓 Student Attention & Behavior Analytics Dashboard")
st.markdown("This dashboard communicates directly with the **FastAPI model server** deployed inside the **Kubernetes cluster** on AWS to perform real-time video evaluation.")

st.markdown("---")

# Video file upload selector
uploaded_file = st.file_uploader("Select Classroom / Student Video File (MP4)", type=["mp4", "avi", "mov"])
target_fps = st.slider("Evaluation Sampling Rate (Frames/Second)", 1, 5, 2, help="Higher FPS increases accuracy but takes longer to run.")

if uploaded_file is not None:
    # Preview uploaded video
    col1, col2 = st.columns([1, 2])
    with col1:
        st.video(uploaded_file)
        
    with col2:
        st.info("Click 'Run Remote Evaluation' below to send this video file to the AWS API endpoint. The model will run YOLO detection, persistence tracking, behavior classification, and emotion estimation on the server, returning aggregated analytics.")
        run_btn = st.button("Run Remote Evaluation 🚀", type="primary")

    if run_btn:
        with st.spinner("Processing video on AWS EC2 node... (This can take a moment depending on length)"):
            # Prepare files payload
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "video/mp4")}
            params = {"fps": target_fps}
            predict_url = f"http://{server_ip}:{server_port}/predict"
            
            try:
                response = requests.post(predict_url, files=files, params=params, timeout=300)
                
                if response.status_code == 200:
                    report = response.json()
                    st.success("Evaluation completed successfully!")
                    
                    st.markdown("### 📊 Overall Session Metrics")
                    
                    # Display metrics row
                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">Overall Attention Score</div>
                                <div class="metric-value" style="color:#1B8A3A;">{report['overall_attention']}%</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with m_col2:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">Unique Students Detected</div>
                                <div class="metric-value" style="color:#C77800;">{report['student_count']}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with m_col3:
                        teacher_text = f"Track #{report['teacher_id']}" if report['teacher_id'] is not None else "None Detected"
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">Teacher Identified (Excluded)</div>
                                <div class="metric-value" style="color:#3F51B5;">{teacher_text}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Display graphs row
                    g_col1, g_col2 = st.columns(2)
                    
                    with g_col1:
                        st.markdown("#### Class Attention Timeline")
                        curve = report.get("class_curve", [])
                        if len(curve) > 0:
                            # Convert curve list to DataFrame
                            df_curve = pd.DataFrame(curve, columns=["Time (s)", "Attention Score"])
                            df_curve["Attention %"] = df_curve["Attention Score"] * 100
                            st.line_chart(df_curve.set_index("Time (s)")["Attention %"])
                        else:
                            st.write("No timeline data generated.")
                            
                    with g_col2:
                        st.markdown("#### Session Behavior Profile")
                        dist = report.get("behavior_dist", {})
                        if dist:
                            df_dist = pd.DataFrame(list(dist.items()), columns=["Behavior", "Frames"])
                            st.bar_chart(df_dist.set_index("Behavior"))
                        else:
                            st.write("No behavior data available.")
                    
                    st.markdown("---")
                    
                    # Student statistics table
                    st.markdown("#### Per-Student Focus Analysis")
                    student_data = report.get("per_student", {})
                    if student_data:
                        rows = []
                        for sid, sinfo in student_data.items():
                            rows.append({
                                "Student Track ID": f"#{sid}",
                                "Attention Percentage": f"{sinfo['attention_pct']}%",
                                "Dominant Behavior": sinfo["dominant_behavior"],
                                "Total Tracked Frames": sinfo["frames"]
                            })
                        df_students = pd.DataFrame(rows)
                        st.dataframe(df_students, use_container_width=True)
                    else:
                        st.write("No individual student focus data available.")
                        
                else:
                    st.error(f"Server Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to model server: {str(e)}")
