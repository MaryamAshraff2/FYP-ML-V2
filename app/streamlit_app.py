# =============================
# app/streamlit_app.py (UPDATED)
# =============================
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

    
import streamlit as st
from ui.upload import render_upload_section
from ui.session import init_session_state
from ui.display import render_results
from core.pipeline.main_pipeline import main_pipeline


    
def run_app():
    st.set_page_config(page_title="Multi-Camera Person ReID", layout="wide")
    st.title("Multi-Camera Person Re-Identification System")

    init_session_state()

    camera_a_file, camera_b_file = render_upload_section()

    if st.button("Run System", type="primary"):
        if camera_a_file is None or camera_b_file is None:
            st.warning("⚠️ Please upload both Camera A and Camera B videos")
        else:
            try:
                with st.spinner("🔄 Processing videos... This may take a few minutes."):
                    results = main_pipeline(camera_a_file, camera_b_file)
                    
                    if results['status'] == 'success':
                        st.session_state["results"] = results
                        st.success("✅ Processing completed successfully!")
                    else:
                        st.error(f"❌ Processing failed: {results.get('message', 'Unknown error')}")
                        
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)  # Show full traceback for debugging

    render_results()


if __name__ == "__main__":
    run_app()
