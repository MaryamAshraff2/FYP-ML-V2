# =============================
# app/ui/upload.py
# =============================

import streamlit as st


def render_upload_section():
    st.subheader("Upload Video Inputs")

    col1, col2 = st.columns(2)

    with col1:
        camera_a_file = st.file_uploader(
            "Upload Camera A Video",
            type=["mp4", "avi", "mov"],
            key="camera_a"
        )

    with col2:
        camera_b_file = st.file_uploader(
            "Upload Camera B Video",
            type=["mp4", "avi", "mov"],
            key="camera_b"
        )

    return camera_a_file, camera_b_file