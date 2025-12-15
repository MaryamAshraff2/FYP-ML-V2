# =============================
# app/ui/session.py
# =============================

import streamlit as st


def init_session_state():
    default_keys = {
        "results": None,
        "camera_a_path": None,
        "camera_b_path": None,
        "processing": False,
    }

    for key, value in default_keys.items():
        if key not in st.session_state:
            st.session_state[key] = value


