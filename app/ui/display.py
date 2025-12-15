# =============================
# app/ui/display.py
# =============================

import streamlit as st


def render_results():
    st.subheader("Results")

    results = st.session_state.get("results")

    if results is None:
        st.info("Run the system to see results")
        return

    # Placeholder visualization logic
    st.success("Processing completed successfully")

    st.json(results)
