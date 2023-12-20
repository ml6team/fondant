import streamlit as st
from streamlit_extras.app_logo import add_logo


def render_sidebar():
    add_logo("content/fondant_logo.png")

    with st.sidebar:
        # Increase the width of the sidebar to accommodate logo
        st.markdown(
            """
            <style>
                section[data-testid="stSidebar"] {
                    width: 350px !important;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("## General Configuration"):
            st.markdown(f"### Base path: \n {st.session_state.base_path}")
