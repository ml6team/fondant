"""Methods for building widget tabs and tables."""

import streamlit as st


def get_default_index(key: str, option_list: list) -> int:
    """Get the default index for a selectbox based on previous selection from session state."""
    if st.session_state[key] is None:
        return 0

    return option_list.index(st.session_state[key])
