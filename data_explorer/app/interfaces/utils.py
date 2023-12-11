"""Methods for building widget tabs and tables."""

import streamlit as st


def get_index_from_state(key: str, option_list: list) -> int:
    """Get the default index for a selectbox based on previous selection from session state."""
    selected_option = st.session_state.get(key)
    if selected_option is None or selected_option not in option_list:
        return 0

    return option_list.index(selected_option)
