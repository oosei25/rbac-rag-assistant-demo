from __future__ import annotations

import os

import requests
import streamlit as st
from requests.auth import HTTPBasicAuth

from ui_components import (
    ROLE_LABELS,
    SECURITY_LAB_PRESETS,
    render_access_trace,
    render_source_cards,
)


API = os.environ.get("API_URL", "http://api:8000")
try:
    API = st.secrets.get("API_URL", API)
except Exception:
    pass

st.set_page_config(page_title="Security Lab", layout="wide")
st.title("Security Lab")
st.markdown(
    "Run the same question through two role policies and compare the authorized "
    "answers, access decisions, and source cards side-by-side."
)

auth = st.session_state.get("auth")
role = st.session_state.get("role")
if not auth:
    st.info("Sign in from the main page to use the Security Lab.")
    st.stop()
if role != "clevel":
    st.warning(
        "Cross-role comparisons are restricted to the C-level demo role because "
        "one side may contain information outside your current role. Sign in as "
        "Cathy to run the lab."
    )
    st.stop()

st.caption(
    "The trace contains department policy and authorized candidate counts only; "
    "it never exposes rejected document titles, paths, or snippets."
)

preset_labels = [preset["label"] for preset in SECURITY_LAB_PRESETS]
selected_label = st.selectbox("Comparison preset", preset_labels)
preset = next(
    item for item in SECURITY_LAB_PRESETS if item["label"] == selected_label
)
role_options = list(ROLE_LABELS)

question = st.text_area(
    "Question used for both roles",
    value=preset["question"],
    height=90,
    key=f"lab_question:{selected_label}",
)
selectors = st.columns(2)
with selectors[0]:
    left_role = st.selectbox(
        "Left role",
        role_options,
        index=role_options.index(preset["left_role"]),
        format_func=lambda value: ROLE_LABELS[value],
        key=f"lab_left:{selected_label}",
    )
with selectors[1]:
    right_role = st.selectbox(
        "Right role",
        role_options,
        index=role_options.index(preset["right_role"]),
        format_func=lambda value: ROLE_LABELS[value],
        key=f"lab_right:{selected_label}",
    )

if st.button(
    "Run comparison",
    type="primary",
    disabled=not question.strip(),
):
    try:
        with st.spinner("Evaluating both access policies…"):
            response = requests.post(
                f"{API}/security-lab/compare",
                json={
                    "question": question,
                    "left_role": left_role,
                    "right_role": right_role,
                },
                auth=HTTPBasicAuth(*auth),
                timeout=240,
            )
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not reach the Security Lab API: {exc}")
        st.stop()

    if response.status_code == 403:
        st.error("Your authenticated role is not allowed to compare other roles.")
        st.stop()
    if not response.ok:
        st.error(f"Security comparison failed with HTTP {response.status_code}.")
        st.stop()

    data = response.json()
    result_columns = st.columns(2)
    for column, side in zip(result_columns, (data["left"], data["right"])):
        with column:
            side_role = side["role"]
            st.subheader(ROLE_LABELS.get(side_role, side_role.title()))
            decision = side["access_trace"]["decision"]
            if decision == "answered":
                st.success("Authorized answer")
            else:
                st.warning("Access-limited response")
            st.markdown(side["answer"])
            render_access_trace(side["access_trace"])
            render_source_cards(side.get("citations") or [])
