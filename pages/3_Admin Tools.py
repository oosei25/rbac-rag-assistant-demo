from __future__ import annotations

import os

import requests
import streamlit as st
from requests.auth import HTTPBasicAuth


API = os.environ.get("API_URL", "http://api:8000")
try:
    API = st.secrets.get("API_URL", API)
except Exception:
    pass

st.set_page_config(page_title="Admin Tools", layout="wide")
st.title("Admin Tools")

auth = st.session_state.get("auth")
role = st.session_state.get("role")
username = auth[0] if auth else None

status_cols = st.columns(2)

with status_cols[0]:
    st.subheader("API health")
    try:
        health = requests.get(f"{API}/healthz", timeout=5)
        if health.ok:
            st.success("API is reachable.")
        else:
            st.error(f"API health check failed with HTTP {health.status_code}.")
    except requests.exceptions.RequestException as exc:
        st.error(f"API is unreachable: {exc}")

with status_cols[1]:
    st.subheader("Runtime")
    try:
        version = requests.get(f"{API}/version", timeout=5)
        if version.ok:
            data = version.json()
            st.write(f"Vector DB: `{data.get('vector_db', 'unknown')}`")
            st.write(f"Ollama model: `{data.get('ollama_model', 'unknown')}`")
        else:
            st.warning(f"Version endpoint returned HTTP {version.status_code}.")
    except requests.exceptions.RequestException as exc:
        st.warning(f"Could not read runtime details: {exc}")

st.divider()
st.subheader("Reindex knowledge base")

if not auth:
    st.info("Sign in from the main page before running admin actions.")
    st.stop()

st.caption(f"Signed in as `{username}` with role `{role}`.")
can_reindex = role in {"engineering", "clevel"}

if not can_reindex:
    st.warning("Only engineering and clevel roles can trigger reindexing.")

if st.button("Reindex documents", type="primary", disabled=not can_reindex):
    try:
        with st.spinner("Reindexing documents..."):
            response = requests.post(
                f"{API}/admin/reindex",
                auth=HTTPBasicAuth(*auth),
                timeout=300,
            )
        if response.ok:
            st.success(f"Indexed {response.json().get('indexed_chunks', 0)} chunks.")
        elif response.status_code == 403:
            st.error("Your role is not allowed to reindex documents.")
        else:
            st.error(f"Reindex failed with HTTP {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as exc:
        st.error(f"Reindex request failed: {exc}")
