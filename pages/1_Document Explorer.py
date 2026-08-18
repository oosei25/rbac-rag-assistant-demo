from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st
from requests.auth import HTTPBasicAuth


API = os.environ.get("API_URL", "http://api:8000")
try:
    API = st.secrets.get("API_URL", API)
except Exception:
    pass

st.set_page_config(page_title="Document Explorer", layout="wide")
st.title("📚 Document Explorer")

auth = st.session_state.get("auth")
role = st.session_state.get("role")
if not auth:
    st.info("Sign in from the main page to browse authorized documents.")
    st.stop()


def api_get(path: str) -> requests.Response | None:
    try:
        response = requests.get(
            f"{API}{path}",
            auth=HTTPBasicAuth(*auth),
            timeout=15,
        )
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not reach the document API: {exc}")
        return None
    if response.status_code == 401:
        st.error("Your session credentials are no longer valid. Sign in again.")
        return None
    if not response.ok:
        st.error(f"Document API returned HTTP {response.status_code}.")
        return None
    return response


response = api_get("/documents")
if response is None:
    st.stop()

documents = response.json()
st.caption(f"Signed in role: `{role}` · Access is enforced by the API.")
if not documents:
    st.info("No documents are visible for your current role.")
    st.stop()

df = pd.DataFrame(documents)
filter_cols = st.columns(2)
with filter_cols[0]:
    departments = sorted(df["department"].unique())
    selected_departments = st.multiselect(
        "Department (limited by your role)",
        departments,
        default=departments,
    )
with filter_cols[1]:
    search_text = st.text_input("Search in title / preview", "")

filtered = df[df["department"].isin(selected_departments)]
if search_text:
    query = search_text.lower()
    filtered = filtered[
        filtered["title"].str.lower().str.contains(query, regex=False)
        | filtered["preview"].str.lower().str.contains(query, regex=False)
    ]

st.subheader(f"Documents ({len(filtered)})")
st.dataframe(
    filtered[["title", "department", "path", "preview"]],
    use_container_width=True,
    hide_index=True,
)

st.subheader("View full document")
if filtered.empty:
    st.info("No documents match your current filters.")
    st.stop()

document_ids = filtered["document_id"].tolist()
labels = {
    row["document_id"]: f"{row['title']} ({row['department']})"
    for _, row in filtered.iterrows()
}
selected_id = st.selectbox(
    "Select a document to view",
    options=document_ids,
    format_func=lambda document_id: labels[document_id],
)
detail_response = api_get(f"/documents/{selected_id}")
if detail_response is None:
    st.stop()

document = detail_response.json()
st.caption(
    f"Department: `{document['department']}` · Path: `{document['path']}`"
)
st.markdown(document["content"])
