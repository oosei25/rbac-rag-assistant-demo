from __future__ import annotations

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Usage Analytics", layout="wide")
st.title("Usage Analytics")

events = st.session_state.get("usage_events", [])

if not events:
    st.info("No questions have been asked in this browser session yet.")
    st.stop()

df = pd.DataFrame(events)
if "citation_count" not in df:
    df["citation_count"] = df.get("source_count", 0)

total_questions = len(df)
success_count = int((df["status"] == "ok").sum())
success_rate = success_count / total_questions if total_questions else 0
avg_latency = int(df["duration_ms"].mean()) if total_questions else 0
avg_citations = round(float(df["citation_count"].mean()), 1) if total_questions else 0.0

cols = st.columns(4)
cols[0].metric("Questions", total_questions)
cols[1].metric("Success rate", f"{success_rate:.0%}")
cols[2].metric("Avg latency", f"{avg_latency} ms")
cols[3].metric("Avg citations", avg_citations)

st.divider()

filters = st.columns(3)
with filters[0]:
    engines = sorted(df["engine"].dropna().unique())
    selected_engines = st.multiselect("Engine", engines, default=engines)
with filters[1]:
    statuses = sorted(df["status"].dropna().unique())
    selected_statuses = st.multiselect("Status", statuses, default=statuses)
with filters[2]:
    roles = sorted(df["role"].dropna().unique())
    selected_roles = st.multiselect("Role", roles, default=roles)

filtered = df[
    df["engine"].isin(selected_engines)
    & df["status"].isin(selected_statuses)
    & df["role"].isin(selected_roles)
]

chart_cols = st.columns(2)
with chart_cols[0]:
    st.subheader("Questions by engine")
    st.bar_chart(filtered["engine"].value_counts())
with chart_cols[1]:
    st.subheader("Requests by status")
    st.bar_chart(filtered["status"].value_counts())

st.subheader("Recent activity")
st.dataframe(
    filtered[
        [
            "timestamp",
            "username",
            "role",
            "engine",
            "status",
            "duration_ms",
            "citation_count",
            "question",
        ]
    ].sort_values("timestamp", ascending=False),
    use_container_width=True,
    hide_index=True,
)
