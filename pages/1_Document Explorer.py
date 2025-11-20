from pathlib import Path

import pandas as pd
import streamlit as st

# Uses same RBAC policy as the API
from app.policy import allowed_departments

st.set_page_config(page_title="Document Explorer", layout="wide")

st.title("📚 Document Explorer")
 

# --- Helpers ----------

def get_current_role() -> str:
    """
    Helper to pull the current user's role from Streamlit session.
    """
    for key in ("user", "current_user", "auth_user"):
        if key in st.session_state and isinstance(st.session_state[key], dict):
            return st.session_state[key].get("role", "guest")

    if "role" in st.session_state:
        return str(st.session_state["role"])

    # Default if not logged in
    return "guest"


# Repo root = parent of /pages
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "resources" / "data"  # resources/data/...


def iter_docs():
    """
    Walk resources/data and yield metadata for each Markdown file.
    """
    for path in DATA_DIR.rglob("*.md"):
        rel = path.relative_to(DATA_DIR)

        # department = first folder under resources/data (engineering, finance, etc.)
        parts = rel.parts
        department = parts[0] if len(parts) > 1 else "general"

        # Read file
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        lines = text.strip().splitlines()
        first_line = lines[0] if lines else ""
        title = first_line.lstrip("# ").strip() or path.stem

        # Short preview (first few non-heading lines)
        body_lines = [ln for ln in lines[1:10] if ln.strip()]
        preview = " ".join(body_lines[:3])[:400] + ("…" if len(body_lines) > 3 else "")

        yield {
            "title": title,
            "department": department,
            "relative_path": str(rel),
            "absolute_path": str(path),
            "preview": preview,
        }


# --- Load docs --------

docs = list(iter_docs())
if not docs:
    st.error("No documents found under `resources/data`. Check the DATA_DIR path.")
    st.stop()

df = pd.DataFrame(docs)

# --- RBAC filter --
role = get_current_role()
allowed_depts = set(allowed_departments(role))

# If allowed_depts is empty, user sees nothing
df = df[df["department"].isin(allowed_depts)]

st.caption(
    f"Signed in role: `{role}` · You can browse departments: "
    + ", ".join(sorted(allowed_depts)) if allowed_depts else
    f"Signed in role: `{role}` · You currently have no document access."
)

if df.empty:
    st.info("No documents are visible for your current role.")
    st.stop()

# --- UI filters -------

cols = st.columns(3)
with cols[0]:
    dept_options = sorted(df["department"].unique())
    selected_depts = st.multiselect(
        "Department (limited by your role)",
        dept_options,
        default=dept_options,
    )

with cols[1]:
    search_text = st.text_input("Search in title / preview", "")

filtered = df[df["department"].isin(selected_depts)]

if search_text:
    q = search_text.lower()
    filtered = filtered[
        filtered["title"].str.lower().str.contains(q)
        | filtered["preview"].str.lower().str.contains(q)
    ]

st.subheader(f"Documents ({len(filtered)})")

st.dataframe(
    filtered[["title", "department", "relative_path", "preview"]],
    use_container_width=True,
    hide_index=True,
)

# --- Detail view ----

st.subheader("View full document")

if not filtered.empty:
    selected_title = st.selectbox(
        "Select a document to view",
        options=filtered["title"].tolist(),
    )

    row = filtered[filtered["title"] == selected_title].iloc[0]

    st.caption(
        f"Department: `{row['department']}` · "
        f"Path: `resources/data/{row['relative_path']}`"
    )

    with open(row["absolute_path"], "r", encoding="utf-8") as f:
        content = f.read()

    # Render the markdown
    st.markdown(content)
else:
    st.info("No documents match your current filters.")