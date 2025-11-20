import os
import uuid
import requests
import streamlit as st
from requests.auth import HTTPBasicAuth

# --- Config -----
API = os.environ.get("API_URL", "http://api:8000")
try:
    API = st.secrets.get("API_URL", API)
except Exception:
    pass

st.set_page_config(page_title="Company RBAC Chat", layout="wide")

# --- Session defaults ------
DEFAULTS = {
    "auth": None,              
    "role": None,              
    "thread_id": None,        
    "login_username": "",
    "login_password": "",
    "last_user_choice": "Choose a demo user…",
    "failed_logins": 0,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# Demo users ------------
DEMO_USERS = [
    {"label": "Peter — Engineering", "username": "Peter",  "password": "pete123"},
    {"label": "Mariam — Marketing",  "username": "Mariam", "password": "mariampass123"},
    {"label": "Natasha — HR",        "username": "Natasha","password": "hrpass123"},
]
DEMO_OPTIONS = ["Choose a demo user…"] + [u["label"] for u in DEMO_USERS] + ["Custom…"]


# Sidebar: authentication ---------
with st.sidebar:
    st.header("Login")

    # Demo picker
    current_choice = st.session_state.get("last_user_choice", DEMO_OPTIONS[0])
    selected_index = DEMO_OPTIONS.index(current_choice) if current_choice in DEMO_OPTIONS else 0
    user_choice = st.selectbox(
        "Quick-pick a user (auto-fills password)",
        DEMO_OPTIONS,
        index=selected_index,
        help="Use these demo accounts to test role-based access quickly.",
    )

    # When the demo choice changes, update the inputs
    if user_choice != st.session_state.last_user_choice:
        st.session_state.last_user_choice = user_choice

        if user_choice in DEMO_OPTIONS[1:-1]:  # a real demo user selected
            demo = DEMO_USERS[DEMO_OPTIONS.index(user_choice) - 1]
            st.session_state.login_username = demo["username"]
            st.session_state.login_password = demo["password"]
        elif user_choice == "Custom…":
            # Let the user type manually
            st.session_state.login_username = ""
            st.session_state.login_password = ""
        else:
            # Reset for "Choose…"
            st.session_state.login_username = ""
            st.session_state.login_password = ""

        # Clear any previous auth/thread when switching identity
        for k in ("auth", "role", "thread_id"):
            st.session_state[k] = None

    # Use a form so Enter submits cleanly
    with st.form(key="login_form", clear_on_submit=False):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        try:
            r = requests.get(f"{API}/login", auth=HTTPBasicAuth(username, password), timeout=10)
            if r.ok:
                st.session_state.auth = (username, password)
                st.session_state.role = r.json().get("role", "unknown")
                st.session_state.failed_logins = 0

                # Create a stable thread_id per signed-in user (used by Graph engine)
                if not st.session_state.thread_id:
                    st.session_state.thread_id = f"{username}-{uuid.uuid4().hex}"

            elif r.status_code in (401, 403):
                st.session_state.failed_logins += 1
                st.error("Incorrect username or password.")
                st.session_state.login_password = ""  
            else:
                st.error(f"Sign-in failed (HTTP {r.status_code}). Please try again.")
        except requests.exceptions.Timeout:
            st.error("The server took too long to respond. Please try again.")
        except requests.exceptions.ConnectionError:
            st.error(f"Can’t reach the API at {API}. Is the server running?")
        except Exception:
            st.error("Something went wrong while signing in. Please try again.")

    if st.session_state.auth:
        u, _ = st.session_state.auth
        st.success(f"Signed in as **{u}** ({st.session_state.role})")
        if st.button("Sign out"):
            for k in ("auth", "role", "thread_id"):
                st.session_state[k] = None
            st.session_state.login_password = ""
            st.success("Signed out.")

# Main: intro + chat -----------

# Intro
with st.container():
    st.title("Ask the Knowledge Base")
    st.markdown(
        """
**What is this app?**

- A retrieval-augmented assistant with **role-based access control (RBAC)**.  
- Two execution modes:
  - **RAG** — fast, single-turn answers with citations.
  - **Graph (LangGraph)** — stateful, multi-step RAG pipeline with conversation memory per **Thread**.
- Answers cite the exact files used. Access is limited by your role (e.g., Marketing can’t see HR docs).
        """
    )

# -- Auth state -----
authed = bool(st.session_state.get("auth"))
user = st.session_state["auth"][0] if authed else None

# --- Engine picker -------
engine = st.selectbox(
    "Engine",
    options=["RAG", "Graph (LangGraph)"],
    index=0,
    help="Choose the backend engine for this question.",
    key="engine_select",
    disabled=not authed,  
)
endpoint = "/chat/rag" if engine.startswith("RAG") else "/chat/graph"

# ---- Thread row --------
col_a, col_b = st.columns([0.8, 0.2], vertical_alignment="bottom")
with col_a:
    st.caption(
        f"Thread: `{st.session_state.get('thread_id', '—')}` "
        "(used only for Graph engine to keep conversation memory)"
    )
with col_b:
    def _new_conv():
        base = st.session_state["auth"][0] if authed else "anon"
        st.session_state["thread_id"] = f"{base}-{uuid.uuid4().hex}"
        st.toast("Started a new conversation thread.")

    st.button("New conversation", disabled=not authed, on_click=_new_conv)

# ----- Query box ------
q = st.text_area(
    "Your question",
    "Summarize the latest marketing report and cite sources.",
    height=80,
    disabled=not authed, 
)


# ----- Submit -----
ask_clicked = st.button(
    "Ask",
    disabled=(not authed) or (not q.strip()),
    help=None if authed else "Sign in to ask a question.",
)

if ask_clicked:
    with st.spinner(f"Thinking with {engine}…"):
        try:
            au = HTTPBasicAuth(*st.session_state["auth"])  # safe: only when authed
            payload = {"message": q}
            if endpoint == "/chat/graph":
                if "thread_id" not in st.session_state:
                    st.session_state["thread_id"] = f"{user}-{uuid.uuid4().hex}"
                payload["thread_id"] = st.session_state["thread_id"]

            r = requests.post(f"{API}{endpoint}", json=payload, auth=au, timeout=120)
            if r.ok:
                data = r.json()
                st.markdown(data.get("answer", ""))
                sources = data.get("sources") or []
                if sources:
                    with st.expander("Sources"):
                        for i, s in enumerate(sources, 1):
                            st.write(f"[{i}] `{s}`")
            else:
                st.error(f"Error: {r.status_code} {r.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

