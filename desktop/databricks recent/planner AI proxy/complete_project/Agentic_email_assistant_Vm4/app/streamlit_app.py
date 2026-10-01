"""
streamlit_app.py — Streamlit UI for Agentic Email Intelligence Assistant
=========================================================================
Frame 1 ke sidebar mein dikhta hai: app/streamlit_app.py
Video mein open nahi kiya gaya — reconstruct from context.

Run:
    streamlit run app/streamlit_app.py

Features:
    - Login screen (JWT auth)
    - Email text input form
    - .eml / .msg file upload
    - Results display: summary, intent, entities, priority
    - Agent logs viewer
"""

from __future__ import annotations

import json
import requests
import streamlit as st
from datetime import timedelta

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Email Intelligence Agent",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def init_session():
    for key, default in {
        "token":         None,
        "username":      None,
        "last_response": None,
        "last_request_id": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default


def is_logged_in() -> bool:
    return bool(st.session_state.get("token"))


def get_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


# ══════════════════════════════════════════════════════════════════════════════
# API HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def api_login(username: str, password: str) -> bool:
    try:
        r = requests.post(
            f"{API_BASE}/token",
            json={"username": username, "password": password},
            timeout=10,
        )
        if r.status_code == 200:
            st.session_state.token    = r.json()["access_token"]
            st.session_state.username = username
            return True
        st.error(f"Login failed: {r.json().get('detail', 'Unknown error')}")
        return False
    except requests.ConnectionError:
        st.error(f"Cannot connect to API at {API_BASE}. Is the server running?")
        return False


def api_analyze(payload: dict) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/analyze",
            json=payload,
            headers=get_headers(),
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()
        st.error(f"API error {r.status_code}: {r.json().get('detail', r.text)}")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def api_upload(file_bytes: bytes, filename: str, client_id: str, client_secret: str) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/upload",
            files={"file": (filename, file_bytes, "application/octet-stream")},
            headers=get_headers(),
            params={"client_id": client_id, "client_secret": client_secret},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()
        st.error(f"Upload error {r.status_code}: {r.json().get('detail', r.text)}")
        return None
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None


def api_get_logs(request_id: str) -> list:
    try:
        r = requests.get(
            f"{API_BASE}/logs/{request_id}",
            headers=get_headers(),
            timeout=15,
        )
        if r.status_code == 200:
            return r.json().get("steps", [])
        return []
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def render_login():
    """Login screen."""
    st.title("📧 Email Intelligence Agent")
    st.caption("Agentic Email Analysis — Summary · Intent · Entities · Priority")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("Login")
        with st.form("login_form"):
            username = st.text_input("Username", value="admin")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            with st.spinner("Logging in..."):
                if api_login(username, password):
                    st.success("Logged in!")
                    st.rerun()


def render_sidebar():
    """Sidebar — user info + navigation."""
    with st.sidebar:
        st.markdown(f"**👤 {st.session_state.username}**")
        st.caption("Email Intelligence Agent v1.0")
        st.divider()

        if st.button("🚪 Logout", use_container_width=True):
            for key in ["token", "username", "last_response", "last_request_id"]:
                st.session_state[key] = None
            st.rerun()

        st.divider()
        st.caption(f"API: `{API_BASE}`")


def render_credentials_form() -> tuple[str, str]:
    """Horizon API credentials input."""
    with st.expander("🔑 Horizon API Credentials", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            client_id = st.text_input("Client ID", placeholder="your-client-id", key="cred_id")
        with col2:
            client_secret = st.text_input("Client Secret", type="password",
                                           placeholder="your-client-secret", key="cred_secret")
    return client_id, client_secret


def render_results(response: dict):
    """Display analysis results."""
    st.success("✅ Analysis complete!")

    col1, col2 = st.columns(2)

    with col1:
        # Summary
        st.subheader("📋 Summary")
        st.info(response.get("summary", "—"))

        # Intent + Priority
        intent   = response.get("intent", "—")
        priority = response.get("priority", {})

        intent_colors = {
            "ID Card Change Request":  "🪪",
            "Address Change Request":  "🏠",
            "General Enquiry":         "❓",
            "Other":                   "📌",
        }
        priority_colors = {
            "Urgent":         "🔴",
            "Access to Care": "🟠",
            "Standard":       "🟢",
        }

        st.metric("Intent",   f"{intent_colors.get(intent, '📧')} {intent}")
        st.metric(
            "Priority",
            f"{priority_colors.get(priority.get('level',''), '⚪')} {priority.get('level', '—')}",
            help=priority.get("explanation", ""),
        )

    with col2:
        # Entities
        st.subheader("🏷️ Extracted Entities")
        entities = response.get("extracted_entities", {})
        entity_display = {
            "HCID":            entities.get("hcid", []),
            "Policy Numbers":  entities.get("policy_numbers", []),
            "Case Numbers":    entities.get("case_numbers", []),
            "Dates":           entities.get("dates", []),
            "Amounts":         entities.get("amounts", []),
            "Phone Numbers":   entities.get("phone_numbers", []),
            "Addresses":       entities.get("addresses", []),
            "Action Items":    entities.get("action_items", []),
        }
        for label, values in entity_display.items():
            if values:
                st.markdown(f"**{label}:** {', '.join(str(v) for v in values)}")

    # Raw JSON expander
    with st.expander("🔍 Raw JSON Response"):
        st.json(response)


def render_logs(request_id: str):
    """Agent step logs viewer."""
    if not request_id:
        return

    with st.expander("📊 Agent Processing Steps", expanded=False):
        with st.spinner("Loading logs..."):
            logs = api_get_logs(request_id)

        if not logs:
            st.caption("No logs found.")
            return

        for log in logs:
            status_icon = "✅" if log.get("status") == "success" else "❌"
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                col1.markdown(f"{status_icon} **{log.get('step_name', '—')}**")
                col2.caption(log.get("timestamp", "")[:19])
                if log.get("duration_ms"):
                    col3.caption(f"{log['duration_ms']}ms")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGES
# ══════════════════════════════════════════════════════════════════════════════

def page_analyze():
    """Text input analysis page."""
    st.header("📝 Analyze Email")

    client_id, client_secret = render_credentials_form()

    tab_text, tab_upload = st.tabs(["✍️ Text Input", "📁 File Upload"])

    # ── Text Input Tab ────────────────────────────────────────────────────────
    with tab_text:
        with st.form("analyze_form"):
            subject = st.text_input("Subject", placeholder="Email subject line")
            body    = st.text_area("Body", height=200, placeholder="Email body content...")
            from_email = st.text_input("From (optional)", placeholder="sender@example.com")

            col1, col2 = st.columns([3, 1])
            with col2:
                submitted = st.form_submit_button("🚀 Analyze", use_container_width=True)

        if submitted:
            if not client_id or not client_secret:
                st.warning("Please enter Horizon API credentials.")
            elif not body:
                st.warning("Please enter email body.")
            else:
                with st.spinner("Analyzing email..."):
                    response = api_analyze({
                        "subject":       subject,
                        "body":          body,
                        "from_email":    from_email or None,
                        "client_id":     client_id,
                        "client_secret": client_secret,
                    })
                if response:
                    st.session_state.last_response = response
                    render_results(response)

    # ── File Upload Tab ───────────────────────────────────────────────────────
    with tab_upload:
        uploaded = st.file_uploader(
            "Upload .eml or .msg file",
            type=["eml", "msg"],
            help="Supports RFC822 .eml and Outlook .msg formats",
        )

        if uploaded:
            st.caption(f"📎 File: `{uploaded.name}` ({uploaded.size} bytes)")

            if st.button("🚀 Analyze File", key="upload_btn"):
                if not client_id or not client_secret:
                    st.warning("Please enter Horizon API credentials.")
                else:
                    with st.spinner(f"Processing {uploaded.name}..."):
                        response = api_upload(
                            uploaded.read(),
                            uploaded.name,
                            client_id,
                            client_secret,
                        )
                    if response:
                        st.session_state.last_response = response
                        render_results(response)


# ══════════════════════════════════════════════════════════════════════════════
# APP ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    init_session()

    if not is_logged_in():
        render_login()
        return

    render_sidebar()
    page_analyze()

    # Show last results + logs if available
    if st.session_state.last_response and st.session_state.last_request_id:
        render_logs(st.session_state.last_request_id)


if __name__ == "__main__":
    main()
