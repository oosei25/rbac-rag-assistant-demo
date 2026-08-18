from __future__ import annotations

from typing import Any


DEMO_USERS = [
    {
        "label": "Emma — Employee",
        "username": "Emma",
        "password": "password",
        "role": "employee",
    },
    {
        "label": "Sam — Finance",
        "username": "Sam",
        "password": "financepass",
        "role": "finance",
    },
    {
        "label": "Mariam — Marketing",
        "username": "Mariam",
        "password": "mariampass123",
        "role": "marketing",
    },
    {
        "label": "Natasha — HR",
        "username": "Natasha",
        "password": "hrpass123",
        "role": "hr",
    },
    {
        "label": "Peter — Engineering",
        "username": "Peter",
        "password": "pete123",
        "role": "engineering",
    },
    {
        "label": "Cathy — C-level",
        "username": "Cathy",
        "password": "cathyceo",
        "role": "clevel",
    },
]

ROLE_LABELS = {
    "employee": "Employee",
    "finance": "Finance",
    "marketing": "Marketing",
    "hr": "HR",
    "engineering": "Engineering",
    "clevel": "C-level",
}

ROLE_PRESET_QUESTIONS = {
    "employee": [
        "What is the employee handbook holiday policy?",
        "Summarize the general company policies available to employees.",
    ],
    "finance": [
        "Summarize the latest quarterly financial results.",
        "What does the finance report say about revenue and reimbursements?",
    ],
    "marketing": [
        "Summarize the latest marketing report and campaign results.",
        "What does the Q4 market report say about campaign performance?",
    ],
    "hr": [
        "What does the HR data say about payroll and compensation?",
        "Summarize the available benefits and performance review information.",
    ],
    "engineering": [
        "Explain the engineering service architecture and API design.",
        "Summarize the deployment and infrastructure guidance.",
    ],
    "clevel": [
        "Compare the latest finance and marketing results.",
        "Summarize key updates across HR, engineering, finance, and marketing.",
    ],
}

SECURITY_LAB_PRESETS = [
    {
        "label": "Marketing vs HR — payroll",
        "question": "What does the payroll data say about compensation?",
        "left_role": "marketing",
        "right_role": "hr",
    },
    {
        "label": "Employee vs HR — compensation",
        "question": "Summarize the available compensation information.",
        "left_role": "employee",
        "right_role": "hr",
    },
    {
        "label": "Marketing vs Finance — financial results",
        "question": "What are the latest financial revenue results?",
        "left_role": "marketing",
        "right_role": "finance",
    },
    {
        "label": "Engineering vs C-level — cross-department",
        "question": "Compare engineering architecture with finance revenue results.",
        "left_role": "engineering",
        "right_role": "clevel",
    },
]

ACCESS_REASON_LABELS = {
    "grounded_authorized_answer": "Grounded answer produced from authorized context.",
    "role_not_authorized": "The authenticated role has no configured document access.",
    "no_authorized_relevant_context": "No relevant context remained inside the role's access boundary.",
    "model_returned_no_grounded_answer": "The model could not ground an answer in the authorized context.",
    "answer_failed_citation_validation": "The answer was withheld because its citations did not validate.",
}


def source_card_view(citation: dict[str, Any]) -> dict[str, Any]:
    """Return only fields intended for an authorized source card."""
    return {
        "citation_id": citation.get("citation_id", "?"),
        "title": citation.get("title") or "Authorized document",
        "department": citation.get("department") or "unknown",
        "section": citation.get("section") or "Document",
        "score": float(citation.get("score", 0.0) or 0.0),
        "snippet": citation.get("snippet") or "",
    }


def render_source_cards(citations: list[dict[str, Any]]) -> None:
    import streamlit as st

    if not citations:
        st.caption("No authorized sources were returned.")
        return
    st.subheader(f"Authorized sources ({len(citations)})")
    for citation in citations:
        card = source_card_view(citation)
        with st.container(border=True):
            st.markdown(f"**[{card['citation_id']}] {card['title']}**")
            st.caption(
                f"{card['department']} · {card['section']} · "
                f"relevance {card['score']:.3f}"
            )
            with st.expander("View authorized excerpt"):
                st.write(card["snippet"])


def render_access_trace(trace: dict[str, Any] | None) -> None:
    import streamlit as st

    if not trace:
        return
    counts = trace.get("candidate_counts") or {}
    applied_filter = trace.get("applied_filter") or {}
    reason = trace.get("reason", "")
    decision = trace.get("decision", "denied")
    with st.expander("Access decision trace", expanded=False):
        summary = st.columns(3)
        summary[0].metric("Authenticated role", trace.get("authenticated_role", "unknown"))
        summary[1].metric("Decision", decision.title())
        summary[2].metric(
            "Authorized sources", trace.get("authorized_source_count", 0)
        )
        st.write(
            "**Allowed departments:** "
            + ", ".join(trace.get("allowed_departments") or ["none"])
        )
        st.write(
            "**Inferred request:** "
            + ", ".join(trace.get("requested_departments") or ["general"])
        )
        initial = applied_filter.get("initial_departments") or []
        fallback = applied_filter.get("fallback_departments") or []
        st.write("**Applied filter:** " + (", ".join(initial) or "deny all"))
        if fallback:
            st.caption("Authorized fallback filter: " + ", ".join(fallback))
        candidate_columns = st.columns(3)
        candidate_columns[0].metric(
            "After policy", counts.get("authorized_after_policy", 0)
        )
        candidate_columns[1].metric(
            "After relevance", counts.get("authorized_after_relevance", 0)
        )
        candidate_columns[2].metric(
            "Selected", counts.get("selected_for_generation", 0)
        )
        if decision == "answered":
            st.success(ACCESS_REASON_LABELS.get(reason, reason))
        else:
            st.warning(ACCESS_REASON_LABELS.get(reason, reason))


__all__ = [
    "DEMO_USERS",
    "ROLE_LABELS",
    "ROLE_PRESET_QUESTIONS",
    "SECURITY_LAB_PRESETS",
    "render_access_trace",
    "render_source_cards",
    "source_card_view",
]
