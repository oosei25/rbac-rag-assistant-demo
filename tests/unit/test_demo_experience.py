from __future__ import annotations

import pytest

from ui_components import (
    DEMO_USERS,
    ROLE_PRESET_QUESTIONS,
    SECURITY_LAB_PRESETS,
    source_card_view,
)


pytestmark = pytest.mark.unit


def test_demo_picker_and_presets_cover_every_representative_role():
    expected_roles = {
        "employee",
        "finance",
        "marketing",
        "hr",
        "engineering",
        "clevel",
    }

    assert {user["role"] for user in DEMO_USERS} == expected_roles
    assert set(ROLE_PRESET_QUESTIONS) == expected_roles
    assert all(ROLE_PRESET_QUESTIONS[role] for role in expected_roles)


def test_security_lab_contains_requested_comparisons():
    pairs = {
        (preset["left_role"], preset["right_role"])
        for preset in SECURITY_LAB_PRESETS
    }

    assert pairs == {
        ("marketing", "hr"),
        ("employee", "hr"),
        ("marketing", "finance"),
        ("engineering", "clevel"),
    }


def test_source_card_view_omits_path_and_internal_document_id():
    card = source_card_view(
        {
            "citation_id": 2,
            "document_id": "private-id",
            "path": "/app/resources/data/hr/private.csv",
            "title": "Benefits",
            "department": "hr",
            "section": "Overview",
            "score": 0.875,
            "snippet": "Authorized excerpt",
        }
    )

    assert card == {
        "citation_id": 2,
        "title": "Benefits",
        "department": "hr",
        "section": "Overview",
        "score": 0.875,
        "snippet": "Authorized excerpt",
    }
    assert "path" not in card
    assert "document_id" not in card

