from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.policy import allowed_departments, allowed_sensitivities


pytestmark = pytest.mark.unit

EXPECTED_MATRIX = {
    "employee": {"general"},
    "finance": {"finance", "general"},
    "marketing": {"marketing", "general"},
    "hr": {"hr", "general"},
    "engineering": {"engineering", "general"},
    "clevel": {"finance", "marketing", "hr", "engineering", "general"},
}


@pytest.mark.parametrize(
    ("role", "expected"),
    EXPECTED_MATRIX.items(),
    ids=EXPECTED_MATRIX,
)
def test_role_policy_matches_independent_matrix(role, expected):
    assert set(allowed_departments(role)) == expected


@pytest.mark.parametrize("role", ["", "guest", "unknown", None])
def test_unknown_roles_fail_closed(role):
    assert allowed_departments(role) == []
    assert allowed_sensitivities(role) == set()


def test_cross_department_fixture_is_a_complete_matrix():
    cases = json.loads(Path("evals/cross_department.json").read_text())
    departments = {"finance", "marketing", "hr", "engineering", "general"}
    observed = {(case["role"], case["target_department"]) for case in cases}
    expected = {
        (role, department)
        for role in EXPECTED_MATRIX
        for department in departments
    }

    assert observed == expected
    assert len(cases) == len(expected) == 30
    for case in cases:
        assert case["allowed"] is (
            case["target_department"] in EXPECTED_MATRIX[case["role"]]
        )

