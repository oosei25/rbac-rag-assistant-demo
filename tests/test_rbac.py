import pytest

from app.policy import allowed_departments, allowed_sensitivities


@pytest.mark.parametrize(
    ("role", "departments"),
    [
        ("employee", ["general"]),
        ("finance", ["finance", "general"]),
        ("clevel", ["engineering", "finance", "general", "hr", "marketing"]),
    ],
)
def test_known_role_access(role, departments):
    assert allowed_departments(role) == departments


@pytest.mark.parametrize("role", ["", "guest", "unknown", None])
def test_unknown_roles_have_no_access(role):
    assert allowed_departments(role) == []
    assert allowed_sensitivities(role) == set()
