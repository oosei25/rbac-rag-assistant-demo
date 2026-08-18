from __future__ import annotations

import pytest

from evals.runner import evaluate


pytestmark = pytest.mark.security

MATRIX_RESULTS = [
    result
    for result in evaluate()["results"]
    if result["suite"] == "cross_department"
]


@pytest.mark.parametrize(
    "result",
    MATRIX_RESULTS,
    ids=[result["name"] for result in MATRIX_RESULTS],
)
def test_complete_role_department_matrix(result):
    assert result["passed"], result["violations"]


def test_matrix_contains_positive_controls_that_prevent_deny_all_success():
    allowed_results = [
        result for result in MATRIX_RESULTS if not result["observed"]["denied"]
    ]

    assert len(MATRIX_RESULTS) == 30
    assert len(allowed_results) == 14
    assert all(result["observed"]["citation_departments"] for result in allowed_results)
