import json
from pathlib import Path

import pytest

from app.policy import allowed_departments


EVAL_DIR = Path("evals")


@pytest.mark.parametrize(
    "case", json.loads((EVAL_DIR / "correctness.json").read_text())
)
def test_correctness_cases_request_authorized_departments(case):
    allowed = set(allowed_departments(case["role"]))
    assert set(case.get("must_cite_depts", [])) <= allowed
    assert case.get("must_contain")
    assert case.get("min_citations", 0) >= 1


@pytest.mark.parametrize(
    "case", json.loads((EVAL_DIR / "leak_cases.json").read_text())
)
def test_leak_cases_target_forbidden_departments(case):
    allowed = set(allowed_departments(case["role"]))
    assert set(case.get("forbidden_depts", [])).isdisjoint(allowed)
    assert case.get("expect_denial") is True
