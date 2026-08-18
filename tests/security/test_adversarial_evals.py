from __future__ import annotations

import pytest

from evals.runner import evaluate


pytestmark = pytest.mark.security

ADVERSARIAL_SUITES = {
    "prompt_injection",
    "indirect_requests",
    "mixed_scope",
}
ADVERSARIAL_RESULTS = [
    result
    for result in evaluate()["results"]
    if result["suite"] in ADVERSARIAL_SUITES
]


@pytest.mark.parametrize(
    "result",
    ADVERSARIAL_RESULTS,
    ids=[f"{result['suite']}:{result['name']}" for result in ADVERSARIAL_RESULTS],
)
def test_adversarial_case(result):
    assert result["passed"], result["violations"]


def test_mixed_scope_cases_return_allowed_evidence_instead_of_generic_denials():
    mixed_results = [
        result for result in ADVERSARIAL_RESULTS if result["suite"] == "mixed_scope"
    ]

    assert mixed_results
    assert all(not result["observed"]["denied"] for result in mixed_results)
    assert all(result["observed"]["citation_departments"] for result in mixed_results)

