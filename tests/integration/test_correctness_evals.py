from __future__ import annotations

import pytest

from evals.runner import evaluate


pytestmark = pytest.mark.integration

CORRECTNESS_RESULTS = [
    result for result in evaluate()["results"] if result["suite"] == "correctness"
]


@pytest.mark.parametrize(
    "result",
    CORRECTNESS_RESULTS,
    ids=[result["name"] for result in CORRECTNESS_RESULTS],
)
def test_deterministic_correctness_case(result):
    assert result["passed"], result["violations"]

