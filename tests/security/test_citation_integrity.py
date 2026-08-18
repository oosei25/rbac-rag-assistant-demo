from __future__ import annotations

import pytest

from evals.runner import evaluate


pytestmark = pytest.mark.security

CITATION_RESULTS = [
    result
    for result in evaluate()["results"]
    if result["suite"] == "citation_integrity"
]


@pytest.mark.parametrize(
    "result",
    CITATION_RESULTS,
    ids=[result["name"] for result in CITATION_RESULTS],
)
def test_citation_integrity_case(result):
    assert result["passed"], result["violations"]


def test_invalid_and_missing_citations_fail_closed_without_sources():
    denied = [result for result in CITATION_RESULTS if result["observed"]["denied"]]

    assert len(denied) == 2
    assert all(result["observed"]["citation_ids"] == [] for result in denied)

