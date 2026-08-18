from __future__ import annotations

import json

import pytest

from evals.runner import evaluate, write_reports


pytestmark = pytest.mark.integration


def test_reports_are_machine_readable_human_readable_and_reproducible(tmp_path):
    report = evaluate()
    json_path, markdown_path = write_reports(report, tmp_path / "first")
    second_json, second_markdown = write_reports(report, tmp_path / "second")

    parsed = json.loads(json_path.read_text())
    markdown = markdown_path.read_text()

    assert parsed["metrics"]["total_cases"] == len(parsed["results"])
    assert parsed["metrics"]["policy_cases"] == 30
    assert parsed["metrics"]["passed"] + parsed["metrics"]["failed"] == len(
        parsed["results"]
    )
    assert "| Unauthorized source leaks |" in markdown
    assert "| Thread-isolation failures |" in markdown
    assert json_path.read_bytes() == second_json.read_bytes()
    assert markdown_path.read_bytes() == second_markdown.read_bytes()

