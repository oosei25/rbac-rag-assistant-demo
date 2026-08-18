from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from app.services.rag_helpers import DENY_MESSAGE, looks_like_deny
from app.services.threads import scoped_thread_id
from evals.harness import DEFAULT_DOCUMENTS, build_harness


EVAL_DIR = Path(__file__).resolve().parent
SUITES = (
    "correctness",
    "cross_department",
    "prompt_injection",
    "indirect_requests",
    "mixed_scope",
    "citation_integrity",
    "thread_isolation",
)


def load_cases(eval_dir: Path = EVAL_DIR) -> dict[str, list[dict]]:
    return {
        suite: json.loads((eval_dir / f"{suite}.json").read_text(encoding="utf-8"))
        for suite in SUITES
    }


def _violation(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def _case_result(
    suite: str,
    case: dict,
    violations: list[dict[str, str]],
    observed: dict,
) -> dict:
    return {
        "suite": suite,
        "name": case["name"],
        "passed": not violations,
        "violations": violations,
        "observed": observed,
    }


def _answer_case(suite: str, case: dict) -> dict:
    harness = build_harness()
    answer, citations = harness.service.generate(case["query"], case["role"])
    cited_departments = [citation.department for citation in citations]
    cited_ids = [citation.citation_id for citation in citations]
    violations: list[dict[str, str]] = []
    denied = looks_like_deny(answer)

    if suite == "correctness":
        if denied:
            violations.append(
                _violation("unexpected_denial", "Authorized correctness case was denied")
            )
        for expected in case.get("must_contain", []):
            if expected.lower() not in answer.lower():
                violations.append(
                    _violation("missing_expected_content", f"Answer omitted {expected!r}")
                )
        allowed_citations = set(case.get("must_cite_depts", []))
        if not set(cited_departments) <= allowed_citations:
            violations.append(
                _violation(
                    "unexpected_source_department",
                    f"Cited {cited_departments}, expected only {sorted(allowed_citations)}",
                )
            )
        if len(citations) < case.get("min_citations", 1):
            violations.append(
                _violation("missing_citation", "Authorized answer lacks required citations")
            )

    elif suite == "cross_department":
        target = case["target_department"]
        if case["allowed"]:
            if denied:
                violations.append(
                    _violation("unexpected_denial", "Allowed matrix cell was denied")
                )
            if case["evidence"] not in answer:
                violations.append(
                    _violation(
                        "missing_expected_content",
                        f"Allowed evidence {case['evidence']} was not returned",
                    )
                )
            if target not in cited_departments:
                violations.append(
                    _violation(
                        "missing_citation",
                        f"Allowed department {target} was not cited",
                    )
                )
        else:
            if not denied:
                violations.append(
                    _violation("expected_denial", "Forbidden matrix cell returned an answer")
                )
            if target in cited_departments:
                violations.append(
                    _violation(
                        "unauthorized_source_leak",
                        f"Forbidden department {target} appeared in citations",
                    )
                )
            if case["evidence"] in answer:
                violations.append(
                    _violation(
                        "unauthorized_content_leak",
                        f"Forbidden evidence {case['evidence']} appeared in the answer",
                    )
                )

    else:
        expect_denial = case.get("expect_denial", False)
        if expect_denial and not denied:
            violations.append(
                _violation("expected_denial", "Adversarial request returned an answer")
            )
        if not expect_denial and denied:
            violations.append(
                _violation("unexpected_denial", "Allowed part of adversarial request was denied")
            )
        for expected in case.get("required_evidence", []):
            if expected not in answer:
                violations.append(
                    _violation("missing_expected_content", f"Answer omitted {expected!r}")
                )
        required_depts = set(case.get("required_cite_depts", []))
        if required_depts and not required_depts <= set(cited_departments):
            violations.append(
                _violation(
                    "missing_citation",
                    f"Required citation departments {sorted(required_depts)} were absent",
                )
            )
        forbidden_depts = set(case.get("forbidden_depts", []))
        leaked_depts = forbidden_depts & set(cited_departments)
        if leaked_depts:
            violations.append(
                _violation(
                    "unauthorized_source_leak",
                    f"Forbidden citation departments: {sorted(leaked_depts)}",
                )
            )
        leaked_evidence = [
            evidence
            for evidence in case.get("forbidden_evidence", [])
            if evidence in answer
        ]
        if leaked_evidence:
            violations.append(
                _violation(
                    "unauthorized_content_leak",
                    f"Forbidden evidence in answer: {leaked_evidence}",
                )
            )
        if expect_denial and citations:
            violations.append(
                _violation("unexpected_source", "Denied answer returned citations")
            )

    return _case_result(
        suite,
        case,
        violations,
        {
            "denied": denied,
            "citation_ids": cited_ids,
            "citation_departments": cited_departments,
        },
    )


def _citation_case(case: dict) -> dict:
    documents = [
        document
        for department in case["document_departments"]
        for document in DEFAULT_DOCUMENTS
        if document["department"] == department
    ]
    harness = build_harness(documents=documents, responses=[case["llm_response"]])
    answer, citations = harness.service.generate_from_documents(case["query"], documents)
    denied = looks_like_deny(answer)
    actual_ids = [citation.citation_id for citation in citations]
    violations: list[dict[str, str]] = []
    if denied != case["expect_denial"]:
        violations.append(
            _violation(
                "citation_validation_failure",
                f"Expected denial={case['expect_denial']}, observed denial={denied}",
            )
        )
    if actual_ids != case["expected_citation_ids"]:
        violations.append(
            _violation(
                "citation_validation_failure",
                f"Expected citation IDs {case['expected_citation_ids']}, got {actual_ids}",
            )
        )
    return _case_result(
        "citation_integrity",
        case,
        violations,
        {"denied": denied, "citation_ids": actual_ids},
    )


def _thread_case(case: dict) -> dict:
    left = scoped_thread_id(case["left_username"], case["left_thread_id"])
    right = scoped_thread_id(case["right_username"], case["right_thread_id"])
    observed_equal = left == right
    violations = []
    if observed_equal != case["expect_equal"]:
        violations.append(
            _violation(
                "thread_isolation_failure",
                f"Expected equality={case['expect_equal']}, observed {observed_equal}",
            )
        )
    return _case_result(
        "thread_isolation",
        case,
        violations,
        {"keys_equal": observed_equal},
    )


def _fixture_digest(eval_dir: Path) -> str:
    digest = hashlib.sha256()
    for suite in SUITES:
        fixture = eval_dir / f"{suite}.json"
        digest.update(fixture.name.encode("utf-8"))
        digest.update(fixture.read_bytes())
    return digest.hexdigest()


def evaluate(eval_dir: Path = EVAL_DIR) -> dict:
    cases = load_cases(eval_dir)
    results = []
    answer_suites = {
        "correctness",
        "cross_department",
        "prompt_injection",
        "indirect_requests",
        "mixed_scope",
    }
    for suite in SUITES:
        for case in cases[suite]:
            if suite in answer_suites:
                results.append(_answer_case(suite, case))
            elif suite == "citation_integrity":
                results.append(_citation_case(case))
            else:
                results.append(_thread_case(case))

    suite_totals = Counter(result["suite"] for result in results)
    suite_passes = Counter(
        result["suite"] for result in results if result["passed"]
    )
    violation_codes = Counter(
        violation["code"]
        for result in results
        for violation in result["violations"]
    )
    leakage_cases = sum(
        1
        for result in results
        if any(
            violation["code"]
            in {"unauthorized_source_leak", "unauthorized_content_leak"}
            for violation in result["violations"]
        )
    )
    total = len(results)
    passed = sum(result["passed"] for result in results)
    metrics = {
        "total_cases": total,
        "policy_cases": suite_totals["cross_department"],
        "passed": passed,
        "failed": total - passed,
        "unauthorized_source_leaks": violation_codes["unauthorized_source_leak"],
        "unauthorized_content_leaks": violation_codes["unauthorized_content_leak"],
        "citation_failures": sum(
            not result["passed"]
            for result in results
            if result["suite"] == "citation_integrity"
        ),
        "thread_isolation_failures": violation_codes["thread_isolation_failure"],
        "leakage_cases": leakage_cases,
        "leakage_rate": round(leakage_cases / total, 6) if total else 0.0,
    }
    by_suite = {
        suite: {
            "total": suite_totals[suite],
            "passed": suite_passes[suite],
            "failed": suite_totals[suite] - suite_passes[suite],
        }
        for suite in SUITES
    }
    return {
        "schema_version": 1,
        "fixture_digest": _fixture_digest(eval_dir),
        "metrics": metrics,
        "suites": by_suite,
        "results": results,
    }


def render_markdown(report: dict) -> str:
    metrics = report["metrics"]
    lines = [
        "# Deterministic Security Evaluation",
        "",
        f"Fixture digest: `{report['fixture_digest']}`",
        "",
        "## Measured results",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    labels = (
        ("Total cases", "total_cases"),
        ("Policy matrix cases", "policy_cases"),
        ("Passed", "passed"),
        ("Failed", "failed"),
        ("Unauthorized source leaks", "unauthorized_source_leaks"),
        ("Unauthorized content leaks", "unauthorized_content_leaks"),
        ("Citation failures", "citation_failures"),
        ("Thread-isolation failures", "thread_isolation_failures"),
        ("Leakage rate (leakage cases / total cases)", "leakage_rate"),
    )
    lines.extend(f"| {label} | {metrics[key]} |" for label, key in labels)
    lines.extend(
        [
            "",
            "## Results by suite",
            "",
            "| Suite | Cases | Passed | Failed |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for suite, values in report["suites"].items():
        lines.append(
            f"| {suite} | {values['total']} | {values['passed']} | {values['failed']} |"
        )
    failures = [result for result in report["results"] if not result["passed"]]
    lines.extend(["", "## Failures", ""])
    if not failures:
        lines.append("None.")
    else:
        for result in failures:
            details = "; ".join(
                f"{item['code']}: {item['detail']}" for item in result["violations"]
            )
            lines.append(f"- `{result['suite']}/{result['name']}` — {details}")
    lines.append("")
    return "\n".join(lines)


def write_reports(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "security-evaluation.json"
    markdown_path = output_dir / "security-evaluation.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic RBAC and RAG security evaluations."
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=EVAL_DIR,
        help="Directory containing evaluation JSON fixtures.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/security-evals"),
        help="Directory for JSON and Markdown reports.",
    )
    args = parser.parse_args()
    report = evaluate(args.eval_dir)
    json_path, markdown_path = write_reports(report, args.output_dir)
    print(json.dumps(report["metrics"], sort_keys=True))
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 1 if report["metrics"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
