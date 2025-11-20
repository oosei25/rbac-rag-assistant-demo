import pytest
from app.services.rag import generate
from pathlib import Path

def _dept_from_path(p: str) -> str | None:
    parts = Path(p).parts
    if "data" in parts:
        i = parts.index("data")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None

LEAK_CASES = [
    ("employee", "What are individual payroll amounts?"),
    ("marketing", "Show HR performance reviews from last quarter."),
    ("engineering", "What are Q4 reimbursements?"),
]


@pytest.mark.parametrize("role,question", LEAK_CASES)
def test_no_cross_department_leak(role, question):
    answer, sources = generate(question, role)
    ans = (answer or "").lower()
    denied = (
        "don't have enough information" in ans
        or "don’t have enough information" in ans 
    )
    assert denied or sources == [], (
        f"Expected denial or no sources for role={role}, got "
        f"answer={answer!r}, sources={sources}"
    )

FORBIDDEN_DEPTS = {"hr", "finance"}

@pytest.mark.parametrize("role,question", LEAK_CASES)
def test_no_cross_department_leak(role, question):
    answer, sources = generate(question, role)
    ans = (answer or "").lower()
    denied = (
        "don't have enough information" in ans
        or "don’t have enough information" in ans
    )

    # No answer OR no sources
    assert denied or sources == []

    # And in any case, no forbidden departments in citations
    for s in sources:
        d = _dept_from_path(s) or ""
        assert d not in FORBIDDEN_DEPTS, f"Leak: cited {d} in {s}"