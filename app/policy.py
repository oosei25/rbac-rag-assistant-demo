import re
from typing import List, Set

ROLE_TO_DEPTS = {
    "finance": {"finance", "general"},
    "marketing": {"marketing", "general"},
    "hr": {"hr", "general"},
    "engineering": {"engineering", "general"},
    "clevel": {"finance", "marketing", "hr", "engineering", "general"},
    "employee": {"general"},
}

DEPT_KEYWORDS = {
    "finance": {
        "revenue","budget","invoice","reimbursement","capex","opex","quarterly financial","cost",
    },
    "marketing": {"campaign","roi","roas","ctr","leads","brand","social","market research"},
    "hr": {"payroll","benefits","recruit","hiring","performance review","leave","attendance","compensation"},
    "engineering": {"architecture","deploy","api","service","microservice","infra","runtime","design"},
    "general": {"policy","handbook","event","faq","holiday","company"},
}

SENS_BY_ROLE = {
    "finance": {"internal"},
    "marketing": {"internal"},
    "hr": {"internal"},
    "engineering": {"internal"},
    "clevel": {"internal"},        # extend if sensitivity tiers added later
    "employee": {"internal"},
}

_WORD_BOUNDARY_TOKENS = {"api","roi","hr"}  # short tokens prone to false positives

def allowed_departments(role: str) -> List[str]:
    return sorted(ROLE_TO_DEPTS.get(role, {"general"}))

def allowed_sensitivities(role: str) -> Set[str]:
    return set(SENS_BY_ROLE.get(role, {"internal"}))

def _normalize(s: str) -> str:
    s = s.lower()
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _match_term(q: str, term: str) -> bool:
    term_n = _normalize(term)
    if term_n in _WORD_BOUNDARY_TOKENS:
        # whole-word match for short tokens
        return re.search(rf"\b{re.escape(term_n)}\b", q) is not None
    # substring is fine for multi-word phrases (e.g., "market research")
    return term_n in q

def infer_requested_departments(query: str) -> Set[str]:
    q = _normalize(query)
    hits = set()
    for dept, kws in DEPT_KEYWORDS.items():
        if any(_match_term(q, k) for k in kws) or re.search(rf"\b{dept}\b", q):
            hits.add(dept)
    return hits or {"general"}  # default to general if none detected