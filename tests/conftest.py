from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)

if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

# Deterministic tiers inject index/model fakes and never depend on local services.
# Assign rather than setdefault so developer shell settings cannot change results.
os.environ["AUTO_INDEX"] = "0"
os.environ["VECTOR_DB"] = "chroma"
os.environ["INTENT_NARROWING"] = "1"
os.environ["PASSAGE_SELECTION"] = "0"
os.environ["RERANK_CE"] = "0"
os.environ["USE_KEYWORD_FALLBACK"] = "0"
os.environ["QDRANT_SCORE_MIN"] = "0.25"
os.environ["CHROMA_DIST_MAX"] = "0.75"
os.environ["LEXICAL_MIN"] = "0.04"
