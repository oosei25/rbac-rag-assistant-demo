from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)

if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

# Tests inject deterministic index/model fakes and never depend on local services.
os.environ.setdefault("AUTO_INDEX", "0")
os.environ.setdefault("VECTOR_DB", "chroma")
