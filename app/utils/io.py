from pathlib import Path
from typing import Optional

# Optional dependency (only needed for CSV)
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # CSV fallback below

def read_file(path: Path) -> str:
    """
    Return text content for supported file types.
    Unsupported/binary files return "" so they are skipped during indexing.
    """
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".csv":
        # Prefer pretty CSV-as-text for retrieval
        if pd is not None:
            try:
                return pd.read_csv(path).to_csv(index=False)
            except Exception:
                pass
        # Fallback: raw text if pandas missing or parse fails
        return path.read_text(encoding="utf-8", errors="ignore")

    # Ignore images, binaries, etc.
    return ""
