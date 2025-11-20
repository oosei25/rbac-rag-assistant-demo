from pathlib import Path
import pandas as pd

def read_file(path: Path) -> str:
    if path.suffix.lower() in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return df.to_csv(index=False)  # stringify table
    return ""
