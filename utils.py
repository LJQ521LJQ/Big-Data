import os
from pathlib import Path

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def resolve_path(p: str) -> str:
    return str(Path(p).expanduser().absolute())
