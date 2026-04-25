from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = SCRIPTS_DIR / "data"


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def scripts_path(*parts: str) -> Path:
    return SCRIPTS_DIR.joinpath(*parts)


def data_path(*parts: str) -> Path:
    return DATA_DIR.joinpath(*parts)


def resolve_path(path_like: PathLike, base: str = "scripts", must_exist: bool = False) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path

    if base == "project":
        candidates = [PROJECT_ROOT / path, SCRIPTS_DIR / path]
    elif base == "data":
        candidates = [DATA_DIR / path, SCRIPTS_DIR / path, PROJECT_ROOT / path]
    else:
        candidates = [SCRIPTS_DIR / path, PROJECT_ROOT / path]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0] if not must_exist else candidates[-1]
