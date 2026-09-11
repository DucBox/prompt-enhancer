"""Đọc cấu hình từ file .env — không phụ thuộc python-dotenv.

Mọi thông tin nhạy cảm (endpoint nội bộ, API key) chỉ nằm trong .env,
file này đã được .gitignore nên không bao giờ lên GitHub.
Xem .env.example để biết các biến cần điền.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

# code/method_1/common/config.py -> code/method_1
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
# code/method_1 -> prompt-enhance
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

_loaded = False


def _parse_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        out[key] = value
    return out


def load_env(env_file: Optional[str] = None, *, force: bool = False) -> Dict[str, str]:
    """Nạp .env vào os.environ. Biến môi trường sẵn có luôn thắng file."""
    global _loaded
    if _loaded and not force:
        return dict(os.environ)

    candidates = []
    if env_file:
        candidates.append(Path(env_file))
    else:
        candidates += [PACKAGE_ROOT / ".env", PROJECT_ROOT / ".env", Path.cwd() / ".env"]

    for path in candidates:
        if path.is_file():
            for key, value in _parse_env_file(path).items():
                os.environ.setdefault(key, value)
            break

    _loaded = True
    return dict(os.environ)


def get(name: str, default: Optional[str] = None) -> Optional[str]:
    load_env()
    value = os.environ.get(name, default)
    return value if value != "" else default


def require(name: str) -> str:
    value = get(name)
    if not value:
        raise RuntimeError(
            f"Thiếu biến môi trường '{name}'.\n"
            f"Hãy copy .env.example thành .env rồi điền giá trị:\n"
            f"    cp {PACKAGE_ROOT / '.env.example'} {PACKAGE_ROOT / '.env'}"
        )
    return value


def get_int(name: str, default: int) -> int:
    value = get(name)
    return int(value) if value is not None else default


def get_float(name: str, default: float) -> float:
    value = get(name)
    return float(value) if value is not None else default
