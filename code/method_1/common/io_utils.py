"""Tiện ích dùng chung: đọc/ghi JSON & JSONL, chọn mẫu test, CLI chung."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

DEFAULT_SEED = 3407


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError("{}:{} không phải JSON hợp lệ: {}".format(path, line_no, exc)) from exc
    return rows


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def sample_for_test(items: Sequence[Any], n: int, seed: int = DEFAULT_SEED) -> List[Any]:
    """Lấy ngẫu nhiên n phần tử, có seed nên lặp lại được giữa các bước."""
    if n >= len(items):
        return list(items)
    rng = random.Random(seed)
    return rng.sample(list(items), n)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Cờ dùng chung cho mọi step."""
    parser.add_argument(
        "--test", action="store_true",
        help="Chế độ thử: chỉ chạy trên một ít mẫu ngẫu nhiên thay vì toàn bộ",
    )
    parser.add_argument(
        "--test_samples", type=int, default=20,
        help="Số mẫu khi bật --test (mặc định 20)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--env_file", default=None, help="Đường dẫn .env tuỳ chọn")
    return parser


def apply_test_mode(items: Sequence[Any], args: argparse.Namespace, label: str = "mẫu") -> List[Any]:
    """Cắt danh sách theo --test và in ra chế độ đang chạy."""
    if getattr(args, "test", False):
        picked = sample_for_test(items, args.test_samples, args.seed)
        print("[TEST] chạy {}/{} {} (seed={})".format(len(picked), len(items), label, args.seed))
        return picked
    print("[FULL] chạy toàn bộ {} {}".format(len(items), label))
    return list(items)


def banner(step: str, title: str) -> None:
    line = "=" * 78
    print("\n{}\n{}  {}\n{}".format(line, step, title, line), flush=True)


def summary(**fields: Any) -> None:
    print("\n--- Tổng kết ---")
    width = max((len(k) for k in fields), default=0)
    for key, value in fields.items():
        print("  {}: {}".format(key.ljust(width), value))
    print()
