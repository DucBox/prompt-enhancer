"""Đọc & kiểm tra dữ liệu train — logic thuần, KHÔNG import torch/transformers/datasets.

Tách khỏi train_prompt_enhancer_qwen36.py để test được ở máy không có GPU
(xem tests/test_data_utils.py).

Vì sao KHÔNG dùng `datasets.load_dataset("json")` để đọc file train:
Arrow cần MỘT schema chung cho cả cột `target_json`. Dòng ảnh chụp có `photo`, dòng
tranh/3D có `art_style`, element `text` có thêm key `text` -> Arrow gộp tất cả thành một
struct và điền `null` cho key dòng đó không có. Kết quả: mọi target bị thêm
`"art_style": null` / `"photo": null` / `"text": null`, và model học sinh ra các key
null đó. Đọc từng dòng bằng `json.loads` thì dict giữ nguyên key và thứ tự như file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

REQUIRED_TOP_LEVEL = ("high_level_description", "style_description", "compositional_deconstruction")
# `id` là tay cầm nội bộ của pipeline sinh dữ liệu (step2d đã bóc). Còn `id` nghĩa là đang
# train nhầm từ step2c_split.
ALWAYS_FORBIDDEN_KEYS = {"id"}
BBOX_PALETTE_KEYS = {"bbox", "color_palette"}


def read_jsonl_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: dòng không phải JSON hợp lệ: {e}") from e
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: mỗi dòng phải là một JSON object")
            rows.append(row)
    return rows


def find_keys(obj: Any, keys: Set[str]) -> Set[str]:
    found: Set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys:
                found.add(k)
            found |= find_keys(v, keys)
    elif isinstance(obj, list):
        for x in obj:
            found |= find_keys(x, keys)
    return found


def null_paths(obj: Any, path: str = "") -> List[str]:
    if obj is None:
        return [path or "<root>"]
    items: Iterable = ()
    if isinstance(obj, dict):
        items = ((f"{path}.{k}" if path else k, v) for k, v in obj.items())
    elif isinstance(obj, list):
        items = ((f"{path}[{i}]", v) for i, v in enumerate(obj))
    out: List[str] = []
    for p, v in items:
        out += null_paths(v, p)
    return out


def target_to_minified_json(value: Any, *, validate: bool, allow_bbox_palette: bool) -> str:
    if isinstance(value, str):
        text = value.strip()
        if not validate:
            return text
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"target is not valid JSON: {e}") from e
    elif isinstance(value, dict):
        obj = value
    else:
        raise ValueError(f"target must be a JSON object or JSON string, got {type(value).__name__}")

    if not isinstance(obj, dict):
        raise ValueError("target JSON must be an object")

    if validate:
        missing = set(REQUIRED_TOP_LEVEL) - set(obj.keys())
        if missing:
            raise ValueError(f"target missing required top-level fields: {sorted(missing)}")
        comp = obj.get("compositional_deconstruction")
        if not isinstance(comp, dict) or "background" not in comp or "elements" not in comp:
            raise ValueError("compositional_deconstruction must contain background and elements")
        nulls = null_paths(obj)
        if nulls:
            raise ValueError(f"target has null values at {nulls[:5]} -- data loader đã chèn key thiếu?")
        if find_keys(obj, ALWAYS_FORBIDDEN_KEYS):
            raise ValueError("target contains 'id' -- train từ step2d_final, không phải step2c_split")
        if not allow_bbox_palette and find_keys(obj, BBOX_PALETTE_KEYS):
            raise ValueError("target contains bbox/color_palette; normalize them out before PE SFT")

    # json.dumps preserves dict insertion order and keeps Vietnamese characters literal.
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
