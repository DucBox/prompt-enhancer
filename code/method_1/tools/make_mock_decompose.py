#!/usr/bin/env python3
"""CÔNG CỤ DEV — tạo decompose.jsonl GIẢ để soi đầu vào của step 1b khi chưa có server.

KHÔNG dùng cho dữ liệu thật. Việc gom nhóm và phân rã ở đây làm bằng luật thô, không
phải bằng model. Step 1b giờ gọi model, nên mock chỉ đủ để chạy `--dry_run` của 1b.

    python tools/make_mock_decompose.py --limit 50
    python step1b_build_subjson.py --decompose_file output/_mock/decompose.jsonl --dry_run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import io_utils, schema  # noqa: E402

PERSON_WORDS = re.compile(
    r"\b(man|men|woman|women|boy|girl|girls|boys|person|people|child|children|vendor|vendors"
    r"|tourist|tourists|passenger|passengers|nam|nữ|cô|anh|chị|em|người)\b", re.I)
VAGUE_WORDS = re.compile(
    r"\b(several|numerous|multiple|many|various|a crowd of|a cluster of|a group of|some)\b", re.I)
BACKDROP_WORDS = re.compile(
    r"\b(tree|trees|sky|cloud|clouds|wall|floor|ground|building|road|grass|leaf|leaves)\b", re.I)


def head_noun(desc: str) -> str:
    words = re.sub(r"^(a|an|the)\s+", "", desc.strip(), flags=re.I).split()
    return " ".join(words[:3]).rstrip(",.").lower() or "vật thể"


def split_facts(desc: str) -> List[Dict[str, Any]]:
    """Chẻ desc thành mệnh đề bằng dấu câu — thô, chỉ đủ để smoke test."""
    chunks = [c.strip().rstrip(".") for c in re.split(r"[,;]| with | featuring ", desc) if c.strip()]
    facts = [{"rank": 0, "kind": "subject", "text": head_noun(desc)}]
    for i, chunk in enumerate(chunks[1:], start=1):
        words = chunk.split()
        facts.append({
            "rank": i,
            "kind": "position" if re.search(r"\b(left|right|center|top|bottom|behind)\b", chunk, re.I)
                    else "attribute",
            "text": " ".join(words[:6]).lower(),
        })
    return facts[:8]


def mock_decompose(target: Dict[str, Any]) -> Dict[str, Any]:
    elements = schema.elements_of(target)
    facts = {str(el["id"]): split_facts(str(el.get("desc", ""))) for el in elements}

    person_ids, backdrop_ids, other_ids = [], [], []
    vague = False
    for el in elements:
        desc = str(el.get("desc", ""))
        if VAGUE_WORDS.search(desc):
            vague = True
        if PERSON_WORDS.search(desc):
            person_ids.append(el["id"])
        elif BACKDROP_WORDS.search(desc):
            backdrop_ids.append(el["id"])
        else:
            other_ids.append(el["id"])

    groups: List[Dict[str, Any]] = []
    if other_ids:
        groups.append({"name": "chủ thể chính", "member_ids": other_ids[:3],
                       "cardinality": "exact:{}".format(min(3, len(other_ids))), "cultural": False})
    if person_ids:
        groups.append({
            "name": "nhóm người", "member_ids": person_ids,
            "cardinality": "vague" if vague else "exact:{}".format(len(person_ids)),
            "cultural": False,
        })
    if len(other_ids) > 3:
        groups.append({"name": "vật thể phụ", "member_ids": other_ids[3:],
                       "cardinality": "vague", "cultural": False})
    if backdrop_ids:
        groups.append({"name": "cảnh nền", "member_ids": backdrop_ids,
                       "cardinality": "vague", "cultural": False})
    if not groups:
        groups.append({"name": "chủ thể chính", "member_ids": [e["id"] for e in elements],
                       "cardinality": "vague", "cultural": False})
    comp = target.get("compositional_deconstruction") or {}
    background = str(comp.get("background") or "").strip()
    background_facts = split_facts(background) if background else []

    style = target.get("style_description") or {}
    style_facts: Dict[str, List[Dict[str, Any]]] = {}
    for field in ("photo", "art_style", "lighting", "aesthetics"):
        value = str(style.get(field) or "").strip()
        if value:
            parts = [p.strip().lower() for p in value.split(",") if p.strip()]
            style_facts[field] = [{"rank": i, "text": p} for i, p in enumerate(parts)]

    field_priority = {"elements": 1, **({"background": 1} if background_facts else {}),
                      **{f: 1 for f in style_facts}}
    return {"concept_groups": groups, "facts": facts, "background_facts": background_facts,
            "style_facts": style_facts, "field_priority": field_priority}


def main() -> None:
    p = argparse.ArgumentParser(description="Tạo decompose.jsonl giả (DEV ONLY)")
    p.add_argument("--targets_file", default="output/step0_normalized/targets.jsonl")
    p.add_argument("--out_file", default="output/_mock/decompose.jsonl")
    p.add_argument("--limit", type=int, default=0, help="0 = tất cả")
    args = p.parse_args()

    print("!! MOCK — dữ liệu giả, chỉ dùng để kiểm tra đường ống chạy thông !!\n")
    rows = io_utils.read_jsonl(Path(args.targets_file))
    if args.limit:
        rows = rows[:args.limit]

    out = [{"id": r["id"], **mock_decompose(r["target_json"])} for r in rows]
    n = io_utils.write_jsonl(Path(args.out_file), out)
    print("ghi {} dòng -> {}".format(n, args.out_file))


if __name__ == "__main__":
    main()
