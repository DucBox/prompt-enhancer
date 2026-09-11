#!/usr/bin/env python3
"""STEP 1a — Gom nhóm khái niệm & phân rã mệnh đề.  [GỌI MODEL]

Đầu vào : <out_root>/step0_normalized/targets.jsonl
Đầu ra  : <out_root>/step1a_decompose/
            decompose/<id>.json   {"concept_groups": [...], "facts": {...}}
            decompose.jsonl       gộp lại thành một file
            failures.json

Chạy một lần rồi CACHE: file <id>.json đã có thì bỏ qua, nên có thể dừng giữa chừng
và chạy lại để tiếp tục. Sau bước này mọi việc sinh dữ liệu đều là code thuần.

Ví dụ:
    python step1a_decompose.py --out_root test_1 --dry_run
    python step1a_decompose.py --out_root test_1 --workers 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import config, io_utils, llm, prompts, schema

STEP = "STEP 1a"

VALID_KINDS = {"subject", "color", "material", "attribute", "action", "position", "count"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM gom nhóm & phân rã mệnh đề")
    p.add_argument("--in_file", default=None, help="Mặc định: <out_root>/step0_normalized/targets.jsonl")
    p.add_argument("--out_dir", default=None, help="Ghi đè thư mục đầu ra của step này")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Để 0 cho deterministic — kết quả được cache và đóng băng")
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--overwrite", action="store_true", help="Bỏ qua cache, gọi lại toàn bộ")
    p.add_argument("--dry_run", action="store_true",
                   help="Chỉ dựng messages và in ra, KHÔNG gọi model")
    return io_utils.add_common_args(p).parse_args()


def validate_decomposition(result: Any, target: Dict[str, Any]) -> List[str]:
    """Kiểm tra đầu ra của model trước khi tin dùng."""
    errors: List[str] = []
    if not isinstance(result, dict):
        return ["đầu ra không phải object"]

    groups = result.get("concept_groups")
    facts = result.get("facts")

    if not isinstance(groups, list) or not groups:
        errors.append("concept_groups rỗng hoặc sai kiểu")
    if not isinstance(facts, dict) or not facts:
        errors.append("facts rỗng hoặc sai kiểu")
    if errors:
        return errors

    element_ids = {el.get("id") for el in schema.elements_of(target)}

    for i, group in enumerate(groups):
        if not isinstance(group, dict):
            errors.append("concept_groups[{}] sai kiểu".format(i))
            continue
        if not group.get("name"):
            errors.append("concept_groups[{}] thiếu name".format(i))
        card = group.get("cardinality")
        if not isinstance(card, str) or not (card == "vague" or card.startswith("exact:")):
            errors.append("concept_groups[{}].cardinality={!r} không hợp lệ".format(i, card))
        members = group.get("member_ids")
        if not isinstance(members, list):
            errors.append("concept_groups[{}].member_ids sai kiểu".format(i))
        else:
            unknown = [m for m in members if m not in element_ids]
            if unknown:
                errors.append("concept_groups[{}] có id lạ: {}".format(i, unknown))

    # Mọi element phải có mệnh đề, và mệnh đề rank 0 phải tồn tại.
    for element_id in element_ids:
        key = str(element_id)
        item = facts.get(key)
        if not isinstance(item, list) or not item:
            errors.append("facts['{}'] thiếu hoặc rỗng".format(key))
            continue
        ranks = [f.get("rank") for f in item if isinstance(f, dict)]
        if 0 not in ranks:
            errors.append("facts['{}'] không có mệnh đề rank 0".format(key))
        for f in item:
            if not isinstance(f, dict) or not f.get("text"):
                errors.append("facts['{}'] có mệnh đề thiếu text".format(key))
                break

    return errors


def normalize_decomposition(result: Dict[str, Any]) -> Dict[str, Any]:
    """Sắp xếp lại cho ổn định và loại bỏ kind lạ."""
    groups = []
    for group in result.get("concept_groups", []):
        groups.append({
            "name": str(group.get("name", "")).strip(),
            "member_ids": sorted(int(m) for m in group.get("member_ids", [])),
            "cardinality": group.get("cardinality", "vague"),
            "cultural": bool(group.get("cultural", False)),
        })

    facts: Dict[str, List[Dict[str, Any]]] = {}
    for key, items in result.get("facts", {}).items():
        cleaned = []
        for f in items:
            if not isinstance(f, dict) or not f.get("text"):
                continue
            kind = f.get("kind")
            cleaned.append({
                "rank": int(f.get("rank", 99)),
                "kind": kind if kind in VALID_KINDS else "attribute",
                "text": str(f["text"]).strip(),
            })
        cleaned.sort(key=lambda f: f["rank"])
        # Đánh lại rank liên tục 0..n-1 để bước 1b cắt theo ngưỡng cho chuẩn.
        for new_rank, f in enumerate(cleaned):
            f["rank"] = new_rank
        facts[str(key)] = cleaned

    return {"concept_groups": groups, "facts": facts}


def main() -> None:
    args = parse_args()
    io_utils.banner(STEP, "LLM gom nhóm khái niệm & phân rã mệnh đề")
    config.load_env(args.env_file)

    in_file = io_utils.resolve(args.in_file, args.out_root, "step0", "targets.jsonl")
    rows = io_utils.read_jsonl(in_file)
    rows = io_utils.apply_test_mode(rows, args, "mẫu")

    out_dir = io_utils.resolve(args.out_dir, args.out_root, "step1a")
    cache_dir = out_dir / "decompose"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        messages = prompts.build_step1a_messages(rows[0]["target_json"])
        print("\n[DRY RUN] dựng được {} messages cho id={}".format(len(messages), rows[0]["id"]))
        for m in messages:
            preview = m["content"][:400].replace("\n", "\n    ")
            print("\n  role={}\n    {}...".format(m["role"], preview))
        return

    todo = []
    n_cached = 0
    for row in rows:
        if not args.overwrite and (cache_dir / "{}.json".format(row["id"])).is_file():
            n_cached += 1
            continue
        todo.append(row)

    print("đã có cache: {} | cần gọi model: {}".format(n_cached, len(todo)))

    failures: List[Dict[str, Any]] = []

    if todo:
        client = llm.LLMClient()
        print("endpoint : {}".format(client.endpoint))
        print("model    : {}\n".format(client.model))

        def process(row: Dict[str, Any]) -> Optional[str]:
            target = row["target_json"]
            messages = prompts.build_step1a_messages(target)
            result = client.chat_json(
                messages, temperature=args.temperature, max_tokens=args.max_tokens,
            )
            errors = validate_decomposition(result, target)
            if errors:
                raise llm.LLMError("đầu ra không hợp lệ: {}".format("; ".join(errors[:4])))
            cleaned = normalize_decomposition(result)
            io_utils.write_json(cache_dir / "{}.json".format(row["id"]), cleaned)
            return row["id"]

        def on_error(row: Dict[str, Any], exc: Exception) -> None:
            failures.append({"id": row["id"], "error": str(exc)[:400]})

        llm.run_parallel(todo, process, workers=args.workers,
                         desc="phân rã", on_error=on_error)

    # Gộp toàn bộ cache thành một jsonl cho bước sau.
    merged: List[Dict[str, Any]] = []
    for row in rows:
        path = cache_dir / "{}.json".format(row["id"])
        if path.is_file():
            item = io_utils.read_json(path)
            merged.append({"id": row["id"], **item})

    n_written = io_utils.write_jsonl(out_dir / "decompose.jsonl", merged)
    if failures:
        io_utils.write_json(out_dir / "failures.json", failures)

    io_utils.summary(**{
        "mẫu xử lý": len(rows),
        "dùng lại cache": n_cached,
        "gọi model": len(todo),
        "thất bại": len(failures),
        "ghi ra decompose.jsonl": n_written,
        "thư mục đầu ra": str(out_dir),
    })


if __name__ == "__main__":
    main()
