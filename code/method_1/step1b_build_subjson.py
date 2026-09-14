#!/usr/bin/env python3
"""STEP 1b — LLM chọn lọc thông tin cho 3 mức short / medium / long.  [GỌI MODEL]

Đầu vào : <out_root>/step0_normalized/targets.jsonl
          <out_root>/step1a_decompose/decompose.jsonl
Đầu ra  : <out_root>/step1b_subjson/
            selection/<id>.json   lựa chọn đã hợp lệ của LLM (cache)
            subjson.jsonl         tối đa 3 dòng mỗi ảnh (short / medium / long)
            stats.json
            partial.json          ảnh chỉ giữ được một phần mức + lý do bỏ từng mức
            failures.json         ảnh không giữ được mức nào (kèm output cuối của LLM)

Mỗi ảnh gọi model MỘT lần cho cả 3 mức. LLM tự quyết giữ/bỏ theo định nghĩa mức trong
STEP1B_SYSTEM (độ phủ × độ sâu × loại thông tin); code không chọn nội dung, chỉ kiểm
tra lựa chọn (chép nguyên văn, lồng nhau, trần từ) và gọi lại kèm danh sách lỗi nếu sai.
Checklist = chủ đề chính của 1a (code tự chèn vào cả ba mức) + các mệnh đề đã chọn.

Hết lượt sửa mà vẫn còn lỗi thì KHÔNG bỏ cả ảnh: giữ các mức tự hợp lệ và lồng nhau với
nhau (subjson.evaluate_selection), chỉ bỏ mức hỏng. Ảnh giữ một phần vẫn được cache; chạy lại
với --retry_partial để gọi lại riêng các ảnh này.

Cache gắn dấu vân tay của (đầu vào + system prompt): 1a đổi kết quả hoặc sửa prompt 1b
thì ảnh đó tự được gọi lại.

Ví dụ:
    python step1b_build_subjson.py --out_root test_1 --dry_run
    python step1b_build_subjson.py --out_root test_1 --workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import config, io_utils, llm, prompts, subjson

STEP = "STEP 1b"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM chọn lọc thông tin cho short/medium/long")
    p.add_argument("--targets_file", default=None)
    p.add_argument("--decompose_file", default=None)
    p.add_argument("--out_dir", default=None, help="Ghi đè thư mục đầu ra của step này")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--max_fix_attempts", type=int, default=2,
                   help="Số lần gọi lại kèm danh sách lỗi khi lựa chọn không hợp lệ")
    p.add_argument("--overwrite", action="store_true", help="Bỏ qua cache, gọi lại toàn bộ")
    p.add_argument("--retry_partial", action="store_true",
                   help="Gọi lại các ảnh trong cache chỉ giữ được một phần mức")
    p.add_argument("--dry_run", action="store_true",
                   help="Chỉ dựng messages và in ra, KHÔNG gọi model")
    return io_utils.add_common_args(p).parse_args()


def fingerprint(sel_input: Dict[str, Any]) -> str:
    blob = json.dumps({"input": sel_input, "system": prompts.STEP1B_SYSTEM},
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


class SelectionError(llm.LLMError):
    """Hết lượt sửa mà không giữ được mức nào; mang theo output cuối của LLM để soi lỗi."""

    def __init__(self, message: str, last_output: Optional[str]) -> None:
        super().__init__(message)
        self.last_output = last_output


def cached_entry(path: Path, expected_fingerprint: str) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        item = io_utils.read_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(item, dict) or item.get("fingerprint") != expected_fingerprint:
        return None
    if not isinstance(item.get("selection"), dict):
        return None
    return item


def cached_selection(path: Path, expected_fingerprint: str) -> Optional[Dict[str, Any]]:
    item = cached_entry(path, expected_fingerprint)
    return item["selection"] if item else None


def entry_levels(item: Dict[str, Any]) -> List[str]:
    """Các mức dùng được của một cache; cache cũ (trước khi giữ một phần) luôn đủ 3 mức."""
    levels = item.get("levels")
    if not isinstance(levels, list):
        return list(subjson.LEVELS)
    return [level for level in subjson.LEVELS if level in levels]


def select_with_fixes(
    client: llm.LLMClient, sel_input: Dict[str, Any], args: argparse.Namespace,
) -> Dict[str, Any]:
    """Gọi LLM, sửa theo lỗi tối đa `max_fix_attempts` lần. Trả về
    {selection, n_attempts, levels, dropped}; hết lượt thì dùng lần thử giữ được nhiều mức nhất
    (hoà thì lấy lần sau). Không giữ được mức nào -> SelectionError."""
    messages = prompts.build_step1b_messages(sel_input)
    errors: List[str] = []
    best: Optional[Dict[str, Any]] = None
    raw: Optional[str] = None
    for attempt in range(args.max_fix_attempts + 1):
        raw = client.chat(messages, temperature=args.temperature,
                          max_tokens=args.max_tokens, json_mode=True)
        try:
            selection = llm.parse_json_response(raw)
            result = subjson.evaluate_selection(selection, sel_input)
            errors = result["errors"]
        except (ValueError, llm.LLMError) as exc:
            selection, result = None, None
            errors = ["không đọc được JSON: {}".format(str(exc)[:200])]
        if result is not None:
            candidate = {"selection": selection, "n_attempts": attempt + 1,
                         "levels": result["levels"], "dropped": result["dropped"]}
            if not errors:
                return candidate
            if result["levels"] and (best is None or len(result["levels"]) >= len(best["levels"])):
                best = candidate
        messages = list(messages) + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": prompts.build_step1b_fix_message(errors)},
        ]
    if best is not None:
        return best
    raise SelectionError("lựa chọn không hợp lệ sau {} lần: {}".format(
        args.max_fix_attempts + 1, "; ".join(errors[:4])), raw)


def main() -> None:
    args = parse_args()
    io_utils.banner(STEP, "LLM chọn lọc thông tin cho short / medium / long")
    config.load_env(args.env_file)

    targets_file = io_utils.resolve(args.targets_file, args.out_root, "step0", "targets.jsonl")
    decompose_file = io_utils.resolve(args.decompose_file, args.out_root, "step1a", "decompose.jsonl")
    out_dir = io_utils.resolve(args.out_dir, args.out_root, "step1b")
    cache_dir = out_dir / "selection"
    cache_dir.mkdir(parents=True, exist_ok=True)

    targets = {r["id"]: r["target_json"] for r in io_utils.read_jsonl(targets_file)}
    decompositions = {r["id"]: r for r in io_utils.read_jsonl(decompose_file)}

    common_ids = sorted(set(targets) & set(decompositions))
    if not common_ids:
        raise SystemExit("Không có id nào xuất hiện ở cả hai file đầu vào.")
    if len(targets) > len(common_ids):
        print("bỏ qua {} ảnh chưa có kết quả bước 1a".format(len(targets) - len(common_ids)))
    common_ids = io_utils.apply_test_mode(common_ids, args, "ảnh")

    inputs: Dict[str, Dict[str, Any]] = {}
    for row_id in common_ids:
        inputs[row_id] = subjson.build_selection_input(targets[row_id], decompositions[row_id])

    if args.dry_run:
        row_id = common_ids[0]
        messages = prompts.build_step1b_messages(inputs[row_id])
        print("\n[DRY RUN] {} messages cho id={}".format(len(messages), row_id))
        print("\n--- system ---\n{}".format(messages[0]["content"][:1200]))
        print("\n--- user cuối cùng ---\n{}".format(messages[-1]["content"]))
        return

    fingerprints = {row_id: fingerprint(inputs[row_id]) for row_id in common_ids}
    todo = []
    n_cached = 0
    for row_id in common_ids:
        path = cache_dir / "{}.json".format(row_id)
        entry = None if args.overwrite else cached_entry(path, fingerprints[row_id])
        if entry is not None and not (args.retry_partial and len(entry_levels(entry)) < len(subjson.LEVELS)):
            n_cached += 1
            continue
        todo.append(row_id)

    print("đã có cache: {} | cần gọi model: {}".format(n_cached, len(todo)))

    failures: List[Dict[str, Any]] = []

    if todo:
        client = llm.LLMClient()
        print("endpoint : {}".format(client.endpoint))
        print("model    : {}\n".format(client.model))

        def process(row_id: str) -> str:
            path = cache_dir / "{}.json".format(row_id)
            if args.overwrite and path.exists():
                path.unlink()  # gọi lại thất bại thì ảnh phải vắng mặt, không rơi về cache cũ
            result = select_with_fixes(client, inputs[row_id], args)
            old = cached_entry(path, fingerprints[row_id])
            if old is not None and len(entry_levels(old)) > len(result["levels"]):
                return row_id  # --retry_partial ra kết quả kém hơn -> giữ kết quả cũ
            io_utils.write_json(path, {
                "fingerprint": fingerprints[row_id],
                "n_attempts": result["n_attempts"],
                "levels": result["levels"],
                "dropped": result["dropped"],
                "input": inputs[row_id],
                "selection": result["selection"],
            })
            return row_id

        def on_error(row_id: str, exc: Exception) -> None:
            failures.append({"id": row_id, "error": str(exc)[:600],
                             "last_output": getattr(exc, "last_output", None)})

        llm.run_parallel(todo, process, workers=args.workers,
                         desc="chọn lọc", on_error=on_error)

    rows: List[Dict[str, Any]] = []
    attempts = Counter()
    partial: List[Dict[str, Any]] = []
    for row_id in common_ids:
        path = cache_dir / "{}.json".format(row_id)
        entry = cached_entry(path, fingerprints[row_id])
        if entry is None:
            continue
        attempts[entry.get("n_attempts", 1)] += 1
        levels = entry_levels(entry)
        if len(levels) < len(subjson.LEVELS):
            partial.append({"id": row_id, "kept": levels, "dropped": entry.get("dropped") or {}})
        for level in levels:
            rows.append(subjson.assemble_subjson(row_id, inputs[row_id], entry["selection"], level))

    n_written = io_utils.write_jsonl(out_dir / "subjson.jsonl", rows)
    # --retry_partial gọi lại thất bại vẫn còn kết quả cũ -> ảnh đó không tính là thất bại.
    has_rows = {r["id"] for r in rows}
    failures = [f for f in failures if f["id"] not in has_rows]
    # Ghi (hoặc xoá) cả hai file mỗi lần chạy để không sót lại kết quả của lần chạy cũ.
    for name, items in (("failures.json", failures), ("partial.json", partial)):
        if items:
            io_utils.write_json(out_dir / name, items)
        elif (out_dir / name).exists():
            (out_dir / name).unlink()

    stats: Dict[str, Any] = {
        "n_images": len(common_ids),
        "n_subjson": n_written,
        "n_partial": len(partial),
        "dropped_levels": dict(Counter(level for p in partial for level in p["dropped"])),
        "attempts": dict(sorted(attempts.items())),
        "levels": {},
    }
    print("\n--- Thống kê theo mức ---")
    print("  {:<8}{:>7}{:>11}{:>13}{:>11}   {}".format(
        "mức", "số mẫu", "nhóm/mẫu", "mệnh đề/mẫu", "từ/mẫu", "phân bố số mệnh đề"))
    for level in subjson.LEVELS:
        items = [r for r in rows if r["detail_level"] == level]
        if not items:
            continue
        n_facts = Counter(len(r["checklist"]) for r in items)
        avg_groups = sum(len(r["groups"]) for r in items) / len(items)
        avg_facts = sum(len(r["checklist"]) for r in items) / len(items)
        avg_words = sum(subjson.checklist_words(r["checklist"]) for r in items) / len(items)
        stats["levels"][level] = {
            "n": len(items),
            "avg_groups": round(avg_groups, 2),
            "avg_checklist_facts": round(avg_facts, 2),
            "avg_checklist_words": round(avg_words, 2),
            "checklist_facts_distribution": dict(sorted(n_facts.items())),
        }
        top = ", ".join("{}:{}".format(k, v) for k, v in sorted(n_facts.items()))
        print("  {:<8}{:>7}{:>11.2f}{:>13.2f}{:>11.1f}   {}".format(
            level, len(items), avg_groups, avg_facts, avg_words, top))
    print("  số lần gọi để ra lựa chọn hợp lệ: {}".format(stats["attempts"]))
    if partial:
        print("  giữ một phần: {} ảnh, mức bị bỏ: {}  (xem partial.json)".format(
            len(partial), stats["dropped_levels"]))

    io_utils.write_json(out_dir / "stats.json", stats)

    io_utils.summary(**{
        "ảnh xử lý": len(common_ids),
        "dùng lại cache": n_cached,
        "gọi model": len(todo),
        "giữ một phần mức": len(partial),
        "thất bại (không giữ được mức nào)": len(failures),
        "sub_json sinh ra": n_written,
        "thư mục đầu ra": str(out_dir),
    })


if __name__ == "__main__":
    main()
