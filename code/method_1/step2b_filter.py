#!/usr/bin/env python3
"""STEP 2b — Lọc chất lượng prompt.  [GỌI MODEL]

Đầu vào : <out_root>/step2a_prompts/prompts.jsonl
Đầu ra  : <out_root>/step2b_filtered/
            judged/<id>__<level>.json
            passed.jsonl     cặp đạt, đi tiếp sang bước 2c
            rejected.jsonl   cặp bị loại, kèm lý do
            report.json

Kiểm tra prompt không THÊM và không THIẾU so với checklist (rút từ sub_json).
Bỏ bước này thì checklist hết là thước đo đáng tin và toàn bộ phần đánh giá
ở bước 5 sập theo.

KHÔNG lọc theo văn phong (dịch máy / liệt kê máy móc) — đó là vấn đề chất lượng
hành văn, xử lý ở bước 2a (system prompt + few-shot), không phải tiêu chí loại bỏ ở đây.

Nên dùng model KHÁC HỌ với model đã sinh prompt ở bước 2a để tránh thiên vị
(có thể trỏ riêng qua JUDGE_BASE_URL / JUDGE_MODEL trong .env).

Ví dụ:
    python step2b_filter.py --out_root test_1 --workers 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import config, io_utils, llm, prompts

STEP = "STEP 2b"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM lọc prompt sai lệch so với checklist")
    p.add_argument("--in_file", default=None, help="Mặc định: <out_root>/step2a_prompts/prompts.jsonl")
    p.add_argument("--out_dir", default=None, help="Ghi đè thư mục đầu ra của step này")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--max_missing", type=int, default=0,
                   help="Số mệnh đề được phép thiếu (mặc định 0)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--ignore_judge", action="store_true",
                   help="KHÔNG gọi judge, coi MỌI prompt là đạt (không chấm gì cả). "
                        "Dùng khi muốn tắt hẳn cổng lọc 2b -- vd debug, hoặc tin tưởng "
                        "thẳng đầu ra của 2a. Không tốn lượt gọi model nào.")
    return io_utils.add_common_args(p).parse_args()


def key_of(row: Dict[str, Any]) -> str:
    return "{}__{}".format(row["id"], row["detail_level"])


def make_client() -> llm.LLMClient:
    """Ưu tiên cấu hình judge riêng nếu .env có khai báo."""
    return llm.LLMClient(
        base_url=config.get("JUDGE_BASE_URL") or config.require("LLM_BASE_URL"),
        api_key=config.get("JUDGE_API_KEY") if config.get("JUDGE_BASE_URL") else config.get("LLM_API_KEY"),
        model=config.get("JUDGE_MODEL") or config.require("LLM_MODEL"),
    )


def decide(verdict: Dict[str, Any], args: argparse.Namespace,
          required_subject: Optional[str] = None) -> Dict[str, Any]:
    """Quyết định đạt/loại từ đầu ra của judge, theo ngưỡng do CLI đặt.

    Chỉ xét THIẾU/THÊM so với checklist. Văn phong (dịch máy, liệt kê máy móc...)
    không phải tiêu chí loại bỏ ở đây — đó là việc của bước 2a.

    `required_subject` (rút từ TÊN FILE, xem step1b_build_subjson.find_required_group_index)
    là chủ thể chính BẮT BUỘC tuyệt đối -- thiếu nó thì loại NGAY, không phụ thuộc
    --max_missing. Không có ràng buộc riêng này thì việc nới --max_missing > 0 để
    giảm tỉ lệ loại có thể vô tình cho lọt đúng trường hợp nguy hiểm nhất: prompt
    thiếu hẳn chủ thể chính (vd "hủ tiếu Nam Vang") nhưng vẫn đạt vì các mệnh đề
    thiếu khác được xem ngang hàng và nằm trong ngưỡng cho phép.
    """
    missing = verdict.get("missing") or []
    extra = verdict.get("extra") or []

    reasons: List[str] = []
    if required_subject and required_subject in missing:
        reasons.append("thiếu chủ thể chính bắt buộc: '{}'".format(required_subject))
    elif len(missing) > args.max_missing:
        reasons.append("thiếu {} mệnh đề".format(len(missing)))
    if extra:
        reasons.append("thêm {} thông tin".format(len(extra)))

    return {
        "passed": not reasons,
        "reasons": reasons,
        "missing": missing,
        "extra": extra,
    }


def main() -> None:
    args = parse_args()
    io_utils.banner(STEP, "LLM lọc chất lượng prompt")
    config.load_env(args.env_file)

    in_file = io_utils.resolve(args.in_file, args.out_root, "step2a", "prompts.jsonl")
    rows = io_utils.read_jsonl(in_file)
    rows = io_utils.apply_test_mode(rows, args, "prompt")

    out_dir = io_utils.resolve(args.out_dir, args.out_root, "step2b")
    cache_dir = out_dir / "judged"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        row = rows[0]
        messages = prompts.build_step2b_messages(row["checklist"], row["user_prompt"])
        print("\n[DRY RUN] {} messages cho {}".format(len(messages), key_of(row)))
        print("\n--- system ---\n{}".format(messages[0]["content"][:800]))
        print("\n--- user ---\n{}".format(messages[-1]["content"][:800]))
        return

    if args.ignore_judge:
        print("[ignore_judge] BỎ QUA judge -- mọi prompt được coi là đạt, không chấm gì cả.\n")
        for row in rows:
            path = cache_dir / "{}.json".format(key_of(row))
            if args.overwrite or not path.is_file():
                io_utils.write_json(path, {"missing": [], "extra": [], "ignored": True})

    todo = []
    n_cached = 0
    for row in rows:
        if not args.overwrite and (cache_dir / "{}.json".format(key_of(row))).is_file():
            n_cached += 1
            continue
        todo.append(row)

    print("đã có cache: {} | cần gọi model: {}".format(n_cached, len(todo)))

    failures: List[Dict[str, Any]] = []

    if todo and not args.ignore_judge:
        client = make_client()
        print("endpoint : {}".format(client.endpoint))
        print("model    : {}\n".format(client.model))

        def process(row: Dict[str, Any]) -> Optional[str]:
            messages = prompts.build_step2b_messages(row["checklist"], row["user_prompt"])
            verdict = client.chat_json(
                messages, temperature=args.temperature, max_tokens=args.max_tokens,
            )
            if not isinstance(verdict, dict):
                raise llm.LLMError("judge trả về không phải object")
            io_utils.write_json(cache_dir / "{}.json".format(key_of(row)), verdict)
            return key_of(row)

        def on_error(row: Dict[str, Any], exc: Exception) -> None:
            failures.append({"key": key_of(row), "error": str(exc)[:400]})

        llm.run_parallel(todo, process, workers=args.workers,
                         desc="chấm lọc", on_error=on_error)

    passed: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    reason_counts: Dict[str, int] = {}

    for row in rows:
        path = cache_dir / "{}.json".format(key_of(row))
        if not path.is_file():
            continue
        result = decide(io_utils.read_json(path), args, row.get("required_subject"))
        if result["passed"]:
            passed.append(row)
        else:
            rejected.append({**row, "reject": result})
            for reason in result["reasons"]:
                label = reason.split(" ")[0] + " " + reason.split(" ")[-1]
                reason_counts[label] = reason_counts.get(label, 0) + 1

    n_judged = len(passed) + len(rejected)
    io_utils.write_jsonl(out_dir / "passed.jsonl", passed)
    io_utils.write_jsonl(out_dir / "rejected.jsonl", rejected)
    if failures:
        io_utils.write_json(out_dir / "failures.json", failures)

    report = {
        "n_input": len(rows),
        "n_judged": n_judged,
        "n_passed": len(passed),
        "n_rejected": len(rejected),
        "pass_rate": round(len(passed) / n_judged, 4) if n_judged else 0.0,
        "reject_reasons": reason_counts,
        "per_level": {},
    }
    for level in ("short", "medium", "long"):
        total = sum(1 for r in rows if r["detail_level"] == level)
        ok = sum(1 for r in passed if r["detail_level"] == level)
        if total:
            report["per_level"][level] = {
                "n": total, "passed": ok, "pass_rate": round(ok / total, 4),
            }
    io_utils.write_json(out_dir / "report.json", report)

    print("\n--- Tỉ lệ đạt theo mức ---")
    for level, info in report["per_level"].items():
        print("  {:<8} {}/{}  = {:.1%}".format(level, info["passed"], info["n"], info["pass_rate"]))
    if reason_counts:
        print("\n--- Lý do loại ---")
        for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            print("  {:<24} {}".format(reason, count))

    io_utils.summary(**{
        "prompt đầu vào": len(rows),
        "đạt": len(passed),
        "bị loại": len(rejected),
        "tỉ lệ đạt": "{:.1%}".format(report["pass_rate"]),
        "thất bại khi gọi": len(failures),
        "thư mục đầu ra": str(out_dir),
    })


if __name__ == "__main__":
    main()
