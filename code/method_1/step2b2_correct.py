#!/usr/bin/env python3
"""STEP 2b2 — Sửa lại prompt bị 2b loại (retry ĐÚNG 1 lần).  [GỌI MODEL]

Đầu vào : <out_root>/step2b_filtered/rejected.jsonl
Đầu ra  : <out_root>/step2b2_corrected/
            attempts/<id>__<level>.json
            corrected_passed.jsonl   sửa xong, chấm lại ĐẠT -- đi tiếp sang 2c
            rejected_final.jsonl     sửa rồi vẫn KHÔNG đạt -- thật sự bỏ
            report.json

BƯỚC NÀY LÀ TUỲ CHỌN -- run_all.sh chỉ chạy nó khi có --retry_rejected. Không bật
thì rejected.jsonl của 2b vẫn là danh sách cuối cùng bị loại, y như trước.

Với mỗi prompt bị loại, đưa đúng lý do bị loại (thiếu/thừa) cho model sửa lại — CHỈ
sửa đúng phần bị nêu lỗi, không viết lại từ đầu — rồi chấm lại bằng đúng judge của
bước 2b (dùng lại step2b_filter.decide(), kể cả ràng buộc chủ đề chính required_facts). Đạt
thì gộp vào tập đạt (đánh dấu corrected=true); vẫn fail thì MỚI thật sự loại — không
lặp lại lần hai.

Ví dụ:
    python step2b2_correct.py --out_root test_1 --workers 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import config, io_utils, llm, prompts
import step2b_filter as step2b

STEP = "STEP 2b2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sửa lại (retry 1 lần) prompt bị 2b loại")
    p.add_argument("--in_file", default=None, help="Mặc định: <out_root>/step2b_filtered/rejected.jsonl")
    p.add_argument("--out_dir", default=None, help="Ghi đè thư mục đầu ra của step này")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--max_missing", type=int, default=0,
                   help="Ngưỡng chấm lại -- giống hệt mặc định của step2b_filter")
    p.add_argument("--overwrite", action="store_true")
    return io_utils.add_common_args(p).parse_args()


def key_of(row: Dict[str, Any]) -> str:
    return "{}__{}".format(row["id"], row["detail_level"])


def main() -> None:
    args = parse_args()
    io_utils.banner(STEP, "Sửa lại prompt bị loại (retry 1 lần)")
    config.load_env(args.env_file)

    in_file = io_utils.resolve(args.in_file, args.out_root, "step2b", "rejected.jsonl")
    rows = io_utils.read_jsonl(in_file)
    rows = io_utils.apply_test_mode(rows, args, "prompt")

    out_dir = io_utils.resolve(args.out_dir, args.out_root, "step2b2")
    cache_dir = out_dir / "attempts"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        io_utils.write_jsonl(out_dir / "corrected_passed.jsonl", [])
        io_utils.write_jsonl(out_dir / "rejected_final.jsonl", [])
        io_utils.summary(**{"không có gì để sửa": 0})
        return

    todo = []
    n_cached = 0
    for row in rows:
        if not args.overwrite and (cache_dir / "{}.json".format(key_of(row))).is_file():
            n_cached += 1
            continue
        todo.append(row)

    print("đã có cache: {} | cần sửa: {}".format(n_cached, len(todo)))

    failures: List[Dict[str, Any]] = []

    if todo:
        writer = llm.LLMClient()
        judge = step2b.make_client()
        print("writer endpoint : {}".format(writer.endpoint))
        print("judge  endpoint : {}\n".format(judge.endpoint))

        def process(row: Dict[str, Any]) -> Optional[str]:
            reject = row.get("reject", {})
            # Chỉ bắt sửa mệnh đề CỐT LÕI; mệnh đề phụ đã được bỏ qua ở 2b thì cũng không
            # nhồi vào đây, tránh prompt bị độn chi tiết vụn cho đủ checklist.
            messages = prompts.build_step2b2_messages(
                row["checklist"],
                reject.get("missing_cot_loi", reject.get("missing", [])),
                reject.get("extra_them_vat_the", reject.get("extra", [])),
                row["user_prompt"],
            )
            raw = writer.chat(messages, temperature=args.temperature, max_tokens=args.max_tokens)
            corrected = step2b2_clean(raw)
            if not corrected:
                raise llm.LLMError("model trả về chuỗi rỗng khi sửa")

            judge_messages = prompts.build_step2b_messages(
                row["checklist"], corrected, row.get("menh_de_theo_nhom"))
            verdict = judge.chat_json(
                judge_messages, temperature=0.0, max_tokens=args.max_tokens,
            )
            if not isinstance(verdict, dict):
                raise llm.LLMError("judge trả về không phải object")

            decision = step2b.decide(verdict, args, step2b.required_facts_of(row))
            io_utils.write_json(cache_dir / "{}.json".format(key_of(row)), {
                "corrected_prompt": corrected,
                "verdict": verdict,
                "decision": decision,
            })
            return key_of(row)

        def on_error(row: Dict[str, Any], exc: Exception) -> None:
            failures.append({"key": key_of(row), "error": str(exc)[:400]})

        llm.run_parallel(todo, process, workers=args.workers,
                         desc="sửa lại", on_error=on_error)

    corrected_passed: List[Dict[str, Any]] = []
    rejected_final: List[Dict[str, Any]] = []

    for row in rows:
        path = cache_dir / "{}.json".format(key_of(row))
        if not path.is_file():
            continue
        attempt = io_utils.read_json(path)
        decision = attempt["decision"]
        if decision["passed"]:
            corrected_passed.append({
                **row,
                "user_prompt": attempt["corrected_prompt"],
                "n_words": len(attempt["corrected_prompt"].split()),
                "corrected": True,
                "original_prompt": row["user_prompt"],
            })
        else:
            rejected_final.append({
                **row,
                "attempted_prompt": attempt["corrected_prompt"],
                "reject_after_retry": decision,
            })

    io_utils.write_jsonl(out_dir / "corrected_passed.jsonl", corrected_passed)
    io_utils.write_jsonl(out_dir / "rejected_final.jsonl", rejected_final)
    if failures:
        io_utils.write_json(out_dir / "failures.json", failures)

    n_attempted = len(corrected_passed) + len(rejected_final)
    report = {
        "n_rejected_input": len(rows),
        "n_attempted": n_attempted,
        "n_recovered": len(corrected_passed),
        "n_still_rejected": len(rejected_final),
        "recovery_rate": round(len(corrected_passed) / n_attempted, 4) if n_attempted else 0.0,
    }
    io_utils.write_json(out_dir / "report.json", report)

    io_utils.summary(**{
        "prompt bị loại đưa vào": len(rows),
        "cứu được (đạt sau sửa)": len(corrected_passed),
        "vẫn loại sau khi sửa": len(rejected_final),
        "tỉ lệ cứu được": "{:.1%}".format(report["recovery_rate"]),
        "thất bại khi gọi": len(failures),
        "thư mục đầu ra": str(out_dir),
    })


def step2b2_clean(text: str) -> str:
    """Gỡ ngoặc kép bao ngoài và gộp xuống dòng -- model hay trả về dạng đó.
    Dùng lại đúng logic của step2a_verbalize.clean_prompt_text (tách riêng ở đây
    để step2b2 không phải import cả module step2a chỉ vì một hàm)."""
    s = " ".join(str(text).split())
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'", "“"):
        s = s[1:-1].strip()
    if s.startswith("“") and s.endswith("”"):
        s = s[1:-1].strip()
    return s


if __name__ == "__main__":
    main()
