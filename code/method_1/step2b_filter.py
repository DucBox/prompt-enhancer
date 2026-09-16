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
import re
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
                   help="Số mệnh đề CỐT LÕI được phép thiếu (mặc định 0). Mệnh đề judge chấm "
                        "là phụ luôn được bỏ qua, không tính vào ngưỡng này.")
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


def required_facts_of(row: Dict[str, Any]) -> List[str]:
    """Mệnh đề chủ đề chính bắt buộc của một dòng; đọc được cả dòng cũ chỉ có required_subject."""
    if row.get("required_facts"):
        return list(row["required_facts"])
    return [row["required_subject"]] if row.get("required_subject") else []


def reason_label(reason: str) -> str:
    """Nhãn gom thống kê: "thiếu 3 mệnh đề" -> "thiếu mệnh đề", bỏ phần chi tiết sau dấu ':'."""
    return re.sub(r"\s*\d+\s+", " ", reason.split(":")[0]).strip()


def missing_items(verdict: Dict[str, Any]) -> List[Dict[str, str]]:
    """Chuẩn hoá "missing" của judge thành [{"menh_de", "muc_do"}].

    Judge trả object có nhãn `muc_do` (cot_loi | phu). Verdict cũ chỉ là danh sách chuỗi --
    coi hết là "cot_loi" để không vô tình nới lỏng dữ liệu đã chấm trước đây.
    """
    items: List[Dict[str, str]] = []
    for item in verdict.get("missing") or []:
        if isinstance(item, dict):
            text = str(item.get("menh_de", "")).strip()
            muc_do = str(item.get("muc_do", "cot_loi")).strip().lower()
        else:
            text, muc_do = str(item).strip(), "cot_loi"
        if text:
            items.append({"menh_de": text, "muc_do": "phu" if muc_do == "phu" else "cot_loi"})
    return items


def extra_items(verdict: Dict[str, Any]) -> List[Dict[str, str]]:
    """Chuẩn hoá "extra" thành [{"thong_tin", "muc_do"}]; verdict cũ (chuỗi) coi là them_vat_the."""
    items: List[Dict[str, str]] = []
    for item in verdict.get("extra") or []:
        if isinstance(item, dict):
            text = str(item.get("thong_tin", "")).strip()
            muc_do = str(item.get("muc_do", "them_vat_the")).strip().lower()
        else:
            text, muc_do = str(item).strip(), "them_vat_the"
        if text:
            items.append({"thong_tin": text,
                          "muc_do": "cam_nhan" if muc_do == "cam_nhan" else "them_vat_the"})
    return items


def decide(verdict: Dict[str, Any], args: argparse.Namespace,
          required_facts: Optional[List[str]] = None) -> Dict[str, Any]:
    """Quyết định đạt/loại từ đầu ra của judge.

    Chỉ xét THIẾU/THÊM so với checklist. Văn phong (dịch máy, liệt kê máy móc...)
    không phải tiêu chí loại bỏ ở đây — đó là việc của bước 2a.

    MODEL quyết định mệnh đề nào quan trọng (`muc_do`), code chỉ chặn phần cốt lõi:
    - thiếu mệnh đề CHỦ ĐỀ CHÍNH (`required_facts`, 1a rút từ high_level_description) -> loại
      ngay dù judge chấm nó là phụ;
    - thiếu mệnh đề "cot_loi" quá `--max_missing` -> loại;
    - thiếu mệnh đề "phu" (lấy nét sâu, ánh sáng ban ngày, nhỏ, ở góc dưới bên phải...) -> BỎ QUA,
      vì user thật không nói những thứ đó và model được train phải tự bổ sung;
    - THÊM vật thể / màu / số lượng mới ("them_vat_the") -> loại ngay: prompt đòi thứ không có
      trong ảnh sẽ dạy model bỏ qua yêu cầu của user;
    - THÊM kiểu cảm nhận, lời dẫn ("cam_nhan": "trông đẹp mắt quá", "tôi đang làm đồ án") ->
      BỎ QUA, bao nhiêu chỗ cũng được. Judge đã cân nhắc "có làm ảnh khác đi không" cho từng
      chỗ, nên code không đặt thêm trần đếm: 5 chỗ tiểu tiết vô hại vẫn lành hơn 1 chỗ thêm
      vật thể. User thật luôn nói thừa kiểu đó; ép sạch thì dữ liệu chỉ còn prompt khô cứng.
    """
    items = missing_items(verdict)
    extras = extra_items(verdict)
    extra = [i["thong_tin"] for i in extras]
    hard_extra = [i["thong_tin"] for i in extras if i["muc_do"] == "them_vat_the"]
    soft_extra = [i["thong_tin"] for i in extras if i["muc_do"] == "cam_nhan"]
    core = [i["menh_de"] for i in items if i["muc_do"] == "cot_loi"]
    minor = [i["menh_de"] for i in items if i["muc_do"] == "phu"]

    reasons: List[str] = []
    lost_theme = [f for f in required_facts or [] if any(f == i["menh_de"] for i in items)]
    if lost_theme:
        reasons.append("thiếu chủ đề chính: {}".format(lost_theme))
    elif len(core) > args.max_missing:
        reasons.append("thiếu {} mệnh đề cốt lõi".format(len(core)))
    if hard_extra:
        reasons.append("thêm {} thông tin".format(len(hard_extra)))

    return {
        "passed": not reasons,
        "reasons": reasons,
        "missing": [i["menh_de"] for i in items],
        "missing_cot_loi": core,
        "missing_phu": minor,
        "extra": extra,
        "extra_them_vat_the": hard_extra,
        "extra_cam_nhan": soft_extra,
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
        messages = prompts.build_step2b_messages(
            row["checklist"], row["user_prompt"], row.get("menh_de_theo_nhom"))
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
            messages = prompts.build_step2b_messages(
                row["checklist"], row["user_prompt"], row.get("menh_de_theo_nhom"))
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
    minor_counts: Dict[str, int] = {}
    soft_extra_counts: Dict[str, int] = {}
    n_passed_with_minor = 0
    n_passed_with_soft_extra = 0

    for row in rows:
        path = cache_dir / "{}.json".format(key_of(row))
        if not path.is_file():
            continue
        result = decide(io_utils.read_json(path), args, required_facts_of(row))
        for fact in result["missing_phu"]:
            minor_counts[fact] = minor_counts.get(fact, 0) + 1
        for text in result["extra_cam_nhan"]:
            soft_extra_counts[text] = soft_extra_counts.get(text, 0) + 1
        if result["passed"]:
            if result["missing_phu"]:
                n_passed_with_minor += 1
            if result["extra_cam_nhan"]:
                n_passed_with_soft_extra += 1
            passed.append(row)
        else:
            rejected.append({**row, "reject": result})
            for reason in result["reasons"]:
                label = reason_label(reason)
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
        # Mệnh đề judge chấm là phụ -> bỏ qua. Soi bảng này để biết judge có nới tay quá không.
        "n_passed_with_missing_phu": n_passed_with_minor,
        "missing_phu_top": dict(sorted(minor_counts.items(), key=lambda kv: -kv[1])[:30]),
        "n_passed_with_extra_cam_nhan": n_passed_with_soft_extra,
        "extra_cam_nhan_top": dict(sorted(soft_extra_counts.items(), key=lambda kv: -kv[1])[:30]),
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
    if minor_counts:
        print("\n--- Bỏ qua (judge chấm là phụ), 10 mệnh đề hay thiếu nhất ---")
        for fact, count in sorted(minor_counts.items(), key=lambda kv: -kv[1])[:10]:
            print("  {:<40} {}".format(fact[:40], count))
    if soft_extra_counts:
        print("\n--- Bỏ qua (judge chấm là cảm nhận), 10 chỗ thêm hay gặp nhất ---")
        for text, count in sorted(soft_extra_counts.items(), key=lambda kv: -kv[1])[:10]:
            print("  {:<40} {}".format(text[:40], count))
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
