#!/usr/bin/env python3
"""STEP 0 — Chuẩn hoá & kiểm định dữ liệu gốc.  [không gọi model]

Đầu vào : thư mục chứa các file .txt/.json, mỗi file là một JSON mô tả ảnh
Đầu ra  : <out_root>/step0_normalized/
            targets/<id>.json    target_json đã sạch, element có id
            targets.jsonl        {"id", "target_json"} — đầu vào cho step 1a
            audit_report.json    thống kê phân phối, dùng làm ngưỡng Richness

Ví dụ:
    python step0_normalize.py --in_dir DATA --out_root test_1 --test --test_samples 20
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

from common import io_utils, schema

STEP = "STEP 0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chuẩn hoá & audit target_json")
    p.add_argument("--in_dir", required=True, help="Thư mục chứa file .txt/.json gốc")
    p.add_argument("--out_dir", default=None, help="Ghi đè thư mục đầu ra của step này")
    p.add_argument("--strict", action="store_true",
                   help="Loại bỏ hẳn mẫu lỗi schema thay vì chỉ cảnh báo")
    return io_utils.add_common_args(p).parse_args()


def percentiles(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(q: float) -> int:
        return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]

    return {
        "min": ordered[0],
        "p25": at(0.25),
        "p50": at(0.50),
        "p75": at(0.75),
        "p90": at(0.90),
        "p99": at(0.99),
        "max": ordered[-1],
        "mean": round(statistics.mean(ordered), 2),
    }


def build_audit(rows: List[Dict[str, Any]], n_input: int, n_bad: int) -> Dict[str, Any]:
    per_file = [schema.stats(r["target_json"]) for r in rows]
    desc_words: List[int] = []
    for row in rows:
        for el in schema.elements_of(row["target_json"]):
            desc_words.append(len(str(el.get("desc", "")).split()))

    total_elements = sum(s["n_elements"] for s in per_file)
    total_text = sum(s["n_text_elements"] for s in per_file)

    return {
        "n_files_input": n_input,
        "n_files_ok": len(rows),
        "n_files_schema_error": n_bad,
        "total_elements": total_elements,
        "text_elements": total_text,
        "text_element_ratio": round(total_text / total_elements, 5) if total_elements else 0.0,
        "distributions": {
            "elements_per_image": percentiles([s["n_elements"] for s in per_file]),
            "words_per_desc": percentiles(desc_words),
            "words_background": percentiles([s["words_background"] for s in per_file]),
            "words_high_level": percentiles([s["words_high_level"] for s in per_file]),
        },
        "richness_reference": {
            "note": "Ngưỡng tham chiếu cho chỉ số Richness ở bước 5",
            "median_elements": percentiles([s["n_elements"] for s in per_file]).get("p50"),
            "median_words_per_desc": percentiles(desc_words).get("p50"),
        },
    }


def print_audit(audit: Dict[str, Any]) -> None:
    print("\n--- Báo cáo audit ---")
    print("  file hợp lệ        : {}/{}".format(audit["n_files_ok"], audit["n_files_input"]))
    print("  file lỗi schema    : {}".format(audit["n_files_schema_error"]))
    print("  tổng element       : {}".format(audit["total_elements"]))
    print("  element type=text  : {} ({:.1%})".format(
        audit["text_elements"], audit["text_element_ratio"]))
    print()
    print("  {:<22}{:>6}{:>6}{:>6}{:>6}{:>6}{:>12}".format(
        "phân phối", "p50", "p75", "p90", "p99", "max", "trung bình"))
    for name, dist in audit["distributions"].items():
        if not dist:
            continue
        print("  {:<22}{:>6}{:>6}{:>6}{:>6}{:>6}{:>12}".format(
            name, dist["p50"], dist["p75"], dist["p90"], dist["p99"], dist["max"], dist["mean"]))


def main() -> None:
    args = parse_args()
    io_utils.banner(STEP, "Chuẩn hoá & kiểm định dữ liệu gốc")

    in_dir = Path(args.in_dir)
    if not in_dir.is_dir():
        raise SystemExit("Không tìm thấy thư mục: {}".format(in_dir))

    files = sorted(p for p in in_dir.iterdir() if p.suffix in {".txt", ".json"})
    if not files:
        raise SystemExit("Không có file .txt/.json nào trong {}".format(in_dir))
    files = io_utils.apply_test_mode(files, args, "file")

    out_dir = io_utils.resolve(args.out_dir, args.out_root, "step0")
    targets_dir = out_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    n_parse_error = 0
    n_schema_error = 0
    error_log: List[Dict[str, Any]] = []

    for path in files:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            n_parse_error += 1
            error_log.append({"file": path.name, "kind": "parse", "detail": str(exc)})
            continue

        if not isinstance(raw, dict):
            n_parse_error += 1
            error_log.append({"file": path.name, "kind": "parse", "detail": "không phải JSON object"})
            continue

        target = schema.normalize(raw)
        errors = schema.validate(target)
        if errors:
            n_schema_error += 1
            error_log.append({"file": path.name, "kind": "schema", "detail": errors})
            if args.strict:
                continue

        io_utils.write_json(targets_dir / "{}.json".format(path.stem), target)
        rows.append({"id": path.stem, "target_json": target})

    if not rows:
        raise SystemExit("Không còn mẫu hợp lệ nào sau khi lọc.")

    n_written = io_utils.write_jsonl(out_dir / "targets.jsonl", rows)
    audit = build_audit(rows, len(files), n_schema_error)
    io_utils.write_json(out_dir / "audit_report.json", audit)
    if error_log:
        io_utils.write_json(out_dir / "errors.json", error_log[:500])

    print_audit(audit)
    io_utils.summary(**{
        "file đọc vào": len(files),
        "lỗi parse": n_parse_error,
        "lỗi schema": n_schema_error,
        "ghi ra targets.jsonl": n_written,
        "thư mục đầu ra": str(out_dir),
    })


if __name__ == "__main__":
    main()
