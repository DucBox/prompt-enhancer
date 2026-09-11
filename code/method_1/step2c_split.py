#!/usr/bin/env python3
"""STEP 2c — Chia tập dữ liệu & lắp dòng huấn luyện cuối cùng.  [không gọi model]

Đầu vào : output/step2b_filtered/passed.jsonl
          output/step0_normalized/targets.jsonl   (để gắn nhãn Y)
          output/step1b_subjson/subjson.jsonl     (để gắn checklist chấm điểm)
Đầu ra  : output/step2c_split/
            train.jsonl  val.jsonl  test.jsonl
            split_report.json

CHIA THEO id ẢNH GỐC, KHÔNG CHIA THEO CẶP PROMPT. Một ảnh sinh ra 3 prompt;
nếu chia theo cặp thì cùng một ảnh sẽ nằm ở cả train lẫn test và điểm đánh giá bị thổi phồng.

Mỗi dòng đầu ra:
    {"id", "mode", "user_prompt", "sub_json", "target_json"}
trong đó target_json là NHÃN Y khi huấn luyện (đầy đủ, giống nhau cho cả 3 mức),
còn sub_json chỉ dùng để chấm điểm ở bước 5.

Ví dụ:
    python step2c_split.py --val_ids 700 --test_ids 700
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List

from common import io_utils

STEP = "STEP 2c"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chia train/val/test theo id ảnh gốc")
    p.add_argument("--in_file", default="output/step2b_filtered/passed.jsonl")
    p.add_argument("--targets_file", default="output/step0_normalized/targets.jsonl")
    p.add_argument("--subjson_file", default="output/step1b_subjson/subjson.jsonl")
    p.add_argument("--out_dir", default="output/step2c_split")
    p.add_argument("--val_ids", type=int, default=700, help="Số ẢNH cho tập val")
    p.add_argument("--test_ids", type=int, default=700, help="Số ẢNH cho tập test")
    p.add_argument("--val_ratio", type=float, default=None,
                   help="Dùng tỉ lệ thay cho số tuyệt đối (ghi đè --val_ids)")
    p.add_argument("--test_ratio", type=float, default=None)
    return io_utils.add_common_args(p).parse_args()


def main() -> None:
    args = parse_args()
    io_utils.banner(STEP, "Chia tập dữ liệu theo id ảnh gốc")

    rows = io_utils.read_jsonl(Path(args.in_file))
    targets = {r["id"]: r["target_json"] for r in io_utils.read_jsonl(Path(args.targets_file))}
    subjsons = {
        "{}__{}".format(r["id"], r["detail_level"]): r
        for r in io_utils.read_jsonl(Path(args.subjson_file))
    }

    all_ids = sorted({r["id"] for r in rows})
    all_ids = io_utils.apply_test_mode(all_ids, args, "ảnh")
    id_set = set(all_ids)
    rows = [r for r in rows if r["id"] in id_set]

    n_total = len(all_ids)
    n_val = int(round(args.val_ratio * n_total)) if args.val_ratio is not None else args.val_ids
    n_test = int(round(args.test_ratio * n_total)) if args.test_ratio is not None else args.test_ids

    # Với tập nhỏ (chế độ --test) thì co lại cho hợp lý thay vì báo lỗi.
    if n_val + n_test >= n_total:
        n_val = max(1, n_total // 10)
        n_test = max(1, n_total // 10)
        print("tập nhỏ — tự co val/test về {}/{} ảnh".format(n_val, n_test))

    shuffled = list(all_ids)
    random.Random(args.seed).shuffle(shuffled)
    val_ids = set(shuffled[:n_val])
    test_ids = set(shuffled[n_val:n_val + n_test])

    def split_of(row_id: str) -> str:
        if row_id in val_ids:
            return "val"
        if row_id in test_ids:
            return "test"
        return "train"

    buckets: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    n_missing_target = 0
    n_missing_sub = 0

    for row in rows:
        target = targets.get(row["id"])
        if target is None:
            n_missing_target += 1
            continue
        key = "{}__{}".format(row["id"], row["detail_level"])
        sub = subjsons.get(key)
        if sub is None:
            n_missing_sub += 1
            continue

        buckets[split_of(row["id"])].append({
            "id": row["id"],
            "mode": row["detail_level"],
            "user_prompt": row["user_prompt"],
            "persona": row.get("persona", {}),
            # sub_json: chỉ dùng để chấm điểm ở bước 5, KHÔNG phải nhãn huấn luyện
            "sub_json": {
                "scene": sub.get("scene", {}),
                "groups": sub.get("groups", []),
                "checklist": sub.get("checklist", []),
            },
            # target_json: NHÃN Y — đầy đủ, giống nhau cho cả 3 mức
            "target_json": target,
        })

    out_dir = Path(args.out_dir)
    report: Dict[str, Any] = {
        "n_images": n_total,
        "n_images_val": len(val_ids),
        "n_images_test": len(test_ids),
        "n_images_train": n_total - len(val_ids) - len(test_ids),
        "splits": {},
    }

    print("\n--- Kết quả chia ---")
    print("  {:<8}{:>10}{:>10}{:>10}{:>10}".format("tập", "số dòng", "short", "medium", "long"))
    for name in ("train", "val", "test"):
        items = buckets[name]
        io_utils.write_jsonl(out_dir / "{}.jsonl".format(name), items)
        counts = {lv: sum(1 for x in items if x["mode"] == lv) for lv in ("short", "medium", "long")}
        report["splits"][name] = {"n_rows": len(items), "per_mode": counts,
                                  "n_images": len({x["id"] for x in items})}
        print("  {:<8}{:>10}{:>10}{:>10}{:>10}".format(
            name, len(items), counts["short"], counts["medium"], counts["long"]))

    # Kiểm tra cứng: không id nào được xuất hiện ở hai tập khác nhau.
    seen: Dict[str, str] = {}
    leaks: List[str] = []
    for name in ("train", "val", "test"):
        for item in buckets[name]:
            other = seen.get(item["id"])
            if other is not None and other != name:
                leaks.append(item["id"])
            seen[item["id"]] = name
    report["id_leaks"] = sorted(set(leaks))
    if leaks:
        print("\n!! RÒ RỈ: {} id xuất hiện ở nhiều tập".format(len(set(leaks))))
    else:
        print("\n  kiểm tra rò rỉ id: OK — không id nào nằm ở hai tập")

    io_utils.write_json(out_dir / "split_report.json", report)

    io_utils.summary(**{
        "ảnh": n_total,
        "dòng train": len(buckets["train"]),
        "dòng val": len(buckets["val"]),
        "dòng test": len(buckets["test"]),
        "thiếu target_json": n_missing_target,
        "thiếu sub_json": n_missing_sub,
        "thư mục đầu ra": str(out_dir),
    })


if __name__ == "__main__":
    main()
