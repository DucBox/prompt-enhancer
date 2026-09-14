#!/usr/bin/env python3
"""STEP 2d — Chuẩn hoá nhãn Y về đúng schema Ideogram 4.  [không gọi model]

Đầu vào : <out_root>/step2c_split/{train,val,test}.jsonl
Đầu ra  : <out_root>/step2d_final/{train,val,test}.jsonl
            finalize_report.json

VIỆC DUY NHẤT CỦA BƯỚC NÀY: bóc `id` khỏi `target_json` và khoá lại thứ tự key.

`id` do step 0 tự gán cho từng element (= chỉ số trong mảng) để step 1a/1b tham
chiếu element khi phân rã mệnh đề. Nó KHÔNG có trong schema Ideogram 4 -- để
nguyên thì model học cách sinh ra một key mà bộ sinh ảnh chưa từng thấy lúc
train. `sub_json.groups[*].member_ids` vẫn dùng được sau khi bóc, vì `id` vốn
bằng đúng vị trí của element trong mảng.

Bước này KHÔNG loại dòng nào và KHÔNG đụng tới `user_prompt`. Chia tập đã xong ở
2c; ở đây chỉ sửa nhãn, nên train/val/test giữ nguyên thành phần.

Ví dụ:
    python step2d_finalize.py --out_root test_1
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

from common import io_utils, schema

STEP = "STEP 2d"

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bóc `id` khỏi nhãn Y, khoá thứ tự key")
    p.add_argument("--in_dir", default=None, help="Ghi đè thư mục đầu vào (mặc định: step2c_split)")
    p.add_argument("--out_dir", default=None, help="Ghi đè thư mục đầu ra của step này")
    return io_utils.add_common_args(p).parse_args()


def finalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["target_json"] = schema.strip_ids(row["target_json"])
    return out


def count_ids(target: Dict[str, Any]) -> int:
    elements = target.get("compositional_deconstruction", {}).get("elements", [])
    return sum(1 for el in elements if isinstance(el, dict) and "id" in el)


def main() -> None:
    args = parse_args()
    io_utils.banner(STEP, "Chuẩn hoá nhãn Y về đúng schema Ideogram 4")

    in_dir = io_utils.resolve(args.in_dir, args.out_root, "step2c")
    out_dir = io_utils.resolve(args.out_dir, args.out_root, "step2d")

    report: Dict[str, Any] = {"splits": {}}
    total_rows = 0
    total_ids = 0

    print("\n  {:<8}{:>10}{:>16}{:>14}".format("tập", "số dòng", "element bóc id", "còn sót"))
    for name in SPLITS:
        path = in_dir / "{}.jsonl".format(name)
        if not path.is_file():
            print("  {:<8}{:>10}".format(name, "(không có)"))
            continue

        rows = io_utils.read_jsonl(path)
        n_ids = sum(count_ids(r["target_json"]) for r in rows)
        final = [finalize_row(r) for r in rows]
        n_left = sum(count_ids(r["target_json"]) for r in final)

        io_utils.write_jsonl(out_dir / "{}.jsonl".format(name), final)
        report["splits"][name] = {
            "n_rows": len(final),
            "n_elements_stripped": n_ids,
            "n_ids_remaining": n_left,
        }
        total_rows += len(final)
        total_ids += n_ids
        print("  {:<8}{:>10}{:>16}{:>14}".format(name, len(final), n_ids, n_left))

    # Kiểm tra cứng: không được sót `id` nào, nếu không nhãn huấn luyện vẫn sai schema.
    leftovers = sum(s["n_ids_remaining"] for s in report["splits"].values())
    if leftovers:
        raise SystemExit("!! còn {} element giữ `id` — nhãn Y chưa sạch".format(leftovers))
    print("\n  kiểm tra: OK — không element nào còn `id`")

    report["n_rows"] = total_rows
    report["n_elements_stripped"] = total_ids
    io_utils.write_json(out_dir / "finalize_report.json", report)

    io_utils.summary(**{
        "dòng": total_rows,
        "element đã bóc id": total_ids,
        "thư mục đầu ra": str(out_dir),
    })


if __name__ == "__main__":
    main()
