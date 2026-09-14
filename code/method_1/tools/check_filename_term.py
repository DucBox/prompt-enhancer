#!/usr/bin/env python3
"""CÔNG CỤ THỐNG KÊ — cụm từ trong tên file có xuất hiện trong user prompt không.

Tên file là chủ đề ảnh (`am_tich_000755` -> "am tich"). Người dùng tìm ảnh ấm tích sẽ gõ
"ấm tích", nên prompt thiếu cụm này là mẫu train lệch. So khớp KHÔNG dấu, theo ranh giới từ.
Tên file hay có từ thừa ("hu_tieu_nam_vang"), nên thuật ngữ đòi hỏi là cụm đầu DÀI NHẤT của
tên file có trong JSON gốc (tối thiểu 2 từ), không phải cả tên.

Mỗi prompt được xếp vào một trong ba loại:
    co_trong_prompt         prompt có thuật ngữ đòi hỏi
    thieu__target_co        prompt thiếu nhưng JSON gốc có -> lỗi của pipeline (1b/2a/2b)
    thieu__target_khong_co  JSON gốc cũng không có -> tên file chỉ là chủ đề tìm kiếm,
                            không phải vật trong ảnh (vd rung_tre_tu_nhien nhưng ảnh là người)

Đầu vào : <out_root>/step2d_final/{train,val,test}.jsonl (không có thì dùng step2c_split)
Đầu ra  : <split_dir>/filename_term_report/
            report.json      thống kê theo tập × mức, theo chủ đề
            missing.jsonl    từng prompt thiếu cụm, kèm loại

    python tools/check_filename_term.py --out_root outputs/test_6
    python tools/check_filename_term.py --split_dir outputs/test_6/step2c_split
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import io_utils, subjson  # noqa: E402

SPLITS = ("train", "val", "test")
CATEGORIES = ("co_trong_prompt", "thieu__target_co", "thieu__target_khong_co")
# Tên thư mục chủ đề bị cắt ở 30 ký tự ("trung_tam_thanh_pho_ho_chi_min") -> từ cuối có thể
# chỉ là tiền tố.
TRUNCATED_SLUG_LEN = 30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Thống kê prompt có chứa cụm từ trong tên file")
    p.add_argument("--out_root", default="output")
    p.add_argument("--split_dir", default=None,
                   help="Thư mục chứa train/val/test.jsonl (mặc định: step2d_final, rồi step2c_split)")
    p.add_argument("--report_dir", default=None,
                   help="Nơi ghi report (mặc định: <split_dir>/filename_term_report)")
    p.add_argument("--top_topics", type=int, default=20, help="Số chủ đề thiếu nhiều nhất in ra")
    return p.parse_args()


def normalize(text: str) -> str:
    """Bỏ dấu, chữ thường, mọi ký tự không phải chữ/số thành một khoảng trắng."""
    ascii_text = subjson.strip_vn_diacritics(text).lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", ascii_text).split())


def topic_slug(row_id: str) -> str:
    return re.sub(r"_\d+$", "", row_id.split("__")[0])


def matched_prefix(row_id: str, text: str) -> Optional[str]:
    """Cụm đầu DÀI NHẤT của tên file xuất hiện trong text (không dấu, theo ranh giới từ).

    Tên file hay có từ thừa ("banh_mi_viet_nam", "hu_tieu_nam_vang", "ao_yem_truyen_thong")
    nên không đòi cả cụm. Tối thiểu 2 từ (hoặc cả tên nếu tên chỉ có 1 từ) -- một từ lẻ như
    "lan" không dấu thì trùng quá nhiều chữ khác.
    """
    words = subjson.filename_keyword_words(row_id.split("__")[0])
    if not words:
        return None
    truncated = len(topic_slug(row_id)) >= TRUNCATED_SLUG_LEN
    normalized = normalize(text)
    for k in range(len(words), min(2, len(words)) - 1, -1):
        body = " ".join(re.escape(w) for w in words[:k])
        tail = "" if (truncated and k == len(words)) else r"(?![a-z0-9])"
        if re.search(r"(?<![a-z0-9])" + body + tail, normalized):
            return " ".join(words[:k])
    return None


def classify(row: Dict[str, Any]) -> Tuple[str, str]:
    """-> (loại, thuật ngữ đòi hỏi). Thuật ngữ đòi hỏi = cụm đầu dài nhất có trong JSON gốc;
    JSON gốc không có cụm nào thì chỉ cần prompt có một cụm đầu bất kỳ (tối thiểu 2 từ)."""
    full_term = " ".join(subjson.filename_keyword_words(row["id"].split("__")[0]))
    in_prompt = matched_prefix(row["id"], row.get("user_prompt") or "")
    in_target = matched_prefix(row["id"], json.dumps(row.get("target_json") or {}, ensure_ascii=False))
    if in_target:
        ok = in_prompt is not None and len(in_prompt.split()) >= len(in_target.split())
        return ("co_trong_prompt" if ok else "thieu__target_co"), in_target
    return ("co_trong_prompt" if in_prompt else "thieu__target_khong_co"), full_term


def pick_split_dir(args: argparse.Namespace) -> Path:
    if args.split_dir:
        return Path(args.split_dir)
    for step in ("step2d", "step2c"):
        path = io_utils.step_dir(args.out_root, step)
        if any((path / "{}.jsonl".format(s)).is_file() for s in SPLITS):
            return path
    raise SystemExit("Không thấy train/val/test.jsonl trong step2d_final hay step2c_split của {}".format(
        args.out_root))


def rate(part: int, total: int) -> str:
    return "{:.1f}%".format(100.0 * part / total) if total else "-"


def main() -> None:
    args = parse_args()
    split_dir = pick_split_dir(args)
    report_dir = Path(args.report_dir) if args.report_dir else split_dir / "filename_term_report"
    io_utils.banner("CHECK", "Cụm từ trong tên file có xuất hiện trong user prompt không")
    print("đọc từ: {}".format(split_dir))

    by_split_mode: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    by_topic: Dict[str, Counter] = defaultdict(Counter)
    images_missing: Dict[str, set] = defaultdict(set)
    missing: List[Dict[str, Any]] = []
    n_rows = 0

    for split in SPLITS:
        path = split_dir / "{}.jsonl".format(split)
        if not path.is_file():
            continue
        for row in io_utils.iter_jsonl(path):
            n_rows += 1
            mode = row.get("mode") or row.get("detail_level") or "?"
            category, term = classify(row)
            by_split_mode[split][mode][category] += 1
            by_split_mode[split]["all"][category] += 1
            by_split_mode["all"][mode][category] += 1
            by_split_mode["all"]["all"][category] += 1
            by_topic[topic_slug(row["id"])][category] += 1
            if category != "co_trong_prompt":
                images_missing[category].add(row["id"])
                missing.append({
                    "id": row["id"], "split": split, "mode": mode, "category": category,
                    "term": term,
                    "language": (row.get("persona") or {}).get("ngôn ngữ"),
                    "user_prompt": row.get("user_prompt"),
                })

    if not n_rows:
        raise SystemExit("Không có dòng nào trong {}".format(split_dir))

    print("\n  {:<7}{:<8}{:>7}{:>10}{:>18}{:>24}".format(
        "tập", "mức", "số dòng", "có cụm", "thiếu (target có)", "thiếu (target không có)"))
    table: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for split in list(SPLITS) + ["all"]:
        if split not in by_split_mode:
            continue
        table[split] = {}
        for mode in ("short", "medium", "long", "all"):
            counts = by_split_mode[split].get(mode)
            if not counts:
                continue
            total = sum(counts.values())
            table[split][mode] = {"n": total, **{c: counts[c] for c in CATEGORIES},
                                  "ti_le_thieu": round(1 - counts["co_trong_prompt"] / total, 4)}
            print("  {:<7}{:<8}{:>7}{:>10}{:>18}{:>24}".format(
                split, mode, total, rate(counts["co_trong_prompt"], total),
                "{} ({})".format(counts["thieu__target_co"], rate(counts["thieu__target_co"], total)),
                "{} ({})".format(counts["thieu__target_khong_co"],
                                 rate(counts["thieu__target_khong_co"], total))))
        print()

    topics = []
    for slug, counts in by_topic.items():
        total = sum(counts.values())
        topics.append({"topic": slug, "n": total, **{c: counts[c] for c in CATEGORIES},
                       "ti_le_thieu": round(1 - counts["co_trong_prompt"] / total, 4)})
    topics.sort(key=lambda t: (-t["ti_le_thieu"], -t["n"]))

    print("  Chủ đề thiếu nhiều nhất (tỉ lệ thiếu | thiếu-target-có | thiếu-target-không-có | số dòng):")
    for t in [t for t in topics if t["ti_le_thieu"] > 0][:args.top_topics]:
        print("    {:<34} {:>6}  {:>4}  {:>4}  {:>5}".format(
            t["topic"], rate(t["n"] - t["co_trong_prompt"], t["n"]),
            t["thieu__target_co"], t["thieu__target_khong_co"], t["n"]))

    languages = Counter((m["category"], m["language"]) for m in missing)
    report = {
        "split_dir": str(split_dir),
        "n_rows": n_rows,
        "by_split_mode": table,
        "images_with_missing_prompt": {c: len(ids) for c, ids in images_missing.items()},
        "missing_by_language": {"{} | {}".format(c, lang): n for (c, lang), n in languages.most_common()},
        "topics": topics,
    }
    io_utils.write_json(report_dir / "report.json", report)
    io_utils.write_jsonl(report_dir / "missing.jsonl", missing)

    overall = table["all"]["all"]
    io_utils.summary(**{
        "tổng số prompt": n_rows,
        "có cụm tên file": "{} ({})".format(overall["co_trong_prompt"], rate(overall["co_trong_prompt"], n_rows)),
        "thiếu, target có (lỗi pipeline)": "{} ({})".format(
            overall["thieu__target_co"], rate(overall["thieu__target_co"], n_rows)),
        "thiếu, target cũng không có": "{} ({})".format(
            overall["thieu__target_khong_co"], rate(overall["thieu__target_khong_co"], n_rows)),
        "thiếu theo ngôn ngữ persona": report["missing_by_language"],
        "report": str(report_dir),
    })


if __name__ == "__main__":
    main()
