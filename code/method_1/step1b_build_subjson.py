#!/usr/bin/env python3
"""STEP 1b — Lắp sub_json, nén theo 2 trục.  [không gọi model]

Đầu vào : <out_root>/step0_normalized/targets.jsonl
          <out_root>/step1a_decompose/decompose.jsonl
Đầu ra  : <out_root>/step1b_subjson/
            subjson.jsonl     3 dòng mỗi ảnh (short / medium / long)
            stats.json

HAI TRỤC NÉN (xem plan.pdf, bước 1b):
  * Bề rộng  — giữ bao nhiêu NHÓM khái niệm
  * Chiều sâu — mỗi element giữ bao nhiêu MỆNH ĐỀ (theo rank)

Chỉ cắt bề rộng là chưa đủ: mỗi desc dài trung bình 19 từ, nên một prompt "short"
giữ 2 element vẫn ra ~38 từ, đặc như JSON.

Toàn bộ quyết định cắt do CODE làm với seed cố định, nên:
  * tái tạo lại được dataset y hệt mà không cần gọi lại model
  * tập mệnh đề còn sống sót = checklist chấm điểm (gold của Tier 1)

Ví dụ:
    python step1b_build_subjson.py --out_root test_1
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import io_utils, schema

STEP = "STEP 1b"

# Cấu hình từng mức: (tỉ lệ nhóm giữ lại), (rank mệnh đề tối đa), (rank cho nhóm chính), (trần element)
LEVEL_CONFIG = {
    "short": {
        "group_ratio": (0.0, 0.0),   # chỉ nhóm số 1 + nhóm bắt buộc
        "max_rank": (1, 1),
        "main_group_max_rank": 1,
        "element_cap": 6,
        "length_hint": "8-20 từ",
        "keep_background": False,
        "style_fields": ("medium", "shot"),
    },
    "medium": {
        "group_ratio": (0.40, 0.60),
        "max_rank": (1, 2),
        "main_group_max_rank": 3,
        "element_cap": 10,
        "length_hint": "30-60 từ",
        "keep_background": "short",
        "style_fields": ("medium", "shot", "lighting"),
    },
    "long": {
        "group_ratio": (0.55, 0.95),
        "max_rank": (2, 3),
        "main_group_max_rank": 8,
        "element_cap": 24,
        "length_hint": "100-200 từ",
        "keep_background": True,
        "style_fields": ("medium", "shot", "lighting", "aesthetics"),
    },
}

LEVELS = ("short", "medium", "long")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nén target_json thành sub_json theo 3 mức")
    p.add_argument("--targets_file", default=None)
    p.add_argument("--decompose_file", default=None)
    p.add_argument("--out_dir", default=None, help="Ghi đè thư mục đầu ra của step này")
    p.add_argument("--levels", default="short,medium,long",
                   help="Các mức cần sinh, cách nhau bằng dấu phẩy")
    return io_utils.add_common_args(p).parse_args()


def first_clause(text: str, max_words: int = 12) -> str:
    """Cắt lấy mệnh đề đầu — cách nén văn bản nền/phong cách bằng code thuần."""
    s = str(text or "").strip()
    for sep in (";", ",", " — ", " - "):
        if sep in s:
            s = s.split(sep)[0].strip()
            break
    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words])
    return s.rstrip(".").strip()


def build_scene(target: Dict[str, Any], level: str) -> Dict[str, Any]:
    """Gợi ý bối cảnh & phong cách, nén theo mức."""
    cfg = LEVEL_CONFIG[level]
    style = target.get("style_description", {}) or {}
    comp = target.get("compositional_deconstruction", {}) or {}

    scene: Dict[str, Any] = {}
    fields = cfg["style_fields"]

    if "medium" in fields and style.get("medium"):
        scene["loai_anh"] = style["medium"]
    if "shot" in fields:
        shot = style.get("photo") or style.get("art_style")
        if shot:
            scene["goc_chup"] = first_clause(shot, 8)
    if "lighting" in fields and style.get("lighting"):
        scene["anh_sang"] = first_clause(style["lighting"], 8)
    if "aesthetics" in fields and style.get("aesthetics"):
        scene["khong_khi"] = style["aesthetics"]

    keep_bg = cfg["keep_background"]
    background = comp.get("background")
    if background:
        if keep_bg is True:
            scene["boi_canh"] = str(background).rstrip(".")
        elif keep_bg == "short":
            scene["boi_canh"] = first_clause(background, 14)

    return scene


def select_groups(
    groups: List[Dict[str, Any]],
    level: str,
    rng: random.Random,
    text_element_ids: set,
) -> Tuple[List[Dict[str, Any]], float]:
    """Trục BỀ RỘNG — chọn nhóm khái niệm nào được giữ.

    Luật cứng: nhóm mang yếu tố văn hoá Việt và nhóm chứa chữ-trong-ảnh luôn được giữ.
    """
    if not groups:
        return [], 1.0

    cfg = LEVEL_CONFIG[level]
    lo, hi = cfg["group_ratio"]
    ratio = rng.uniform(lo, hi) if hi > lo else lo

    n_total = len(groups)
    n_keep = max(1, int(round(ratio * n_total)))

    kept_indices = set()
    # Nhóm số 0 (quan trọng nhất) luôn giữ.
    kept_indices.add(0)
    for i, group in enumerate(groups):
        if group.get("cultural"):
            kept_indices.add(i)
        if text_element_ids and set(group.get("member_ids", [])) & text_element_ids:
            kept_indices.add(i)

    # Bổ sung theo thứ tự quan trọng cho đủ n_keep.
    for i in range(n_total):
        if len(kept_indices) >= n_keep:
            break
        kept_indices.add(i)

    selected = [groups[i] for i in sorted(kept_indices)]
    breadth_drop = 1.0 - (len(selected) / n_total)
    return selected, breadth_drop


def select_facts(
    facts: List[Dict[str, Any]],
    max_rank: int,
) -> List[Dict[str, Any]]:
    """Trục CHIỀU SÂU — mỗi element giữ mệnh đề có rank <= max_rank.

    Mệnh đề rank 0 (danh từ chủ thể) luôn được giữ, kể cả khi max_rank < 0.
    """
    kept = [f for f in facts if f.get("rank", 99) <= max_rank]
    if not kept and facts:
        kept = [facts[0]]
    return kept


def build_subjson(
    row_id: str,
    target: Dict[str, Any],
    decomposition: Dict[str, Any],
    level: str,
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    cfg = LEVEL_CONFIG[level]
    groups = decomposition.get("concept_groups", []) or []
    facts_map = decomposition.get("facts", {}) or {}

    text_element_ids = {
        el.get("id") for el in schema.elements_of(target) if el.get("type") == "text"
    }

    selected_groups, breadth_drop = select_groups(groups, level, rng, text_element_ids)
    if not selected_groups:
        return None

    lo_rank, hi_rank = cfg["max_rank"]
    element_cap = cfg["element_cap"]

    out_groups: List[Dict[str, Any]] = []
    checklist: List[str] = []
    n_elements_used = 0
    n_facts_kept = 0
    n_facts_total = sum(len(v) for v in facts_map.values())

    for group_index, group in enumerate(selected_groups):
        # Nhóm quan trọng nhất được giữ sâu hơn các nhóm phụ.
        is_main = group_index == 0
        max_rank = cfg["main_group_max_rank"] if is_main else rng.randint(lo_rank, hi_rank)

        member_ids = group.get("member_ids", []) or []
        group_facts: List[str] = []

        # KHÔNG dedup theo chữ trên toàn nhóm. Hai thành viên khác nhau (2 phụ nữ,
        # 2 chiếc bông tai...) rất hay cùng chung rank0 ("người phụ nữ", "bông tai
        # vàng"). Dedup toàn nhóm sẽ xoá mất chủ ngữ của thành viên thứ hai, chỉ còn
        # trơ thuộc tính không rõ thuộc về ai -- rồi model đọc đúng `cardinality` và
        # tự suy ra người/vật còn thiếu, nhưng checklist mất dấu vết nên bị chấm oan
        # là "thêm tin".
        for element_id in member_ids:
            if n_elements_used >= element_cap:
                break
            element_facts = facts_map.get(str(element_id), [])
            if not element_facts:
                continue
            n_elements_used += 1
            for fact in select_facts(element_facts, max_rank):
                group_facts.append(fact["text"])
                n_facts_kept += 1

        # Tên nhóm luôn được thêm vào checklist -- không chỉ khi group_facts rỗng.
        # Với nhóm văn hoá, đây là thứ quan trọng nhất cần verify (đúng tên gọi
        # Việt Nam), nên bắt buộc có mặt kể cả khi nhóm đã có sẵn nhiều facts khác.
        if not group_facts:
            group_facts = [group["name"]]
            n_facts_kept += 1
        elif group.get("cultural") and group["name"] not in group_facts:
            group_facts = [group["name"]] + group_facts

        entry = {
            "name": group["name"],
            "cardinality": group.get("cardinality", "vague"),
            "cultural": bool(group.get("cultural", False)),
            "member_ids": member_ids,
            "facts": group_facts,
        }
        out_groups.append(entry)
        checklist.extend(group_facts)

        # Ràng buộc số lượng chỉ phát khi đếm được chính xác và có từ 2 cá thể trở lên.
        card = group.get("cardinality", "vague")
        if card.startswith("exact:"):
            try:
                n = int(card.split(":", 1)[1])
            except ValueError:
                n = 0
            if n >= 2:
                checklist.append("số lượng: {} {}".format(n, group["name"]))

    depth_drop = 1.0 - (n_facts_kept / n_facts_total) if n_facts_total else 0.0
    scene = build_scene(target, level)

    # scene (góc chụp, ánh sáng, bối cảnh...) cũng là thông tin đưa cho model 2a viết prompt,
    # nên PHẢI có mặt trong checklist — thiếu nó thì mọi câu nhắc tới ánh sáng/bối cảnh
    # sẽ bị step 2b chấm oan là "thêm tin" dù thực ra 2a chỉ đang tả đúng phần được cấp.
    checklist.extend(v for v in scene.values() if v)

    return {
        "id": row_id,
        "detail_level": level,
        "length_hint": cfg["length_hint"],
        "breadth_drop_rate": round(breadth_drop, 3),
        "depth_drop_rate": round(depth_drop, 3),
        "scene": scene,
        "groups": out_groups,
        "checklist": checklist,
    }


def main() -> None:
    args = parse_args()
    io_utils.banner(STEP, "Lắp sub_json — nén theo 2 trục (bề rộng × chiều sâu)")

    targets_file = io_utils.resolve(args.targets_file, args.out_root, "step0", "targets.jsonl")
    decompose_file = io_utils.resolve(args.decompose_file, args.out_root, "step1a", "decompose.jsonl")
    out_dir = io_utils.resolve(args.out_dir, args.out_root, "step1b")

    targets = {r["id"]: r["target_json"] for r in io_utils.read_jsonl(targets_file)}
    decompositions = {r["id"]: r for r in io_utils.read_jsonl(decompose_file)}

    common_ids = sorted(set(targets) & set(decompositions))
    missing = len(targets) - len(common_ids)
    if not common_ids:
        raise SystemExit("Không có id nào xuất hiện ở cả hai file đầu vào.")
    if missing:
        print("bỏ qua {} ảnh chưa có kết quả bước 1a".format(missing))

    common_ids = io_utils.apply_test_mode(common_ids, args, "ảnh")

    levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]
    for level in levels:
        if level not in LEVEL_CONFIG:
            raise SystemExit("Mức không hợp lệ: {}".format(level))

    rows: List[Dict[str, Any]] = []
    per_level: Dict[str, List[Dict[str, Any]]] = {lv: [] for lv in levels}
    n_skipped = 0

    for row_id in common_ids:
        for level in levels:
            # Seed theo (id, level) để mỗi mẫu tái tạo được độc lập với thứ tự chạy.
            rng = random.Random("{}|{}|{}".format(args.seed, row_id, level))
            sub = build_subjson(row_id, targets[row_id], decompositions[row_id], level, rng)
            if sub is None:
                n_skipped += 1
                continue
            rows.append(sub)
            per_level[level].append(sub)

    n_written = io_utils.write_jsonl(out_dir / "subjson.jsonl", rows)

    stats: Dict[str, Any] = {"n_images": len(common_ids), "n_subjson": n_written, "levels": {}}
    print("\n--- Thống kê theo mức ---")
    print("  {:<9}{:>8}{:>14}{:>14}{:>14}".format(
        "mức", "số mẫu", "nhóm/mẫu", "mệnh đề/mẫu", "bỏ chiều sâu"))
    for level in levels:
        items = per_level[level]
        if not items:
            continue
        avg_groups = sum(len(s["groups"]) for s in items) / len(items)
        avg_facts = sum(len(s["checklist"]) for s in items) / len(items)
        avg_depth_drop = sum(s["depth_drop_rate"] for s in items) / len(items)
        stats["levels"][level] = {
            "n": len(items),
            "avg_groups": round(avg_groups, 2),
            "avg_checklist_facts": round(avg_facts, 2),
            "avg_depth_drop_rate": round(avg_depth_drop, 3),
        }
        print("  {:<9}{:>8}{:>14.2f}{:>14.2f}{:>13.1%}".format(
            level, len(items), avg_groups, avg_facts, avg_depth_drop))

    io_utils.write_json(out_dir / "stats.json", stats)

    io_utils.summary(**{
        "ảnh xử lý": len(common_ids),
        "sub_json sinh ra": n_written,
        "bỏ qua": n_skipped,
        "thư mục đầu ra": str(out_dir),
    })


if __name__ == "__main__":
    main()
