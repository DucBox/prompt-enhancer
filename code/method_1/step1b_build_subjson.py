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
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import io_utils, schema

STEP = "STEP 1b"

# Cấu hình từng mức: (tỉ lệ nhóm giữ lại), (rank mệnh đề tối đa), (rank cho nhóm chính), (trần element)
#
# group_element_cap — trần số element của TỪNG nhóm (main lẫn phụ). Không có trần này,
#   một nhóm nhiều thành viên (vd "ba người trên thuyền", hay một nhóm văn hoá phụ gộp
#   4 element) sẽ gộp hết vào short, ra hàng chục mệnh đề cho một mức lẽ ra chỉ 8-20 từ.
# max_cultural_groups      — trần số nhóm văn hoá được ÉP giữ. Luật "nhóm văn hoá luôn
#   giữ" không có trần thì một ảnh có 4 nhóm cùng gắn cultural=True (ví dụ 4 món ăn/đồ
#   vật) sẽ phá vỡ hoàn toàn ngân sách short. Vượt trần thì ưu tiên giữ nhóm đứng trước
#   (đã xếp theo salience từ bước 1a), phần còn lại vẫn có cơ hội lọt vào ở medium/long.
LEVEL_CONFIG = {
    "short": {
        "group_ratio": (0.0, 0.0),   # chỉ nhóm số 1 + nhóm bắt buộc
        "max_rank": (1, 1),
        "main_group_max_rank": 1,
        "group_element_cap": 2,
        "max_cultural_groups": 2,
        "element_cap": 6,
        "max_checklist_words": 16,
        "length_hint": "8-20 từ",
        "keep_background": False,
        "style_fields": ("medium", "shot"),
    },
    "medium": {
        "group_ratio": (0.40, 0.60),
        "max_rank": (1, 2),
        "main_group_max_rank": 3,
        "group_element_cap": 4,
        "max_cultural_groups": 3,
        "element_cap": 10,
        "max_checklist_words": 50,
        "length_hint": "30-60 từ",
        "keep_background": "short",
        "style_fields": ("medium", "shot", "lighting"),
    },
    "long": {
        "group_ratio": (0.55, 0.95),
        "max_rank": (2, 3),
        "main_group_max_rank": 8,
        "group_element_cap": None,        # không giới hạn riêng, dùng element_cap chung
        "max_cultural_groups": None,      # không giới hạn
        "element_cap": 24,
        "max_checklist_words": None,      # không giới hạn -- xem callout "đuôi 95%" ở plan.pdf
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



# Cắt cứng theo số từ hay để lại một cụm CỤT giữa chừng (vd "...on the", "...post"
# thay vì "...post fence") -- bug thật (server test_6): checklist đòi model nhắc
# đúng nguyên văn một cụm không trọn nghĩa như vậy, không ai viết prompt lại chèn
# một câu bị cắt cụt. Sau khi cắt theo max_words, lùi lại bỏ các từ chức năng
# (giới từ/mạo từ) còn treo lơ lửng ở cuối, để cụm giữ lại luôn kết thúc trọn nghĩa.
_DANGLING_TAIL_WORDS = {
    "a", "an", "the", "on", "in", "at", "of", "with", "by", "from", "to",
    "behind", "above", "below", "under", "near", "for", "and", "or", "but",
    "into", "onto", "over", "through", "across", "toward", "towards",
}


def first_clause(text: str, max_words: int = 12) -> str:
    """Cắt lấy mệnh đề đầu — cách nén văn bản nền/phong cách bằng code thuần."""
    s = str(text or "").strip()
    for sep in (";", ",", " — ", " - "):
        if sep in s:
            s = s.split(sep)[0].strip()
            break
    words = s.split()
    if len(words) > max_words:
        words = words[:max_words]
        while words and words[-1].lower().strip(".,") in _DANGLING_TAIL_WORDS:
            words.pop()
        s = " ".join(words)
    return s.rstrip(".").strip()



# "photograph" chiếm 99.4% mẫu (994/1000) — mặc định gần như tuyệt đối, không mang
# thông tin phân biệt. Chỉ đáng đưa vào checklist khi medium KHÁC giá trị mặc định này
# (illustration, 3d_render, painting...) — lúc đó mới là một lựa chọn thật sự đáng nói.
DEFAULT_MEDIUM = "photograph"

# Góc máy "mặc định/tầm thường" — không ai đặt hàng ảnh lại nói ra những từ này.
# Chỉ giữ lại goc_chup khi câu tả góc máy chứa một từ khoá THỰC SỰ đáng chú ý
# (góc cao/thấp, cận cảnh, toàn cảnh, trên không...); nếu không có từ khoá nào trong
# số này thì bỏ hẳn goc_chup — đây cũng chính là loại "ngôn ngữ kỹ thuật khung hình"
# mà STEP2A_SYSTEM đang cấm model dùng, nên không thể bắt checklist đòi hỏi nó.
NOTABLE_ANGLE_KEYWORDS = (
    "high-angle", "high angle", "low-angle", "low angle",
    "aerial", "bird's-eye", "bird's eye", "overhead", "top-down",
    "drone", "from above", "from below", "worm's-eye", "worm's eye",
    "close-up", "closeup", "extreme close-up", "macro",
    "wide shot", "full shot", "full-body shot", "full body shot", "dutch angle",
    "wide-angle", "wide angle", "ultra-wide", "ultrawide", "fisheye", "telephoto",
)

# Nguồn dữ liệu không nhất quán: có ảnh viết "eye-level, wide shot" (2 mệnh đề tách
# riêng bằng dấu phẩy), có ảnh viết "eye-level wide shot" (dính liền, không dấu phẩy).
# Ở trường hợp dính liền, nếu giữ nguyên cả cụm thì "eye-level" (mặc định, bị cấm)
# vẫn lọt vào checklist cùng từ khoá đáng chú ý -- phải cắt bỏ phần mặc định này ra
# khỏi cụm trước khi đưa vào checklist.
_BORING_ANGLE_WORDS = ("eye-level", "eye level")


def notable_angle_clause(shot: str) -> Optional[str]:
    """Trả về mệnh đề chứa từ khoá góc máy đáng chú ý, hoặc None nếu góc máy mặc định."""
    s = str(shot or "")
    for clause in s.split(","):
        clause_low = clause.lower()
        if any(kw in clause_low for kw in NOTABLE_ANGLE_KEYWORDS):
            cleaned = clause.strip().rstrip(".")
            for boring in _BORING_ANGLE_WORDS:
                cleaned = re.sub(re.escape(boring), "", cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.strip(" ,-")
            if cleaned:
                return cleaned
    return None


def build_scene(target: Dict[str, Any], level: str) -> Dict[str, Any]:
    """Gợi ý bối cảnh & phong cách, nén theo mức."""
    cfg = LEVEL_CONFIG[level]
    style = target.get("style_description", {}) or {}
    comp = target.get("compositional_deconstruction", {}) or {}

    scene: Dict[str, Any] = {}
    fields = cfg["style_fields"]

    if "medium" in fields and style.get("medium"):
        medium = style["medium"]
        if str(medium).strip().lower() != DEFAULT_MEDIUM:
            scene["loai_anh"] = medium
    if "shot" in fields:
        shot = style.get("photo") or style.get("art_style")
        if shot:
            clause = notable_angle_clause(shot)
            if clause:
                scene["goc_chup"] = first_clause(clause, 8)
    if "lighting" in fields and style.get("lighting"):
        scene["anh_sang"] = first_clause(style["lighting"], 8)
    if "aesthetics" in fields and style.get("aesthetics"):
        # aesthetics là một DANH SÁCH TAG mood tiếng Anh rời rạc (vd "elegant,
        # historical, serene"), không phải câu mô tả -- không ai viết prompt lại
        # liệt kê 3 tính từ cách nhau bằng dấu phẩy kiểu gắn nhãn mood-board.
        # Chỉ giữ tag đầu (nổi bật nhất theo salience) làm GỢI Ý cho 2a viết,
        # và KHÔNG bắt buộc trong checklist chấm điểm (xem SCENE_HINT_ONLY_KEYS).
        scene["khong_khi"] = first_clause(style["aesthetics"], 3)

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
) -> Tuple[List[Dict[str, Any]], float, set]:
    """Trục BỀ RỘNG — chọn nhóm khái niệm nào được giữ.

    Luật cứng: nhóm mang yếu tố văn hoá Việt và nhóm chứa chữ-trong-ảnh luôn được giữ.

    Trả thêm `text_forced_ids`: id() của các nhóm được giữ CHỈ VÌ chứa chữ-trong-ảnh
    (không phải nhóm chính, không phải nhóm văn hoá, không nằm trong ngân sách bề
    rộng bình thường). 1a hay gom một nhóm "cảnh nền" lớn (chục element không liên
    quan) mà tình cờ chứa 1 element chữ — nếu giữ nguyên cả nhóm thì short/medium sẽ
    phình to dù chỉ cần đúng element chữ đó. build_subjson dùng tập này để CHỈ giữ
    phần tử chữ, bỏ qua phần còn lại của nhóm catch-all.
    """
    if not groups:
        return [], 1.0, set()

    cfg = LEVEL_CONFIG[level]
    lo, hi = cfg["group_ratio"]
    ratio = rng.uniform(lo, hi) if hi > lo else lo

    n_total = len(groups)
    n_keep = max(1, int(round(ratio * n_total)))

    # Ngân sách bề rộng "bình thường": nhóm chính + nhóm văn hoá (có trần) + lấp đầy.
    # Vượt trần thì ưu tiên nhóm đứng trước (đã xếp theo salience) -- phần còn lại
    # vẫn có cơ hội lọt vào ở mức nén nhẹ hơn (medium/long không có trần này).
    max_cultural = cfg.get("max_cultural_groups")
    base_kept = {0}
    n_cultural_kept = 0
    for i, group in enumerate(groups):
        if i == 0 or not group.get("cultural"):
            continue  # nhóm chính đã chắc chắn được giữ, không tính vào trần văn hoá
        if max_cultural is not None and n_cultural_kept >= max_cultural:
            continue
        base_kept.add(i)
        n_cultural_kept += 1
    for i in range(n_total):
        if len(base_kept) >= n_keep:
            break
        base_kept.add(i)

    kept_indices = set(base_kept)
    text_forced_ids: set = set()
    if text_element_ids:
        for i, group in enumerate(groups):
            if i in kept_indices:
                continue
            if set(group.get("member_ids", [])) & text_element_ids:
                kept_indices.add(i)
                text_forced_ids.add(id(group))

    selected = [groups[i] for i in sorted(kept_indices)]
    breadth_drop = 1.0 - (len(selected) / n_total)
    return selected, breadth_drop, text_forced_ids


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


def trim_checklist_to_word_budget(
    out_groups: List[Dict[str, Any]],
    so_luong_lines: List[Tuple[int, str]],
    scene: Dict[str, Any],
    max_words: Optional[int],
) -> List[str]:
    """Trần TỔNG SỐ TỪ của checklist -- không có trần này, đuôi phân phối vẫn có thể
    đòi hỏi checklist chứa nhiều từ hơn hẳn length_hint cho phép (bug thật: server
    test_6, p90 short cần ~30 từ cho ngân sách chỉ 8-20 từ -- mâu thuẫn trực tiếp với
    luật "PHẢI ĐẦY ĐỦ" ở STEP2A_SYSTEM, vì checklist đưa ra vốn đã không thể nói hết
    trong khoảng từ cho phép).

    Thứ tự ưu tiên GIỮ (drop theo chiều ngược lại, thấp ưu tiên nhất trước):
      1. mệnh đề của các nhóm, theo đúng thứ tự salience -- nhóm cuối bị đụng tới trước
      2. dòng "số lượng"
      3. scene (ánh sáng, góc chụp...)
    Không bao giờ trim một nhóm xuống dưới 1 mệnh đề -- nhóm đó vẫn được đưa cho 2a
    viết (qua "name"), nên checklist phải còn ít nhất một mệnh đề tương ứng để 2b
    không chấm oan phần nội dung mà 2a hoàn toàn có quyền nhắc tới.

    Sửa TRỰC TIẾP `out_groups[i]["facts"]` và `scene` (in-place) để những gì đưa cho
    2a viết (qua sub_json["groups"]/["scene"]) luôn khớp đúng với checklist chấm điểm
    ở 2b -- tách rời hai thứ này ra là lặp lại đúng bug đã sửa trước đó (thêm scene
    vào spec nhưng quên thêm vào checklist).
    """
    def wc(s: str) -> int:
        return len(s.split())

    if max_words is None:
        checklist = [f for g in out_groups for f in g["facts"]]
        checklist += [line for _, line in so_luong_lines]
        checklist += [v for k, v in scene.items() if v and k != "khong_khi"]
        return checklist

    scene_keys = [k for k, v in scene.items() if v and k != "khong_khi"]
    total = (
        sum(wc(f) for g in out_groups for f in g["facts"])
        + sum(wc(line) for _, line in so_luong_lines)
        + sum(wc(scene[k]) for k in scene_keys)
    )

    while total > max_words and scene_keys:
        k = scene_keys.pop()
        total -= wc(scene[k])
        del scene[k]

    while total > max_words and so_luong_lines:
        _, line = so_luong_lines.pop()
        total -= wc(line)

    gi = len(out_groups) - 1
    while total > max_words and gi >= 0:
        facts = out_groups[gi]["facts"]
        if len(facts) > 1:
            total -= wc(facts.pop())
        else:
            gi -= 1

    checklist = [f for g in out_groups for f in g["facts"]]
    checklist += [line for _, line in so_luong_lines]
    checklist += [scene[k] for k in scene_keys]
    return checklist


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

    selected_groups, breadth_drop, text_forced_ids = select_groups(
        groups, level, rng, text_element_ids,
    )
    if not selected_groups:
        return None

    lo_rank, hi_rank = cfg["max_rank"]
    element_cap = cfg["element_cap"]

    out_groups: List[Dict[str, Any]] = []
    so_luong_lines: List[Tuple[int, str]] = []
    n_elements_used = 0
    n_facts_kept = 0
    n_facts_total = sum(len(v) for v in facts_map.values())

    for group_index, group in enumerate(selected_groups):
        # Nhóm quan trọng nhất được giữ sâu hơn các nhóm phụ.
        is_main = group_index == 0
        max_rank = cfg["main_group_max_rank"] if is_main else rng.randint(lo_rank, hi_rank)

        member_ids = group.get("member_ids", []) or []

        # Nhóm này được giữ CHỈ VÌ chứa 1 element chữ (không phải nhóm chính/văn hoá,
        # không nằm trong ngân sách bề rộng bình thường) -- thường là một nhóm
        # "cảnh nền" gộp chục element không liên quan. Chỉ giữ đúng (các) element
        # chữ, bỏ qua phần còn lại, để không kéo cả đống nội dung phụ vào short/medium.
        if id(group) in text_forced_ids:
            member_ids = [m for m in member_ids if m in text_element_ids]

        # Trần số element cho TỪNG nhóm (main lẫn phụ). Không có trần này, một nhóm
        # nhiều thành viên (vd "ba người trên thuyền", hay một nhóm văn hoá phụ gộp
        # 4 element) sẽ gộp hết vào short, ra hàng chục mệnh đề cho một mức lẽ ra chỉ
        # 8-20 từ. Nhóm chính được giữ nhiều hơn 1 chút vì nó là chủ thể quan trọng nhất.
        group_cap = cfg.get("group_element_cap")
        if group_cap is not None:
            cap = group_cap + 1 if is_main else group_cap
            member_ids = member_ids[:cap]

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

        # Ràng buộc số lượng chỉ phát khi đếm được chính xác và có từ 2 cá thể trở lên.
        # BỎ QUA khi tên nhóm đã là danh từ tập hợp/cặp (đôi, cặp, bộ...) -- lúc đó
        # "n" đang đếm số MEMBER nguyên tử (vd 2 chiếc đũa), còn tên nhóm đã tự mang
        # nghĩa "một cặp/bộ" rồi. Ghép thẳng "{n} {tên}" sẽ ra một khẳng định số lượng
        # khác hẳn và thường SAI (vd "đôi đũa" + n=2 -> "2 đôi đũa" = 4 chiếc, trong khi
        # ảnh chỉ có đúng 1 đôi/2 chiếc). Bug thật thấy ở test_6: cau_vang_000914
        # ("đôi bàn tay đá khổng lồ" -- chỉ có 1 đôi/2 bàn tay -- bị ghi thành "2 đôi").
        card = group.get("cardinality", "vague")
        name_lower = group["name"].strip().lower()
        is_collective_name = any(
            name_lower.startswith(prefix)
            for prefix in ("đôi ", "cặp ", "bộ ", "cụm ", "chuỗi ", "xâu ", "dãy ")
        )
        if card.startswith("exact:") and not is_collective_name:
            try:
                n = int(card.split(":", 1)[1])
            except ValueError:
                n = 0
            if n >= 2:
                so_luong_lines.append((group_index, "số lượng: {} {}".format(n, group["name"])))

    depth_drop = 1.0 - (n_facts_kept / n_facts_total) if n_facts_total else 0.0
    scene = build_scene(target, level)

    # scene (góc chụp, ánh sáng, bối cảnh...) cũng là thông tin đưa cho model 2a viết prompt,
    # nên PHẢI có mặt trong checklist — thiếu nó thì mọi câu nhắc tới ánh sáng/bối cảnh
    # sẽ bị step 2b chấm oan là "thêm tin" dù thực ra 2a chỉ đang tả đúng phần được cấp.
    #
    # NGOẠI LỆ: "khong_khi" (aesthetics/mood) chỉ là GỢI Ý văn phong cho 2a, không phải
    # nội dung cụ thể để 2b chấm điểm -- gây 71% prompt long bị loại oan (server test_6)
    # vì đây vốn là một danh sách tag mood tiếng Anh rời rạc, không phải câu mô tả mà
    # người dùng thật sẽ nói ra.
    checklist = trim_checklist_to_word_budget(
        out_groups, so_luong_lines, scene, cfg.get("max_checklist_words"),
    )

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
