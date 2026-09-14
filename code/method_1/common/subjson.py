"""Logic thuần của STEP 1b: dựng đầu vào cho LLM chọn lọc, kiểm tra lựa chọn, lắp sub_json.

Không gọi mạng -- tách riêng để test được ở máy không có server, và để common/prompts.py
dựng few-shot của 1b mà không phải import step1b.

Luồng:
    decomposition (1a) --build_selection_input--> đầu vào cho LLM
    LLM chọn cho cả 3 mức --validate_selection--> danh sách lỗi (rỗng = hợp lệ)
    --assemble_subjson--> sub_json của từng mức (checklist ghép thẳng từ lựa chọn)

Code KHÔNG quyết định giữ/bỏ nội dung nào -- đó là việc của LLM theo định nghĩa mức.
Code chỉ kiểm tra LLM tuân thủ: chép nguyên văn, lồng nhau, chủ thể bắt buộc, trần từ.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional

from common import schema

LEVELS = ("short", "medium", "long")
LENGTH_HINT = {"short": "8-20 từ", "medium": "30-60 từ", "long": "100-200 từ"}
MAX_CHECKLIST_WORDS: Dict[str, Optional[int]] = {"short": 16, "medium": 50, "long": None}
SHORT_MAX_IDEAS = 2
STYLE_FIELDS = ("photo", "art_style", "lighting", "aesthetics")

# Hai nhóm trùng tên được gắn hậu tố " [2]", " [3]" để LLM tham chiếu không nhập nhằng;
# hậu tố bị bóc đi khi lắp sub_json.
_DUP_SUFFIX = re.compile(r" \[\d+\]$")


# Tên file trong bộ dữ liệu là slug không dấu của CHỦ THỂ CHÍNH thật sự (vd
# "hu_tieu_nam_vang_001001"). Đây là tín hiệu độc lập với việc 1a xếp nhóm nào đứng đầu.
def strip_vn_diacritics(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def filename_keyword_words(row_id: str) -> List[str]:
    """"ruoc_kieu_002128" -> ["ruoc", "kieu"] -- bỏ hậu tố số thứ tự cuối cùng."""
    slug = re.sub(r"_\d+$", "", row_id)
    return [w for w in slug.split("_") if w]


def find_required_group_index(groups: List[Dict[str, Any]], row_id: str) -> Optional[int]:
    keywords = filename_keyword_words(row_id)
    if not keywords:
        return None
    for i, group in enumerate(groups):
        name_ascii = strip_vn_diacritics(str(group.get("name", ""))).lower()
        if all(kw in name_ascii for kw in keywords):
            return i
    return None


def _cardinality_n(cardinality: Any) -> Optional[int]:
    if isinstance(cardinality, str) and cardinality.startswith("exact:"):
        try:
            return int(cardinality.split(":", 1)[1])
        except ValueError:
            return None
    return None


def count_fact(cardinality: Any) -> Optional[str]:
    n = _cardinality_n(cardinality)
    return "số lượng: {}".format(n) if n is not None and n >= 2 else None


def is_plural(cardinality: Any) -> bool:
    if cardinality == "vague":
        return True
    n = _cardinality_n(cardinality)
    return n is not None and n >= 2


def original_name(label: str) -> str:
    return _DUP_SUFFIX.sub("", label)


def _texts(items: Any) -> List[str]:
    return [str(f["text"]) for f in items or [] if isinstance(f, dict) and f.get("text")]


def _dedupe(items: List[str]) -> List[str]:
    out: List[str] = []
    for x in items:
        if x not in out:
            out.append(x)
    return out


def build_selection_input(
    target: Dict[str, Any],
    decomposition: Dict[str, Any],
    required_group_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Sắp xếp lại decomposition theo nhóm cho LLM đọc. Không bỏ bớt thông tin nào."""
    facts_map = decomposition.get("facts") or {}

    raw_groups: List[Dict[str, Any]] = []
    grouped_ids = set()
    for group in decomposition.get("concept_groups") or []:
        members = list(group.get("member_ids") or [])
        grouped_ids.update(members)
        raw_groups.append({
            "name": str(group.get("name", "")).strip(),
            "cultural": bool(group.get("cultural", False)),
            "cardinality": group.get("cardinality", "vague"),
            "elements": [_texts(facts_map.get(str(m))) for m in members
                         if _texts(facts_map.get(str(m)))],
        })

    # Element không thuộc nhóm nào vẫn là thông tin của ảnh -- thành một nhóm riêng.
    for el in schema.elements_of(target):
        if el.get("id") in grouped_ids:
            continue
        facts = _texts(facts_map.get(str(el.get("id"))))
        if facts:
            raw_groups.append({"name": facts[0], "cultural": False,
                               "cardinality": "exact:1", "elements": [facts]})

    seen: Dict[str, int] = {}
    groups: List[Dict[str, Any]] = []
    for group in raw_groups:
        n = seen.get(group["name"], 0) + 1
        seen[group["name"]] = n
        entry: Dict[str, Any] = {
            "name": group["name"] if n == 1 else "{} [{}]".format(group["name"], n),
            "cultural": group["cultural"],
            "cardinality": group["cardinality"],
        }
        so_luong = count_fact(group["cardinality"])
        if so_luong:
            entry["so_luong"] = so_luong
        entry["elements"] = group["elements"]
        groups.append(entry)

    required = None
    if required_group_index is not None and 0 <= required_group_index < len(groups):
        required = groups[required_group_index]["name"]

    style_facts = decomposition.get("style_facts") or {}
    return {
        "high_level_description": str(target.get("high_level_description", "")),
        "medium": str((target.get("style_description") or {}).get("medium", "")),
        "required_subject": required,
        "groups": groups,
        "background": _texts(decomposition.get("background_facts")),
        "style": {k: _texts(style_facts.get(k)) for k in STYLE_FIELDS if _texts(style_facts.get(k))},
    }


def _parse_level(
    level: str, raw: Any, sel_input: Dict[str, Any], errors: List[str],
) -> Optional[Dict[str, Any]]:
    """Đọc lựa chọn của một mức; lỗi được ghi vào `errors`, phần hợp lệ được giữ lại."""
    if not isinstance(raw, dict):
        errors.append("thiếu mức '{}'".format(level))
        return None
    by_name = {g["name"]: g for g in sel_input["groups"]}

    raw_groups = raw.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        errors.append("{}: 'groups' rỗng hoặc sai kiểu".format(level))
        return None

    groups: List[Dict[str, Any]] = []
    for item in raw_groups:
        name = item.get("name") if isinstance(item, dict) else None
        if name not in by_name:
            errors.append("{}: nhóm {!r} không có trong đầu vào".format(level, name))
            continue
        if any(g["name"] == name for g in groups):
            errors.append("{}: nhóm {!r} bị chọn hai lần".format(level, name))
            continue
        src = by_name[name]
        allowed = {f for element in src["elements"] for f in element}
        if src.get("so_luong"):
            allowed.add(src["so_luong"])
        facts = item.get("facts") or []
        if not isinstance(facts, list):
            errors.append("{}: 'facts' của nhóm {!r} sai kiểu".format(level, name))
            facts = []
        bad = [f for f in facts if f not in allowed]
        if bad:
            errors.append("{}: nhóm {!r} có mệnh đề không chép nguyên văn từ đầu vào: {}".format(
                level, name, bad))
        kept = _dedupe([f for f in facts if f in allowed])
        # Tên nhóm thường chỉ là nhãn (vd "cảnh nền") -- chọn nhóm có cá thể mà không chọn
        # mệnh đề nào thì không có gì để nói. Nhóm văn hoá thì tên chính là nội dung.
        if src["elements"] and not src["cultural"] and not kept:
            errors.append("{}: nhóm {!r} được chọn nhưng không có mệnh đề nào".format(level, name))
        groups.append({"name": name, "facts": kept})

    background = raw.get("background") or []
    if not isinstance(background, list):
        errors.append("{}: 'background' sai kiểu".format(level))
        background = []
    bad = [f for f in background if f not in sel_input["background"]]
    if bad:
        errors.append("{}: background có mệnh đề không chép nguyên văn từ đầu vào: {}".format(level, bad))
    kept_background = _dedupe([f for f in background if f in sel_input["background"]])

    style = raw.get("style") or {}
    if not isinstance(style, dict):
        errors.append("{}: 'style' sai kiểu".format(level))
        style = {}
    kept_style: Dict[str, List[str]] = {}
    for key, facts in style.items():
        if key not in sel_input["style"]:
            errors.append("{}: style có trường {!r} không có trong đầu vào".format(level, key))
            continue
        if not isinstance(facts, list):
            errors.append("{}: style[{!r}] sai kiểu".format(level, key))
            continue
        bad = [f for f in facts if f not in sel_input["style"][key]]
        if bad:
            errors.append("{}: style[{!r}] có mệnh đề không chép nguyên văn từ đầu vào: {}".format(
                level, key, bad))
        kept = _dedupe([f for f in facts if f in sel_input["style"][key]])
        if kept:
            kept_style[key] = kept

    medium = raw.get("medium", False)
    if not isinstance(medium, bool):
        errors.append("{}: 'medium' phải là true/false".format(level))
        medium = False

    return {"groups": groups, "background": kept_background, "style": kept_style, "medium": medium}


def _level_view(parsed: Dict[str, Any], sel_input: Dict[str, Any]) -> Dict[str, Any]:
    """Nội dung cuối cùng của một mức -- dùng CHUNG cho kiểm tra trần từ và lắp sub_json,
    để thứ được đếm và thứ đưa cho 2a / checklist luôn là một."""
    by_name = {g["name"]: g for g in sel_input["groups"]}
    groups_out: List[Dict[str, Any]] = []
    for group in parsed["groups"]:
        src = by_name[group["name"]]
        name = original_name(group["name"])
        facts = list(group["facts"])
        if (src["cultural"] or not facts) and name not in facts:
            facts = [name] + facts
        has_count = bool(src.get("so_luong")) and src["so_luong"] in facts
        groups_out.append({
            "name": name,
            "cultural": src["cultural"],
            "cardinality": src["cardinality"],
            "so_nhieu": is_plural(src["cardinality"]) and not has_count,
            "facts": facts,
        })
    style = {k: parsed["style"][k] for k in STYLE_FIELDS if parsed["style"].get(k)}
    medium = sel_input.get("medium") if parsed["medium"] and sel_input.get("medium") else None

    checklist = [f for g in groups_out for f in g["facts"]]
    checklist += parsed["background"]
    checklist += [f for k in STYLE_FIELDS for f in style.get(k, [])]
    if medium:
        checklist.append(medium)
    return {"groups": groups_out, "background": list(parsed["background"]),
            "style": style, "medium": medium, "checklist": checklist}


def checklist_words(checklist: List[str]) -> int:
    return sum(len(item.split()) for item in checklist)


def short_idea_count(parsed: Dict[str, Any], sel_input: Dict[str, Any]) -> int:
    """Số ý NGOÀI danh từ chủ thể: thuộc tính, số lượng, bối cảnh, nhóm thêm, loại ảnh."""
    by_name = {g["name"]: g for g in sel_input["groups"]}
    ideas = max(0, len(parsed["groups"]) - 1) + len(parsed["background"])
    ideas += 1 if parsed["medium"] else 0
    for group in parsed["groups"]:
        nouns = {element[0] for element in by_name[group["name"]]["elements"] if element}
        ideas += sum(1 for f in group["facts"] if f not in nouns)
    return ideas


def validate_selection(selection: Any, sel_input: Dict[str, Any]) -> List[str]:
    if not isinstance(selection, dict):
        return ["đầu ra không phải object"]
    errors: List[str] = []
    parsed = {level: _parse_level(level, selection.get(level), sel_input, errors) for level in LEVELS}

    required = sel_input.get("required_subject")
    for level, p in parsed.items():
        if p is None:
            continue
        if required and all(g["name"] != required for g in p["groups"]):
            errors.append("{}: thiếu nhóm chủ thể bắt buộc {!r}".format(level, required))
        cap = MAX_CHECKLIST_WORDS[level]
        if cap is not None:
            words = checklist_words(_level_view(p, sel_input)["checklist"])
            if words > cap:
                errors.append("{}: tổng {} từ, vượt trần {} từ".format(level, words, cap))

    short = parsed["short"]
    if short is not None:
        if short["style"]:
            errors.append("short: không được chọn 'style'")
        if len(short["background"]) > 1:
            errors.append("short: tối đa 1 ý bối cảnh")
        ideas = short_idea_count(short, sel_input)
        if ideas > SHORT_MAX_IDEAS:
            errors.append("short: có {} ý ngoài chủ thể chính, tối đa {}".format(ideas, SHORT_MAX_IDEAS))

    for low, high in (("short", "medium"), ("medium", "long")):
        a, b = parsed[low], parsed[high]
        if a is None or b is None:
            continue
        b_groups = {g["name"]: set(g["facts"]) for g in b["groups"]}
        for group in a["groups"]:
            if group["name"] not in b_groups:
                errors.append("lồng nhau: nhóm {!r} có ở {} nhưng thiếu ở {}".format(
                    group["name"], low, high))
                continue
            lost = [f for f in group["facts"] if f not in b_groups[group["name"]]]
            if lost:
                errors.append("lồng nhau: nhóm {!r} ở {} có {} nhưng {} không có".format(
                    group["name"], low, lost, high))
        lost_bg = [f for f in a["background"] if f not in b["background"]]
        if lost_bg:
            errors.append("lồng nhau: background {} có ở {} nhưng thiếu ở {}".format(lost_bg, low, high))
        for key, facts in a["style"].items():
            lost_style = [f for f in facts if f not in b["style"].get(key, [])]
            if lost_style:
                errors.append("lồng nhau: style[{!r}] {} có ở {} nhưng thiếu ở {}".format(
                    key, lost_style, low, high))
        if a["medium"] and not b["medium"]:
            errors.append("lồng nhau: medium có ở {} nhưng thiếu ở {}".format(low, high))

    return errors


def assemble_subjson(
    row_id: str, sel_input: Dict[str, Any], selection: Dict[str, Any], level: str,
) -> Dict[str, Any]:
    """Lắp sub_json của một mức từ lựa chọn ĐÃ QUA validate_selection."""
    errors: List[str] = []
    parsed = _parse_level(level, selection.get(level), sel_input, errors)
    if parsed is None:
        raise ValueError("lựa chọn không có mức {}: {}".format(level, errors))
    view = _level_view(parsed, sel_input)

    required_subject = None
    required = sel_input.get("required_subject")
    if required:
        for picked, group in zip(parsed["groups"], view["groups"]):
            if picked["name"] == required and group["facts"]:
                required_subject = group["facts"][0]
                break

    return {
        "id": row_id,
        "detail_level": level,
        "length_hint": LENGTH_HINT[level],
        "groups": view["groups"],
        "background": view["background"],
        "style": view["style"],
        "medium": view["medium"],
        "checklist": view["checklist"],
        "required_subject": required_subject,
    }
