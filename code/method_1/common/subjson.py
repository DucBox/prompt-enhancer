"""Logic thuần của STEP 1b: dựng đầu vào cho LLM chọn lọc, kiểm tra lựa chọn, lắp sub_json.

Không gọi mạng -- tách riêng để test được ở máy không có server, và để common/prompts.py
dựng few-shot của 1b mà không phải import step1b.

Luồng:
    decomposition (1a) --build_selection_input--> đầu vào cho LLM
    LLM chọn cho cả 3 mức --validate_selection--> danh sách lỗi (rỗng = hợp lệ)
    --assemble_subjson--> sub_json của từng mức (checklist ghép thẳng từ lựa chọn)

Code KHÔNG quyết định giữ/bỏ nội dung nào -- đó là việc của LLM theo định nghĩa mức.
Code chỉ kiểm tra LLM tuân thủ: chép nguyên văn, lồng nhau, trần từ.

Chủ đề chính (`chu_de_chinh`) do 1a rút từ high_level_description, có tham khảo gợi ý từ
tên thư mục. Code TỰ chèn nó vào cả ba mức -- LLM của 1b không chọn nên không làm mất được.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import schema

LEVELS = ("short", "medium", "long")
LENGTH_HINT = {"short": "8-20 từ", "medium": "30-60 từ", "long": "100-200 từ"}
MAX_CHECKLIST_WORDS: Dict[str, Optional[int]] = {"short": 16, "medium": 50, "long": None}
# Trần LLM nhắm tới (ghi trong prompt) và ngưỡng code thực sự chấp nhận là hai số khác nhau:
# LLM đếm từ không chính xác, hay lệch vài tiếng. Đổi ngưỡng này KHÔNG làm mất cache 1b vì
# dấu vân tay chỉ tính prompt.
ACCEPT_CHECKLIST_WORDS: Dict[str, Optional[int]] = {"short": 20, "medium": 55, "long": None}
SHORT_MAX_IDEAS = 2
STYLE_FIELDS = ("photo", "art_style", "lighting", "aesthetics")

# Hai nhóm trùng tên được gắn hậu tố " [2]", " [3]" để LLM tham chiếu không nhập nhằng;
# hậu tố bị bóc đi khi lắp sub_json.
_DUP_SUFFIX = re.compile(r" \[\d+\]$")


TOPIC_TERMS_FILE = Path(__file__).resolve().parent / "topic_terms.json"


# Tên file trong bộ dữ liệu là slug không dấu của thư mục chủ đề (vd "hu_tieu_nam_vang_001001").
# Chỉ là GỢI Ý: slug hay thừa chữ ("banh_mi_viet_nam"), bị cắt ("..._ho_chi_min"), hoặc chỉ là
# từ khoá tìm kiếm mà ảnh không thể hiện. Chủ đề thật do 1a đọc từ high_level_description.
def strip_vn_diacritics(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def topic_slug(row_id: str) -> str:
    return re.sub(r"_\d+$", "", row_id)


def filename_keyword_words(row_id: str) -> List[str]:
    """"ruoc_kieu_002128" -> ["ruoc", "kieu"] -- bỏ hậu tố số thứ tự cuối cùng."""
    return [w for w in topic_slug(row_id).split("_") if w]


@lru_cache(maxsize=None)
def load_topic_terms(path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """slug -> mục trong topic_terms.json. Không có file thì trả rỗng (gợi ý chỉ còn tên thư mục)."""
    file = Path(path) if path else TOPIC_TERMS_FILE
    if not file.is_file():
        return {}
    data = json.loads(file.read_text(encoding="utf-8"))
    return {t["slug"]: t for t in data.get("topics", []) if t.get("slug")}


def topic_hint(row_id: str, topics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Gợi ý chủ đề đưa cho 1a: tên thư mục + thuật ngữ có dấu nếu bảng chủ đề có."""
    hint: Dict[str, Any] = {"ten_thu_muc": " ".join(filename_keyword_words(row_id))}
    topic = topics.get(topic_slug(row_id))
    if topic:
        hint["thuat_ngu"] = topic["thuat_ngu"]
        # Không đưa "lien_quan" (Bát Tràng, Nhật Tân...): chỉ đúng với một phần ảnh, dễ bị 1a
        # chèn vào chủ đề khi mô tả gốc không nhắc tới.
        if topic.get("dong_nghia"):
            hint["dong_nghia"] = list(topic["dong_nghia"])
    return hint


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

    style_facts = decomposition.get("style_facts") or {}
    return {
        "high_level_description": str(target.get("high_level_description", "")),
        "chu_de_chinh": _texts(decomposition.get("chu_de_chinh")),
        "medium": str((target.get("style_description") or {}).get("medium", "")),
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
    # Có chủ đề chính thì mức đó đã có nội dung -- được phép không chọn thêm nhóm nào.
    if not isinstance(raw_groups, list) or (not raw_groups and not sel_input.get("chu_de_chinh")):
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
    theme = list(sel_input.get("chu_de_chinh") or [])

    # Chủ đề chính đứng đầu checklist; mệnh đề trùng với nó (vd tên nhóm văn hoá) chỉ tính một lần.
    checklist = list(theme) + [f for g in groups_out for f in g["facts"]]
    checklist += parsed["background"]
    checklist += [f for k in STYLE_FIELDS for f in style.get(k, [])]
    if medium:
        checklist.append(medium)
    return {"chu_de_chinh": theme, "groups": groups_out, "background": list(parsed["background"]),
            "style": style, "medium": medium, "checklist": _dedupe(checklist)}


def checklist_words(checklist: List[str]) -> int:
    return sum(len(item.split()) for item in checklist)


def chosen_words(view: Dict[str, Any]) -> int:
    """Số từ LLM 1b tự chọn -- không tính chủ đề chính, vì LLM không bỏ được nó."""
    return checklist_words(view["checklist"]) - checklist_words(view["chu_de_chinh"])


def short_idea_count(parsed: Dict[str, Any], sel_input: Dict[str, Any]) -> int:
    """Số ý NGOÀI danh từ chủ thể: thuộc tính, số lượng, bối cảnh, nhóm thêm, loại ảnh."""
    by_name = {g["name"]: g for g in sel_input["groups"]}
    ideas = max(0, len(parsed["groups"]) - 1) + len(parsed["background"])
    ideas += 1 if parsed["medium"] else 0
    for group in parsed["groups"]:
        nouns = {element[0] for element in by_name[group["name"]]["elements"] if element}
        ideas += sum(1 for f in group["facts"] if f not in nouns)
    return ideas


def _nesting_errors(low: str, high: str, a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
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


def evaluate_selection(selection: Any, sel_input: Dict[str, Any]) -> Dict[str, Any]:
    """Kiểm tra lựa chọn, trả về:
        errors  mọi lỗi (gửi lại cho LLM sửa)
        levels  các mức dùng được -- tự hợp lệ VÀ lồng nhau với các mức khác được giữ
        dropped {mức bị bỏ: lý do}

    Một mức hỏng không kéo theo các mức còn lại: mỗi dòng train là một cặp độc lập
    (prompt của mức đó -> JSON gốc đầy đủ). Lồng nhau chỉ xét giữa các mức được giữ; hai mức
    tự hợp lệ mà không lồng nhau thì bỏ mức ít chi tiết hơn.
    """
    if not isinstance(selection, dict):
        return {"errors": ["đầu ra không phải object"], "levels": [],
                "dropped": {level: ["đầu ra không phải object"] for level in LEVELS}}
    level_errors: Dict[str, List[str]] = {level: [] for level in LEVELS}
    parsed = {level: _parse_level(level, selection.get(level), sel_input, level_errors[level])
              for level in LEVELS}

    for level, p in parsed.items():
        if p is None:
            continue
        cap = ACCEPT_CHECKLIST_WORDS[level]
        if cap is not None:
            words = chosen_words(_level_view(p, sel_input))
            if words > cap:
                level_errors[level].append("{}: tổng {} từ, vượt trần {} từ".format(
                    level, words, MAX_CHECKLIST_WORDS[level]))

    short = parsed["short"]
    if short is not None:
        if short["style"]:
            level_errors["short"].append("short: không được chọn 'style'")
        if len(short["background"]) > 1:
            level_errors["short"].append("short: tối đa 1 ý bối cảnh")
        ideas = short_idea_count(short, sel_input)
        if ideas > SHORT_MAX_IDEAS:
            level_errors["short"].append(
                "short: có {} ý ngoài chủ thể chính, tối đa {}".format(ideas, SHORT_MAX_IDEAS))

    errors = [e for level in LEVELS for e in level_errors[level]]
    for low, high in (("short", "medium"), ("medium", "long")):
        if parsed[low] is not None and parsed[high] is not None:
            errors += _nesting_errors(low, high, parsed[low], parsed[high])

    dropped = {level: errs for level, errs in level_errors.items() if errs}
    # Đi từ mức chi tiết nhất xuống: mỗi mức phải nằm trong mức lớn gần nhất còn được giữ
    # (lồng nhau có tính bắc cầu nên so với mức gần nhất là đủ).
    kept: List[str] = []
    for level in reversed(LEVELS):
        if level in dropped:
            continue
        if kept:
            nest = _nesting_errors(level, kept[-1], parsed[level], parsed[kept[-1]])
            if nest:
                dropped[level] = nest
                continue
        kept.append(level)

    return {"errors": errors, "levels": [level for level in LEVELS if level in kept],
            "dropped": dropped}


def validate_selection(selection: Any, sel_input: Dict[str, Any]) -> List[str]:
    return evaluate_selection(selection, sel_input)["errors"]


def assemble_subjson(
    row_id: str, sel_input: Dict[str, Any], selection: Dict[str, Any], level: str,
) -> Dict[str, Any]:
    """Lắp sub_json của một mức nằm trong `levels` của evaluate_selection."""
    errors: List[str] = []
    parsed = _parse_level(level, selection.get(level), sel_input, errors)
    if parsed is None:
        raise ValueError("lựa chọn không có mức {}: {}".format(level, errors))
    view = _level_view(parsed, sel_input)

    return {
        "id": row_id,
        "detail_level": level,
        "length_hint": LENGTH_HINT[level],
        "chu_de_chinh": view["chu_de_chinh"],
        "groups": view["groups"],
        "background": view["background"],
        "style": view["style"],
        "medium": view["medium"],
        "checklist": view["checklist"],
        # 2b loại ngay nếu judge báo thiếu bất kỳ mệnh đề nào ở đây, bất kể --max_missing.
        "required_facts": list(view["chu_de_chinh"]),
    }
