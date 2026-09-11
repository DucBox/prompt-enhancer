"""Chuẩn hoá và kiểm tra schema của target_json.

Luật (chốt ở bước 0 của plan):
  * bỏ `bbox` và `color_palette` ở mọi tầng
  * giữ thứ tự key ổn định để chuỗi target khi SFT là deterministic
  * `style_description` phải có `photo` HOẶC `art_style`, không được cả hai
  * mỗi element phải có `desc`, `type` thuộc {obj, text}
  * gán `id` ổn định cho từng element (chính là chỉ số trong mảng)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

DROP_KEYS = {"bbox", "color_palette"}

TOP_ORDER = ["high_level_description", "style_description", "compositional_deconstruction"]
STYLE_ORDER = ["aesthetics", "lighting", "photo", "art_style", "medium"]
COMP_ORDER = ["background", "elements"]
ELEM_ORDER = ["id", "type", "text", "desc"]


def strip_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_keys(v) for k, v in obj.items() if k not in DROP_KEYS}
    if isinstance(obj, list):
        return [strip_keys(x) for x in obj]
    return obj


def reorder(d: Dict[str, Any], order: List[str]) -> Dict[str, Any]:
    out = {k: d[k] for k in order if k in d}
    out.update({k: v for k, v in d.items() if k not in out})
    return out


def normalize(raw: Dict[str, Any], *, assign_ids: bool = True) -> Dict[str, Any]:
    obj = strip_keys(raw)

    if isinstance(obj.get("style_description"), dict):
        obj["style_description"] = reorder(obj["style_description"], STYLE_ORDER)

    comp = obj.get("compositional_deconstruction")
    if isinstance(comp, dict):
        elements = comp.get("elements")
        if isinstance(elements, list):
            fixed = []
            for index, el in enumerate(elements):
                if isinstance(el, dict):
                    if assign_ids:
                        el = dict(el)
                        el["id"] = index
                    fixed.append(reorder(el, ELEM_ORDER))
                else:
                    fixed.append(el)
            comp["elements"] = fixed
        obj["compositional_deconstruction"] = reorder(comp, COMP_ORDER)

    return reorder(obj, TOP_ORDER)


def validate(obj: Dict[str, Any]) -> List[str]:
    """Trả về danh sách lỗi; rỗng nghĩa là hợp lệ."""
    errors: List[str] = []

    for key in TOP_ORDER:
        if key not in obj:
            errors.append("thiếu trường '{}'".format(key))

    style = obj.get("style_description")
    if not isinstance(style, dict):
        errors.append("style_description không phải object")
    else:
        has_photo = "photo" in style
        has_art = "art_style" in style
        if has_photo and has_art:
            errors.append("style_description có cả photo lẫn art_style")
        elif not has_photo and not has_art:
            errors.append("style_description thiếu cả photo lẫn art_style")

    comp = obj.get("compositional_deconstruction")
    if not isinstance(comp, dict):
        errors.append("compositional_deconstruction không phải object")
        return errors

    if not comp.get("background"):
        errors.append("thiếu compositional_deconstruction.background")

    elements = comp.get("elements")
    if not isinstance(elements, list) or not elements:
        errors.append("elements rỗng hoặc không phải list")
        return errors

    for index, el in enumerate(elements):
        if not isinstance(el, dict):
            errors.append("elements[{}] không phải object".format(index))
            continue
        etype = el.get("type")
        if etype not in {"obj", "text"}:
            errors.append("elements[{}].type={!r} không thuộc (obj, text)".format(index, etype))
        if etype == "text" and not el.get("text"):
            errors.append("elements[{}] type=text nhưng thiếu 'text'".format(index))
        if not el.get("desc"):
            errors.append("elements[{}] thiếu 'desc'".format(index))

    return errors


def elements_of(target: Dict[str, Any]) -> List[Dict[str, Any]]:
    return target.get("compositional_deconstruction", {}).get("elements", []) or []


def element_by_id(target: Dict[str, Any], element_id: int) -> Dict[str, Any]:
    for el in elements_of(target):
        if el.get("id") == element_id:
            return el
    raise KeyError("Không có element id={}".format(element_id))


def stats(target: Dict[str, Any]) -> Dict[str, int]:
    elements = elements_of(target)
    comp = target.get("compositional_deconstruction", {})
    return {
        "n_elements": len(elements),
        "n_text_elements": sum(1 for e in elements if e.get("type") == "text"),
        "words_high_level": len(str(target.get("high_level_description", "")).split()),
        "words_background": len(str(comp.get("background", "")).split()),
        "words_desc_total": sum(len(str(e.get("desc", "")).split()) for e in elements),
    }
