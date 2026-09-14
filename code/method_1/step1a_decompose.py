#!/usr/bin/env python3
"""STEP 1a — Gom nhóm khái niệm & phân rã mệnh đề.  [GỌI MODEL]

Đầu vào : <out_root>/step0_normalized/targets.jsonl
Đầu ra  : <out_root>/step1a_decompose/
            decompose/<id>.json   {"chu_de_chinh", "khop_goi_y", "concept_groups", "facts",
                                   "background_facts", "style_facts", "field_priority",
                                   "goi_y_chu_de"}
            decompose.jsonl       gộp lại thành một file
            failures.json

Phân rã ĐẦY ĐỦ mọi trường mang nội dung, mỗi mệnh đề một ý, bằng tiếng Việt, xếp hạng
theo độ quan trọng TRONG TỪNG TRƯỜNG:
    chu_de_chinh      <- high_level_description (1-3 mệnh đề: bức ảnh VỀ CÁI GÌ)
    facts             <- elements[].desc
    background_facts  <- compositional_deconstruction.background
    style_facts       <- style_description.{photo | art_style, lighting, aesthetics}
Không phân rã `medium` (enum).
Bước này KHÔNG bỏ bớt thông tin -- chọn giữ/bỏ theo mức là việc của 1b.

Gợi ý chủ đề (tên thư mục + common/topic_terms.json) chỉ để 1a THAM KHẢO cách gọi tên chủ đề;
1a tự xác nhận ảnh có thể hiện chủ đề đó không (`khop_goi_y`).

Chạy một lần rồi CACHE: file <id>.json đã có (đúng schema hiện tại VÀ cùng gợi ý chủ đề) thì
bỏ qua, nên có thể dừng giữa chừng và chạy lại để tiếp tục.

Ví dụ:
    python step1a_decompose.py --out_root test_1 --dry_run
    python step1a_decompose.py --out_root test_1 --workers 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import config, io_utils, llm, prompts, schema, subjson

STEP = "STEP 1a"

VALID_KINDS = {"subject", "color", "material", "attribute", "action", "position", "count"}

# Các trường style được phân rã. `medium` là enum nên không phân rã.
STYLE_FACT_FIELDS = ("photo", "art_style", "lighting", "aesthetics")

# Chủ đề chính là ý cốt lõi, không phải bản tóm tắt -- quá 3 mệnh đề là đang liệt kê chi tiết.
MAX_THEME_FACTS = 3
MAX_THEME_WORDS = 10

# Độ ưu tiên XUYÊN TRƯỜNG. Hiện chưa đánh giá nên mọi trường bằng nhau; sau này có thể
# cho LLM chấm lại thành 2-3 mà không phải đổi schema.
DEFAULT_FIELD_PRIORITY = 1

# Cache sinh ra trước khi có các khoá này là schema cũ -- phải gọi lại, không dùng lẫn.
CACHE_REQUIRED_KEYS = ("chu_de_chinh", "background_facts", "style_facts", "field_priority")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM gom nhóm & phân rã mệnh đề")
    p.add_argument("--in_file", default=None, help="Mặc định: <out_root>/step0_normalized/targets.jsonl")
    p.add_argument("--out_dir", default=None, help="Ghi đè thư mục đầu ra của step này")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Để 0 cho deterministic — kết quả được cache và đóng băng")
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--overwrite", action="store_true", help="Bỏ qua cache, gọi lại toàn bộ")
    p.add_argument("--dry_run", action="store_true",
                   help="Chỉ dựng messages và in ra, KHÔNG gọi model")
    return io_utils.add_common_args(p).parse_args()


def expected_style_fields(target: Dict[str, Any]) -> List[str]:
    style = target.get("style_description") or {}
    return [f for f in STYLE_FACT_FIELDS if str(style.get(f) or "").strip()]


def _check_ranked(items: Any, label: str, errors: List[str]) -> None:
    if not isinstance(items, list) or not items:
        errors.append("{} thiếu hoặc rỗng".format(label))
        return
    ranks = [f.get("rank") for f in items if isinstance(f, dict)]
    if 0 not in ranks:
        errors.append("{} không có mệnh đề rank 0".format(label))
    if any(not isinstance(f, dict) or not str(f.get("text") or "").strip() for f in items):
        errors.append("{} có mệnh đề thiếu text".format(label))


def validate_decomposition(result: Any, target: Dict[str, Any]) -> List[str]:
    """Kiểm tra đầu ra của model trước khi tin dùng."""
    errors: List[str] = []
    if not isinstance(result, dict):
        return ["đầu ra không phải object"]

    groups = result.get("concept_groups")
    facts = result.get("facts")

    if not isinstance(groups, list) or not groups:
        errors.append("concept_groups rỗng hoặc sai kiểu")
    if not isinstance(facts, dict) or not facts:
        errors.append("facts rỗng hoặc sai kiểu")
    if errors:
        return errors

    _check_ranked(result.get("chu_de_chinh"), "chu_de_chinh", errors)
    theme = result.get("chu_de_chinh")
    if isinstance(theme, list) and len(theme) > MAX_THEME_FACTS:
        errors.append("chu_de_chinh có {} mệnh đề, tối đa {}".format(len(theme), MAX_THEME_FACTS))
    if isinstance(theme, list):
        n_words = sum(len(str(f.get("text", "")).split()) for f in theme if isinstance(f, dict))
        if n_words > MAX_THEME_WORDS:
            errors.append("chu_de_chinh có tổng {} từ, tối đa {}".format(n_words, MAX_THEME_WORDS))

    element_ids = {el.get("id") for el in schema.elements_of(target)}

    for i, group in enumerate(groups):
        if not isinstance(group, dict):
            errors.append("concept_groups[{}] sai kiểu".format(i))
            continue
        if not group.get("name"):
            errors.append("concept_groups[{}] thiếu name".format(i))
        card = group.get("cardinality")
        if not isinstance(card, str) or not (card == "vague" or card.startswith("exact:")):
            errors.append("concept_groups[{}].cardinality={!r} không hợp lệ".format(i, card))
        members = group.get("member_ids")
        if not isinstance(members, list):
            errors.append("concept_groups[{}].member_ids sai kiểu".format(i))
        else:
            unknown = [m for m in members if m not in element_ids]
            if unknown:
                errors.append("concept_groups[{}] có id lạ: {}".format(i, unknown))

    for element_id in element_ids:
        key = str(element_id)
        _check_ranked(facts.get(key), "facts['{}']".format(key), errors)

    comp = target.get("compositional_deconstruction") or {}
    if str(comp.get("background") or "").strip():
        _check_ranked(result.get("background_facts"), "background_facts", errors)

    expected = expected_style_fields(target)
    style_facts = result.get("style_facts")
    if expected:
        if not isinstance(style_facts, dict):
            errors.append("style_facts thiếu hoặc sai kiểu")
        else:
            for field in expected:
                _check_ranked(style_facts.get(field), "style_facts['{}']".format(field), errors)
            # Bắt nhầm photo <-> art_style, hoặc lỡ phân rã cả medium.
            unexpected = sorted(set(style_facts) - set(expected))
            if unexpected:
                errors.append("style_facts có trường không có trong đầu vào: {}".format(unexpected))

    return errors


def _clean_ranked(items: Any, with_kind: bool) -> List[Dict[str, Any]]:
    cleaned = []
    for f in items or []:
        if not isinstance(f, dict) or not str(f.get("text") or "").strip():
            continue
        item: Dict[str, Any] = {"rank": int(f.get("rank", 99))}
        if with_kind:
            kind = f.get("kind")
            item["kind"] = kind if kind in VALID_KINDS else "attribute"
        item["text"] = str(f["text"]).strip()
        cleaned.append(item)
    cleaned.sort(key=lambda f: f["rank"])
    # Đánh lại rank liên tục 0..n-1 để bước sau chọn theo thứ hạng cho chuẩn.
    for new_rank, f in enumerate(cleaned):
        f["rank"] = new_rank
    return cleaned


def normalize_decomposition(result: Dict[str, Any]) -> Dict[str, Any]:
    """Sắp xếp lại cho ổn định, loại kind lạ, gắn field_priority."""
    groups = []
    for group in result.get("concept_groups", []):
        groups.append({
            "name": str(group.get("name", "")).strip(),
            "member_ids": sorted(int(m) for m in group.get("member_ids", [])),
            "cardinality": group.get("cardinality", "vague"),
            "cultural": bool(group.get("cultural", False)),
        })

    facts = {str(key): _clean_ranked(items, with_kind=True)
             for key, items in (result.get("facts") or {}).items()}
    background_facts = _clean_ranked(result.get("background_facts"), with_kind=True)
    style_facts = {
        field: _clean_ranked(items, with_kind=False)
        for field, items in (result.get("style_facts") or {}).items()
        if field in STYLE_FACT_FIELDS
    }

    field_priority = {"elements": DEFAULT_FIELD_PRIORITY}
    if background_facts:
        field_priority["background"] = DEFAULT_FIELD_PRIORITY
    for field in style_facts:
        field_priority[field] = DEFAULT_FIELD_PRIORITY

    return {
        "chu_de_chinh": _clean_ranked(result.get("chu_de_chinh"), with_kind=False),
        "khop_goi_y": bool(result.get("khop_goi_y", False)),
        "concept_groups": groups,
        "facts": facts,
        "background_facts": background_facts,
        "style_facts": style_facts,
        "field_priority": field_priority,
    }


def is_cache_current(path: Path, hint: Optional[Dict[str, Any]] = None) -> bool:
    """Đúng schema hiện tại; có `hint` thì cache còn phải sinh ra từ đúng gợi ý chủ đề đó
    (sửa topic_terms.json thì ảnh thuộc chủ đề đó tự được gọi lại)."""
    try:
        item = io_utils.read_json(path)
    except (OSError, ValueError):
        return False
    if not isinstance(item, dict) or not all(k in item for k in CACHE_REQUIRED_KEYS):
        return False
    return hint is None or item.get("goi_y_chu_de") == hint


def main() -> None:
    args = parse_args()
    io_utils.banner(STEP, "LLM gom nhóm khái niệm & phân rã mệnh đề")
    config.load_env(args.env_file)

    in_file = io_utils.resolve(args.in_file, args.out_root, "step0", "targets.jsonl")
    rows = io_utils.read_jsonl(in_file)
    rows = io_utils.apply_test_mode(rows, args, "mẫu")

    out_dir = io_utils.resolve(args.out_dir, args.out_root, "step1a")
    cache_dir = out_dir / "decompose"
    cache_dir.mkdir(parents=True, exist_ok=True)

    topics = subjson.load_topic_terms()
    print("bảng chủ đề: {} slug ({})".format(len(topics), subjson.TOPIC_TERMS_FILE))
    hints = {row["id"]: subjson.topic_hint(row["id"], topics) for row in rows}

    if args.dry_run:
        row = rows[0]
        messages = prompts.build_step1a_messages(row["target_json"], hints[row["id"]])
        print("\n[DRY RUN] dựng được {} messages cho id={}".format(len(messages), row["id"]))
        print("\n--- user cuối cùng ---\n{}".format(messages[-1]["content"][:1200]))
        return

    todo = []
    n_cached = 0
    n_stale = 0
    for row in rows:
        path = cache_dir / "{}.json".format(row["id"])
        if not args.overwrite and path.is_file():
            if is_cache_current(path, hints[row["id"]]):
                n_cached += 1
                continue
            n_stale += 1
        todo.append(row)

    print("đã có cache: {} | cache schema cũ (gọi lại): {} | cần gọi model: {}".format(
        n_cached, n_stale, len(todo)))

    failures: List[Dict[str, Any]] = []

    if todo:
        client = llm.LLMClient()
        print("endpoint : {}".format(client.endpoint))
        print("model    : {}\n".format(client.model))

        def process(row: Dict[str, Any]) -> Optional[str]:
            target = row["target_json"]
            messages = prompts.build_step1a_messages(target, hints[row["id"]])
            result = client.chat_json(
                messages, temperature=args.temperature, max_tokens=args.max_tokens,
            )
            errors = validate_decomposition(result, target)
            if errors:
                raise llm.LLMError("đầu ra không hợp lệ: {}".format("; ".join(errors[:4])))
            cleaned = normalize_decomposition(result)
            cleaned["goi_y_chu_de"] = hints[row["id"]]
            io_utils.write_json(cache_dir / "{}.json".format(row["id"]), cleaned)
            return row["id"]

        def on_error(row: Dict[str, Any], exc: Exception) -> None:
            failures.append({"id": row["id"], "error": str(exc)[:400]})

        llm.run_parallel(todo, process, workers=args.workers,
                         desc="phân rã", on_error=on_error)

    # Gộp cache thành một jsonl cho bước sau -- chỉ lấy file đúng schema hiện tại.
    merged: List[Dict[str, Any]] = []
    for row in rows:
        path = cache_dir / "{}.json".format(row["id"])
        if path.is_file() and is_cache_current(path, hints[row["id"]]):
            merged.append({"id": row["id"], **io_utils.read_json(path)})

    n_written = io_utils.write_jsonl(out_dir / "decompose.jsonl", merged)
    if failures:
        io_utils.write_json(out_dir / "failures.json", failures)

    io_utils.summary(**{
        "mẫu xử lý": len(rows),
        "dùng lại cache": n_cached,
        "cache cũ (schema/gợi ý đổi)": n_stale,
        "ảnh khớp gợi ý chủ đề": sum(1 for m in merged if m.get("khop_goi_y")),
        "gọi model": len(todo),
        "thất bại": len(failures),
        "ghi ra decompose.jsonl": n_written,
        "thư mục đầu ra": str(out_dir),
    })


if __name__ == "__main__":
    main()
