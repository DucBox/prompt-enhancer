#!/usr/bin/env python3
"""Test cho pipeline method_1.

Chạy:  python tests/test_pipeline.py        (từ thư mục code/method_1)
   hoặc: python -m pytest tests/ -q

Phạm vi:
  * Logic thuần (step 0, 1b, 2c, persona, config): test đầy đủ hành vi.
  * Phần gọi model: KHÔNG gọi mạng — chỉ test những gì test được:
    dựng payload/headers/endpoint, parse response, validate đầu ra, dựng messages.
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import config, io_utils, llm, prompts, schema, subjson  # noqa: E402
import step1b_build_subjson as s1b  # noqa: E402
import step2a_verbalize as s2a  # noqa: E402
import step2b_filter as s2b  # noqa: E402
import step2b2_correct as s2b2  # noqa: E402
import step1a_decompose as s1a  # noqa: E402


SAMPLE_TARGET = {
    "high_level_description": "A photo of three friends in front of Lăng Bác.",
    "style_description": {
        "aesthetics": "documentary",
        "lighting": "nắng sớm",
        "photo": "eye-level, wide shot, deep focus",
        "medium": "photograph",
    },
    "compositional_deconstruction": {
        "background": "Quảng trường Ba Đình rộng; bầu trời trong xanh, vài đám mây trắng.",
        "elements": [
            {"id": 0, "type": "obj", "desc": "Nam thanh niên mặc áo sơ mi trắng đứng bên trái."},
            {"id": 1, "type": "obj", "desc": "Nam thanh niên mặc áo phông đen đứng giữa."},
            {"id": 2, "type": "obj", "desc": "Cô gái mặc áo dài trắng đứng bên phải."},
            {"id": 3, "type": "obj", "desc": "Lăng Chủ tịch Hồ Chí Minh bằng đá granite xám."},
            {"id": 4, "type": "obj", "desc": "Hàng cây xanh hai bên quảng trường."},
        ],
    },
}

SAMPLE_DECOMP = {
    "concept_groups": [
        {"name": "nhóm bạn", "member_ids": [0, 1, 2], "cardinality": "exact:3", "cultural": False},
        {"name": "Lăng Bác", "member_ids": [3], "cardinality": "exact:1", "cultural": True},
        {"name": "cảnh nền", "member_ids": [4], "cardinality": "vague", "cultural": False},
    ],
    "facts": {
        "0": [{"rank": 0, "kind": "subject", "text": "nam thanh niên"},
              {"rank": 1, "kind": "color", "text": "áo sơ mi trắng"},
              {"rank": 2, "kind": "position", "text": "đứng bên trái"}],
        "1": [{"rank": 0, "kind": "subject", "text": "nam thanh niên"},
              {"rank": 1, "kind": "color", "text": "áo phông đen"},
              {"rank": 2, "kind": "position", "text": "đứng giữa"}],
        "2": [{"rank": 0, "kind": "subject", "text": "cô gái"},
              {"rank": 1, "kind": "attribute", "text": "mặc áo dài trắng"},
              {"rank": 2, "kind": "position", "text": "đứng bên phải"}],
        "3": [{"rank": 0, "kind": "subject", "text": "Lăng Bác"},
              {"rank": 1, "kind": "material", "text": "bằng đá granite xám"}],
        "4": [{"rank": 0, "kind": "subject", "text": "hàng cây xanh"},
              {"rank": 1, "kind": "position", "text": "hai bên quảng trường"}],
    },
    "background_facts": [
        {"rank": 0, "kind": "subject", "text": "quảng trường Ba Đình"},
        {"rank": 1, "kind": "attribute", "text": "rộng"},
        {"rank": 2, "kind": "subject", "text": "bầu trời trong xanh"},
        {"rank": 3, "kind": "subject", "text": "vài đám mây trắng"},
    ],
    "style_facts": {
        "photo": [{"rank": 0, "text": "toàn cảnh"},
                  {"rank": 1, "text": "ngang tầm mắt"},
                  {"rank": 2, "text": "lấy nét sâu"}],
        "lighting": [{"rank": 0, "text": "nắng sớm"}],
        "aesthetics": [{"rank": 0, "text": "phong cách phóng sự"}],
    },
}


# =============================================================================
# STEP 0 — schema
# =============================================================================

class TestStripIds(unittest.TestCase):
    """`id` là tay cầm nội bộ, không thuộc schema Ideogram 4 -- phải bóc khỏi nhãn Y.

    Bắt được lỗi thật: outputs/test_6 có 1631 element mang `id` lọt vào target_json
    của tập train/val/test, đúng thứ mà CaptionVerifier của Ideogram báo
    "unknown keys ['id']".
    """

    def test_removes_id_from_every_element(self):
        out = schema.strip_ids(SAMPLE_TARGET)
        for el in out["compositional_deconstruction"]["elements"]:
            self.assertNotIn("id", el)

    def test_does_not_mutate_input(self):
        before = json.dumps(SAMPLE_TARGET, sort_keys=True)
        schema.strip_ids(SAMPLE_TARGET)
        self.assertEqual(json.dumps(SAMPLE_TARGET, sort_keys=True), before)

    def test_keeps_everything_except_id(self):
        out = schema.strip_ids(SAMPLE_TARGET)
        for src, got in zip(schema.elements_of(SAMPLE_TARGET),
                            out["compositional_deconstruction"]["elements"]):
            self.assertEqual({k: v for k, v in src.items() if k != "id"}, got)

    def test_element_key_order_matches_ideogram(self):
        out = schema.strip_ids(SAMPLE_TARGET)
        for el in out["compositional_deconstruction"]["elements"]:
            expected = [k for k in ("type", "text", "desc") if k in el]
            self.assertEqual(list(el), expected)

    def test_photo_captions_put_photo_before_medium(self):
        style = {"medium": "photograph", "lighting": "soft", "photo": "35mm",
                 "aesthetics": "warm"}
        self.assertEqual(list(schema.reorder(style, schema.STYLE_ORDER)),
                         ["aesthetics", "lighting", "photo", "medium"])

    def test_non_photo_captions_put_art_style_after_medium(self):
        """Ideogram đảo thứ tự cho caption không phải ảnh chụp -- xem
        third_party/ideogram4/src/ideogram4/caption_verifier.py."""
        style = {"art_style": "flat vector", "medium": "illustration",
                 "lighting": "even", "aesthetics": "minimal"}
        self.assertEqual(list(schema.reorder(style, schema.STYLE_ORDER)),
                         ["aesthetics", "lighting", "medium", "art_style"])

    def test_strip_ids_reorders_art_style_after_medium(self):
        target = json.loads(json.dumps(SAMPLE_TARGET))
        target["style_description"] = {"aesthetics": "a", "lighting": "l",
                                       "art_style": "flat vector", "medium": "illustration"}
        out = schema.strip_ids(target)
        self.assertEqual(list(out["style_description"]),
                         ["aesthetics", "lighting", "medium", "art_style"])


class TestSchema(unittest.TestCase):

    def test_strips_bbox_and_palette_recursively(self):
        raw = {
            "high_level_description": "x",
            "style_description": {"medium": "photograph", "photo": "wide",
                                  "color_palette": ["#fff"]},
            "compositional_deconstruction": {
                "background": "bg",
                "elements": [{"type": "obj", "desc": "d",
                              "bbox": [1, 2, 3, 4], "color_palette": ["#000"]}],
            },
        }
        out = schema.normalize(raw)
        text = json.dumps(out)
        self.assertNotIn("bbox", text)
        self.assertNotIn("color_palette", text)

    def test_assigns_stable_element_ids(self):
        raw = json.loads(json.dumps(SAMPLE_TARGET))
        for el in raw["compositional_deconstruction"]["elements"]:
            el.pop("id", None)
        out = schema.normalize(raw)
        ids = [el["id"] for el in schema.elements_of(out)]
        self.assertEqual(ids, [0, 1, 2, 3, 4])

    def test_key_order_is_deterministic(self):
        out = schema.normalize(json.loads(json.dumps(SAMPLE_TARGET)))
        self.assertEqual(list(out.keys()), schema.TOP_ORDER)

    def test_valid_sample_has_no_errors(self):
        self.assertEqual(schema.validate(SAMPLE_TARGET), [])

    def test_rejects_photo_and_art_style_together(self):
        bad = json.loads(json.dumps(SAMPLE_TARGET))
        bad["style_description"]["art_style"] = "oil painting"
        errors = schema.validate(bad)
        self.assertTrue(any("cả photo lẫn art_style" in e for e in errors))

    def test_rejects_text_element_without_text_field(self):
        bad = json.loads(json.dumps(SAMPLE_TARGET))
        bad["compositional_deconstruction"]["elements"][0]["type"] = "text"
        errors = schema.validate(bad)
        self.assertTrue(any("thiếu 'text'" in e for e in errors))

    def test_stats_counts_words(self):
        st = schema.stats(SAMPLE_TARGET)
        self.assertEqual(st["n_elements"], 5)
        self.assertEqual(st["n_text_elements"], 0)
        self.assertGreater(st["words_desc_total"], 0)


# =============================================================================
# STEP 1a — validate đầu ra của model (không gọi mạng)
# =============================================================================

class TestDecomposeValidation(unittest.TestCase):

    def test_accepts_good_output(self):
        self.assertEqual(s1a.validate_decomposition(SAMPLE_DECOMP, SAMPLE_TARGET), [])

    def test_rejects_unknown_element_id(self):
        bad = json.loads(json.dumps(SAMPLE_DECOMP))
        bad["concept_groups"][0]["member_ids"] = [0, 99]
        errors = s1a.validate_decomposition(bad, SAMPLE_TARGET)
        self.assertTrue(any("id lạ" in e for e in errors))

    def test_rejects_bad_cardinality(self):
        bad = json.loads(json.dumps(SAMPLE_DECOMP))
        bad["concept_groups"][0]["cardinality"] = "khoảng 3"
        errors = s1a.validate_decomposition(bad, SAMPLE_TARGET)
        self.assertTrue(any("cardinality" in e for e in errors))

    def test_rejects_missing_facts_for_element(self):
        bad = json.loads(json.dumps(SAMPLE_DECOMP))
        del bad["facts"]["3"]
        errors = s1a.validate_decomposition(bad, SAMPLE_TARGET)
        self.assertTrue(any("facts['3']" in e for e in errors))

    def test_rejects_facts_without_rank_zero(self):
        bad = json.loads(json.dumps(SAMPLE_DECOMP))
        bad["facts"]["3"] = [{"rank": 1, "kind": "color", "text": "xám"}]
        errors = s1a.validate_decomposition(bad, SAMPLE_TARGET)
        self.assertTrue(any("rank 0" in e for e in errors))

    def test_normalize_reindexes_ranks_and_fixes_kind(self):
        messy = {
            "concept_groups": [{"name": "g", "member_ids": [1, 0],
                                "cardinality": "exact:2", "cultural": False}],
            "facts": {"0": [{"rank": 7, "kind": "weird", "text": "b"},
                            {"rank": 2, "kind": "subject", "text": "a"}]},
        }
        out = s1a.normalize_decomposition(messy)
        self.assertEqual([f["rank"] for f in out["facts"]["0"]], [0, 1])
        self.assertEqual(out["facts"]["0"][0]["text"], "a")   # rank 2 đứng trước rank 7
        self.assertEqual(out["facts"]["0"][1]["kind"], "attribute")  # kind lạ -> attribute
        self.assertEqual(out["concept_groups"][0]["member_ids"], [0, 1])  # đã sắp xếp

    def test_rejects_missing_background_facts(self):
        bad = json.loads(json.dumps(SAMPLE_DECOMP))
        del bad["background_facts"]
        errors = s1a.validate_decomposition(bad, SAMPLE_TARGET)
        self.assertTrue(any("background_facts" in e for e in errors))

    def test_rejects_background_facts_without_rank_zero(self):
        bad = json.loads(json.dumps(SAMPLE_DECOMP))
        bad["background_facts"] = [{"rank": 1, "kind": "color", "text": "xanh"}]
        errors = s1a.validate_decomposition(bad, SAMPLE_TARGET)
        self.assertTrue(any("background_facts" in e and "rank 0" in e for e in errors))

    def test_rejects_missing_style_field(self):
        bad = json.loads(json.dumps(SAMPLE_DECOMP))
        del bad["style_facts"]["lighting"]
        errors = s1a.validate_decomposition(bad, SAMPLE_TARGET)
        self.assertTrue(any("style_facts['lighting']" in e for e in errors))

    def test_rejects_style_field_not_in_target(self):
        """Ảnh chụp (có `photo`) mà model lại trả `art_style` -> phải bắt được nhầm lẫn."""
        bad = json.loads(json.dumps(SAMPLE_DECOMP))
        bad["style_facts"]["art_style"] = [{"rank": 0, "text": "tranh khắc gỗ"}]
        errors = s1a.validate_decomposition(bad, SAMPLE_TARGET)
        self.assertTrue(any("art_style" in e and "không có trong đầu vào" in e for e in errors))

    def test_rejects_decomposing_medium(self):
        bad = json.loads(json.dumps(SAMPLE_DECOMP))
        bad["style_facts"]["medium"] = [{"rank": 0, "text": "ảnh chụp"}]
        errors = s1a.validate_decomposition(bad, SAMPLE_TARGET)
        self.assertTrue(any("medium" in e for e in errors))

    def test_art_style_target_expects_art_style_not_photo(self):
        target = json.loads(json.dumps(SAMPLE_TARGET))
        target["style_description"] = {
            "aesthetics": "folk art, flat", "lighting": "flat even illumination",
            "medium": "illustration", "art_style": "woodblock print style, bold black outlines",
        }
        good = json.loads(json.dumps(SAMPLE_DECOMP))
        good["style_facts"] = {
            "art_style": [{"rank": 0, "text": "tranh khắc gỗ"}, {"rank": 1, "text": "nét viền đen đậm"}],
            "lighting": [{"rank": 0, "text": "ánh sáng đều"}],
            "aesthetics": [{"rank": 0, "text": "nghệ thuật dân gian"}, {"rank": 1, "text": "phẳng"}],
        }
        self.assertEqual(s1a.validate_decomposition(good, target), [])

    def test_normalize_reindexes_background_and_style_ranks(self):
        messy = {
            "concept_groups": [], "facts": {},
            "background_facts": [{"rank": 5, "kind": "weird", "text": "b"},
                                 {"rank": 2, "kind": "subject", "text": "a"}],
            "style_facts": {"lighting": [{"rank": 3, "text": "y"}, {"rank": 1, "text": "x"}],
                            "medium": [{"rank": 0, "text": "ảnh chụp"}]},
        }
        out = s1a.normalize_decomposition(messy)
        self.assertEqual([f["text"] for f in out["background_facts"]], ["a", "b"])
        self.assertEqual([f["rank"] for f in out["background_facts"]], [0, 1])
        self.assertEqual(out["background_facts"][1]["kind"], "attribute")
        self.assertEqual([f["text"] for f in out["style_facts"]["lighting"]], ["x", "y"])
        self.assertNotIn("kind", out["style_facts"]["lighting"][0])
        self.assertNotIn("medium", out["style_facts"])

    def test_normalize_sets_every_field_priority_to_one(self):
        out = s1a.normalize_decomposition(SAMPLE_DECOMP)
        self.assertEqual(out["field_priority"], {
            "elements": 1, "background": 1, "photo": 1, "lighting": 1, "aesthetics": 1,
        })

    def test_cache_from_old_schema_is_stale(self):
        """Cache sinh trước khi có background_facts/style_facts không được dùng lẫn."""
        with tempfile.TemporaryDirectory() as d:
            old = Path(d) / "old.json"
            new = Path(d) / "new.json"
            io_utils.write_json(old, {"concept_groups": [], "facts": {}})
            io_utils.write_json(new, s1a.normalize_decomposition(SAMPLE_DECOMP))
            self.assertFalse(s1a.is_cache_current(old))
            self.assertTrue(s1a.is_cache_current(new))

    def test_system_prompt_covers_new_fields_and_skips_medium(self):
        text = prompts.STEP1A_SYSTEM
        for key in ("background_facts", "style_facts", '"photo"', '"art_style"',
                    '"lighting"', '"aesthetics"'):
            self.assertIn(key, text)
        self.assertIn('KHÔNG phân rã "medium"', text)
        self.assertIn("KHÔNG bỏ bớt thông tin", text)


# =============================================================================
# STEP 1b — LLM chọn lọc short / medium / long
# =============================================================================

GOOD_SELECTION = {
    "short": {
        "groups": [{"name": "nhóm bạn", "facts": ["nam thanh niên", "cô gái", "số lượng: 3"]},
                   {"name": "Lăng Bác", "facts": []}],
        "background": [], "style": {}, "medium": False,
    },
    "medium": {
        "groups": [{"name": "nhóm bạn",
                    "facts": ["nam thanh niên", "cô gái", "số lượng: 3", "mặc áo dài trắng"]},
                   {"name": "Lăng Bác", "facts": ["bằng đá granite xám"]}],
        "background": ["quảng trường Ba Đình"], "style": {}, "medium": False,
    },
    "long": {
        "groups": [{"name": "nhóm bạn",
                    "facts": ["nam thanh niên", "cô gái", "số lượng: 3", "mặc áo dài trắng",
                              "áo sơ mi trắng", "áo phông đen", "đứng bên trái", "đứng giữa",
                              "đứng bên phải"]},
                   {"name": "Lăng Bác", "facts": ["bằng đá granite xám"]},
                   {"name": "cảnh nền", "facts": ["hàng cây xanh", "hai bên quảng trường"]}],
        "background": ["quảng trường Ba Đình", "rộng", "bầu trời trong xanh", "vài đám mây trắng"],
        "style": {"photo": ["toàn cảnh"], "lighting": ["nắng sớm"]},
        "medium": False,
    },
}


def _copy(obj):
    return json.loads(json.dumps(obj))


class TestSubJsonSelection(unittest.TestCase):
    """STEP 1b: LLM chọn, code chỉ dựng đầu vào, KIỂM TRA lựa chọn và lắp sub_json."""

    def sel_input(self, row_id="img1", decomp=None):
        decomp = decomp or SAMPLE_DECOMP
        index = subjson.find_required_group_index(decomp["concept_groups"], row_id)
        return subjson.build_selection_input(SAMPLE_TARGET, decomp, index)

    def errors(self, selection, row_id="img1"):
        return subjson.validate_selection(selection, self.sel_input(row_id))

    # --- dựng đầu vào -------------------------------------------------------

    def test_input_keeps_every_fact_background_and_style(self):
        inp = self.sel_input()
        by_name = {g["name"]: g for g in inp["groups"]}
        self.assertEqual(len(by_name["nhóm bạn"]["elements"]), 3)
        self.assertEqual(inp["background"],
                         [f["text"] for f in SAMPLE_DECOMP["background_facts"]])
        self.assertEqual(set(inp["style"]), {"photo", "lighting", "aesthetics"})
        self.assertEqual(inp["medium"], "photograph")

    def test_count_fact_only_for_exact_two_or_more(self):
        self.assertEqual(subjson.count_fact("exact:3"), "số lượng: 3")
        self.assertIsNone(subjson.count_fact("exact:1"))
        self.assertIsNone(subjson.count_fact("vague"))
        by_name = {g["name"]: g for g in self.sel_input()["groups"]}
        self.assertEqual(by_name["nhóm bạn"].get("so_luong"), "số lượng: 3")
        self.assertNotIn("so_luong", by_name["Lăng Bác"])

    def test_orphan_element_becomes_its_own_group(self):
        decomp = _copy(SAMPLE_DECOMP)
        decomp["concept_groups"] = [g for g in decomp["concept_groups"] if g["name"] != "cảnh nền"]
        names = [g["name"] for g in self.sel_input(decomp=decomp)["groups"]]
        self.assertIn("hàng cây xanh", names)

    def test_duplicate_group_names_are_disambiguated(self):
        decomp = _copy(SAMPLE_DECOMP)
        decomp["concept_groups"][2]["name"] = "nhóm bạn"
        names = [g["name"] for g in self.sel_input(decomp=decomp)["groups"]]
        self.assertEqual(names.count("nhóm bạn"), 1)
        self.assertIn("nhóm bạn [2]", names)
        self.assertEqual(subjson.original_name("nhóm bạn [2]"), "nhóm bạn")

    def test_required_subject_from_filename(self):
        self.assertEqual(self.sel_input("lang_bac_000123")["required_subject"], "Lăng Bác")
        self.assertIsNone(self.sel_input("img1")["required_subject"])

    # --- kiểm tra lựa chọn ----------------------------------------------------

    def test_accepts_good_selection(self):
        self.assertEqual(self.errors(GOOD_SELECTION), [])

    def test_rejects_rewritten_fact(self):
        bad = _copy(GOOD_SELECTION)
        bad["long"]["groups"][0]["facts"].append("cô gái mặc áo dài")
        self.assertTrue(any("nguyên văn" in e for e in self.errors(bad)))

    def test_rejects_unknown_group(self):
        bad = _copy(GOOD_SELECTION)
        bad["long"]["groups"].append({"name": "con chó", "facts": []})
        self.assertTrue(any("không có trong đầu vào" in e for e in self.errors(bad)))

    def test_rejects_missing_level(self):
        bad = _copy(GOOD_SELECTION)
        del bad["medium"]
        self.assertTrue(any("thiếu mức 'medium'" in e for e in self.errors(bad)))

    def test_rejects_nesting_violation(self):
        bad = _copy(GOOD_SELECTION)
        bad["medium"]["groups"][0]["facts"].remove("cô gái")
        self.assertTrue(any("lồng nhau" in e for e in self.errors(bad)))

    def test_rejects_short_with_style(self):
        bad = _copy(GOOD_SELECTION)
        bad["short"]["style"] = {"photo": ["toàn cảnh"]}
        self.assertTrue(any("short: không được chọn 'style'" in e for e in self.errors(bad)))

    def test_rejects_short_with_too_many_ideas(self):
        bad = _copy(GOOD_SELECTION)
        bad["short"]["groups"][0]["facts"].append("mặc áo dài trắng")
        self.assertTrue(any("ý ngoài chủ thể chính" in e for e in self.errors(bad)))

    def test_short_may_take_one_location_from_background(self):
        good = _copy(GOOD_SELECTION)
        good["short"] = {"groups": [{"name": "nhóm bạn", "facts": ["nam thanh niên", "cô gái"]}],
                         "background": ["quảng trường Ba Đình"], "style": {}, "medium": False}
        self.assertEqual(self.errors(good), [])

    def test_rejects_word_cap_exceeded(self):
        with mock.patch.dict(subjson.ACCEPT_CHECKLIST_WORDS, {"short": 5}):
            self.assertTrue(any("vượt trần" in e for e in self.errors(GOOD_SELECTION)))

    def test_accepts_small_overshoot_of_prompt_cap(self):
        # Vượt trần trong prompt nhưng vẫn dưới ngưỡng chấp nhận -> hợp lệ.
        with mock.patch.dict(subjson.MAX_CHECKLIST_WORDS, {"short": 1, "medium": 1}):
            self.assertEqual(self.errors(GOOD_SELECTION), [])

    def test_accept_threshold_is_looser_than_prompt_cap(self):
        for level in ("short", "medium"):
            self.assertGreater(subjson.ACCEPT_CHECKLIST_WORDS[level], subjson.MAX_CHECKLIST_WORDS[level])
        self.assertEqual(subjson.ACCEPT_CHECKLIST_WORDS, {"short": 20, "medium": 55, "long": None})

    def test_rejects_missing_required_subject(self):
        bad = _copy(GOOD_SELECTION)
        bad["short"]["groups"] = [bad["short"]["groups"][0]]
        errors = self.errors(bad, row_id="lang_bac_000123")
        self.assertTrue(any("short: thiếu nhóm chủ thể bắt buộc" in e for e in errors))

    def test_rejects_label_group_without_facts(self):
        bad = _copy(GOOD_SELECTION)
        bad["long"]["groups"][2]["facts"] = []
        self.assertTrue(any("không có mệnh đề nào" in e for e in self.errors(bad)))

    # --- lắp sub_json ---------------------------------------------------------

    def assemble(self, level, row_id="img1", selection=None):
        return subjson.assemble_subjson(row_id, self.sel_input(row_id),
                                        selection or GOOD_SELECTION, level)

    def test_checklist_is_exactly_what_2a_is_given(self):
        persona = {"vai": "v", "giọng": "g", "ngôn ngữ": "n"}
        for level in subjson.LEVELS:
            sub = self.assemble(level)
            spec = s2a.build_spec(sub, persona)
            given = [f for g in spec["groups"] for f in g["facts"]]
            given += spec.get("boi_canh", []) + spec.get("phong_cach", [])
            self.assertEqual(given, sub["checklist"], level)

    def test_cultural_name_is_checklist_item_but_label_is_not(self):
        checklist = self.assemble("long")["checklist"]
        self.assertIn("Lăng Bác", checklist)
        self.assertNotIn("cảnh nền", checklist)
        self.assertNotIn("nhóm bạn", checklist)

    def test_so_nhieu_only_when_plural_without_count(self):
        groups = {g["name"]: g for g in self.assemble("short")["groups"]}
        self.assertFalse(groups["nhóm bạn"]["so_nhieu"])
        self.assertFalse(groups["Lăng Bác"]["so_nhieu"])
        no_count = _copy(GOOD_SELECTION)
        for level in subjson.LEVELS:
            no_count[level]["groups"][0]["facts"].remove("số lượng: 3")
        groups = {g["name"]: g for g in self.assemble("short", selection=no_count)["groups"]}
        self.assertTrue(groups["nhóm bạn"]["so_nhieu"])

    def test_required_subject_points_to_checklist_item(self):
        sub = self.assemble("short", row_id="lang_bac_000123")
        self.assertEqual(sub["required_subject"], "Lăng Bác")
        self.assertIn(sub["required_subject"], sub["checklist"])

    def test_length_hint_per_level(self):
        self.assertEqual(self.assemble("short")["length_hint"], "8-20 từ")
        self.assertEqual(self.assemble("long")["length_hint"], "100-200 từ")

    # --- cache ---------------------------------------------------------------

    def test_fingerprint_changes_when_input_changes(self):
        inp = self.sel_input()
        changed = _copy(inp)
        changed["background"].append("thêm")
        self.assertNotEqual(s1b.fingerprint(inp), s1b.fingerprint(changed))
        self.assertEqual(s1b.fingerprint(inp), s1b.fingerprint(_copy(inp)))

    def test_cached_selection_rejects_stale_fingerprint(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.json"
            io_utils.write_json(path, {"fingerprint": "abc", "selection": GOOD_SELECTION})
            self.assertIsNone(s1b.cached_selection(path, "khác"))
            self.assertEqual(s1b.cached_selection(path, "abc"), GOOD_SELECTION)


# =============================================================================
# STEP 2a — persona & làm sạch text
# =============================================================================

class TestVerbalize(unittest.TestCase):

    def test_persona_has_exactly_three_axes(self):
        """Trục 'độ cẩu thả' đã bị bỏ — chính tả sai thuộc mô-đun correction riêng."""
        persona = s2a.sample_persona(random.Random(1), 0.2)
        self.assertEqual(set(persona), {"vai", "giọng", "ngôn ngữ"})

    def test_persona_never_mentions_sloppiness(self):
        for i in range(50):
            blob = json.dumps(s2a.sample_persona(random.Random(i), 0.5), ensure_ascii=False)
            self.assertNotIn("cẩu thả", blob)
            self.assertNotIn("chính tả", blob)

    def test_vai_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(w for _, w in s2a.VAI_WEIGHTED), 1.0, places=6)

    def test_user_pho_thong_is_the_majority_role(self):
        rng = random.Random(0)
        n = 5000
        counts = {}
        for _ in range(n):
            vai = s2a.sample_vai(rng)
            counts[vai] = counts.get(vai, 0) + 1
        share = counts["user phổ thông"] / n
        self.assertAlmostEqual(share, 0.60, delta=0.03)
        # và nó phải áp đảo mọi vai còn lại
        for vai, count in counts.items():
            if vai != "user phổ thông":
                self.assertLess(count, counts["user phổ thông"])

    def test_every_role_still_reachable(self):
        rng = random.Random(7)
        seen = {s2a.sample_vai(rng) for _ in range(5000)}
        self.assertEqual(seen, set(s2a.VAI))

    def test_english_rate_zero_and_one(self):
        for i in range(30):
            self.assertEqual(s2a.sample_persona(random.Random(i), 0.0)["ngôn ngữ"],
                             s2a.NGON_NGU_VN)
            self.assertEqual(s2a.sample_persona(random.Random(i), 1.0)["ngôn ngữ"],
                             s2a.NGON_NGU_EN)

    def test_english_rate_is_roughly_respected(self):
        rng = random.Random(0)
        n = 4000
        english = sum(
            1 for _ in range(n)
            if s2a.sample_persona(rng, 0.2)["ngôn ngữ"] == s2a.NGON_NGU_EN
        )
        self.assertAlmostEqual(english / n, 0.2, delta=0.03)

    def test_clean_prompt_strips_quotes_and_newlines(self):
        self.assertEqual(s2a.clean_prompt_text('"Ảnh chụp\n  Lăng Bác"'), "Ảnh chụp Lăng Bác")
        self.assertEqual(s2a.clean_prompt_text("“Ảnh đẹp”"), "Ảnh đẹp")

    def test_spec_only_exposes_subjson_content(self):
        sel_input = subjson.build_selection_input(SAMPLE_TARGET, SAMPLE_DECOMP)
        sub = subjson.assemble_subjson("i", sel_input, GOOD_SELECTION, "long")
        spec = s2a.build_spec(sub, {"vai": "v", "giọng": "g", "ngôn ngữ": "n"})
        blob = json.dumps(spec, ensure_ascii=False)
        # Không rò rỉ target_json, cũng không rò rỉ nhãn nhóm (2b sẽ chấm là thêm tin)
        self.assertNotIn("target_json", blob)
        self.assertNotIn("cảnh nền", blob)
        self.assertEqual(len(spec["groups"]), len(sub["groups"]))


# =============================================================================
# STEP 2b — quy tắc đạt/loại
# =============================================================================

class _Args:
    max_missing = 0


class TestCorrection(unittest.TestCase):
    """STEP 2b2 -- sửa lại (retry 1 lần) prompt bị 2b loại, thay vì bỏ trắng."""

    def test_step2b2_messages_carry_old_prompt_and_reasons(self):
        messages = prompts.build_step2b2_messages(
            ["áo dài", "màu đỏ"], ["màu đỏ"], ["cái nón"], "một người mặc áo dài",
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["prompt_cu"], "một người mặc áo dài")
        self.assertEqual(payload["thieu"], ["màu đỏ"])
        self.assertEqual(payload["thua"], ["cái nón"])
        self.assertEqual(payload["checklist"], ["áo dài", "màu đỏ"])

    def test_step2b2_system_only_asks_to_fix_stated_issues(self):
        text = prompts.STEP2B2_SYSTEM
        self.assertIn("CHỈ sửa đúng phần bị nêu lỗi", text)

    def test_clean_strips_wrapping_quotes(self):
        self.assertEqual(s2b2.step2b2_clean('"một câu đã sửa"'), "một câu đã sửa")

    def test_clean_collapses_newlines(self):
        self.assertEqual(s2b2.step2b2_clean("dòng một\ndòng hai"), "dòng một dòng hai")

    def test_recovered_row_keeps_original_prompt_for_audit(self):
        """corrected_passed phải giữ lại prompt gốc để audit -- không ghi đè mất dấu vết."""
        row = {"id": "x", "detail_level": "short", "user_prompt": "cũ",
              "checklist": [], "required_subject": None}
        attempt_prompt = "mới đã sửa"
        merged = {**row, "user_prompt": attempt_prompt,
                 "n_words": len(attempt_prompt.split()), "corrected": True,
                 "original_prompt": row["user_prompt"]}
        self.assertEqual(merged["original_prompt"], "cũ")
        self.assertEqual(merged["user_prompt"], "mới đã sửa")
        self.assertTrue(merged["corrected"])


class TestFilterDecision(unittest.TestCase):

    def test_clean_verdict_passes(self):
        out = s2b.decide({"missing": [], "extra": []}, _Args())
        self.assertTrue(out["passed"])

    def test_missing_fails(self):
        out = s2b.decide({"missing": ["áo dài"], "extra": []}, _Args())
        self.assertFalse(out["passed"])

    def test_extra_fails(self):
        out = s2b.decide({"missing": [], "extra": ["con mèo"]}, _Args())
        self.assertFalse(out["passed"])

    def test_robotic_field_is_ignored_if_present(self):
        """Văn phong không phải tiêu chí loại bỏ ở bước 2b — dù verdict có 'robotic' hay không."""
        verdict = {"missing": [], "extra": [], "robotic": True}
        self.assertTrue(s2b.decide(verdict, _Args())["passed"])
        self.assertNotIn("robotic", s2b.decide(verdict, _Args()))

    def test_missing_tolerance_is_configurable(self):
        class Tolerant(_Args):
            max_missing = 1

        verdict = {"missing": ["x"], "extra": []}
        self.assertTrue(s2b.decide(verdict, Tolerant())["passed"])

    def test_handles_absent_keys(self):
        self.assertTrue(s2b.decide({}, _Args())["passed"])

    def test_required_subject_missing_fails_even_with_max_missing_tolerance(self):
        """Chốt chặn cho ý user: chủ thể chính (rút từ tên file, vd 'hủ tiếu Nam
        Vang') phải BẮT BUỘC tuyệt đối -- thiếu nó thì loại ngay dù --max_missing
        đang cho phép bỏ qua các mệnh đề khác."""
        class Tolerant(_Args):
            max_missing = 5

        verdict = {"missing": ["hủ tiếu Nam Vang"], "extra": []}
        out = s2b.decide(verdict, Tolerant(), required_subject="hủ tiếu Nam Vang")
        self.assertFalse(out["passed"])

    def test_required_subject_present_does_not_block_pass(self):
        verdict = {"missing": [], "extra": []}
        out = s2b.decide(verdict, _Args(), required_subject="hủ tiếu Nam Vang")
        self.assertTrue(out["passed"])

    def test_no_required_subject_falls_back_to_normal_tolerance(self):
        class Tolerant(_Args):
            max_missing = 1

        verdict = {"missing": ["màu xanh"], "extra": []}
        out = s2b.decide(verdict, Tolerant(), required_subject=None)
        self.assertTrue(out["passed"])

    def test_ignore_judge_verdict_shape_always_passes(self):
        """--ignore_judge ghi verdict rỗng {"missing": [], "extra": [], "ignored": True}
        cho MỌI dòng thay vì gọi model -- decide() phải luôn trả về đạt với verdict
        này, kể cả khi có required_subject (không check gì cả theo đúng yêu cầu)."""
        verdict = {"missing": [], "extra": [], "ignored": True}
        out = s2b.decide(verdict, _Args(), required_subject="hủ tiếu Nam Vang")
        self.assertTrue(out["passed"])

    def test_ignore_judge_flag_exists_and_defaults_false(self):
        with mock.patch.object(sys, "argv", ["step2b_filter.py", "--out_root", "x"]):
            self.assertFalse(s2b.parse_args().ignore_judge)
        with mock.patch.object(sys, "argv",
                               ["step2b_filter.py", "--out_root", "x", "--ignore_judge"]):
            self.assertTrue(s2b.parse_args().ignore_judge)


# =============================================================================
# LLM client — test tất cả những gì test được mà không gọi mạng
# =============================================================================

class TestLLMClientOffline(unittest.TestCase):

    def test_endpoint_built_from_base_url_variants(self):
        cases = [
            ("http://h:30080", "http://h:30080/v1/chat/completions"),
            ("http://h:30080/", "http://h:30080/v1/chat/completions"),
            ("http://h:30080/v1", "http://h:30080/v1/chat/completions"),
            ("http://h:30080/v1/chat/completions", "http://h:30080/v1/chat/completions"),
        ]
        for base, expected in cases:
            client = llm.LLMClient(base_url=base, api_key="k", model="m")
            self.assertEqual(client.endpoint, expected, base)

    def test_payload_matches_curl_shape(self):
        payload = llm.build_payload(
            "DeepSeek-V4-Flash", [{"role": "user", "content": "Hello!"}], temperature=0.0,
        )
        self.assertEqual(payload["model"], "DeepSeek-V4-Flash")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "Hello!"}])
        # Phải serialise được, vì nó sẽ đi thẳng vào body JSON
        json.dumps(payload)

    def test_payload_json_mode_and_max_tokens(self):
        payload = llm.build_payload("m", [], json_mode=True, max_tokens=128)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["max_tokens"], 128)

    def test_headers_use_bearer(self):
        headers = llm.build_headers("abc")
        self.assertEqual(headers["Authorization"], "Bearer abc")
        self.assertEqual(headers["Content-Type"], "application/json")

    def test_extract_content(self):
        resp = {"choices": [{"message": {"role": "assistant", "content": "xin chào"}}]}
        self.assertEqual(llm.extract_content(resp), "xin chào")

    def test_extract_content_raises_clear_error(self):
        with self.assertRaises(llm.LLMError):
            llm.extract_content({"choices": []})
        with self.assertRaises(llm.LLMError):
            llm.extract_content({"error": "boom"})

    def test_parse_json_handles_code_fence(self):
        self.assertEqual(llm.parse_json_response('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(llm.parse_json_response('```\n{"a": 1}\n```'), {"a": 1})

    def test_parse_json_handles_surrounding_prose(self):
        self.assertEqual(llm.parse_json_response('Đây là kết quả: {"a": 1} xong.'), {"a": 1})

    def test_parse_json_preserves_vietnamese(self):
        out = llm.parse_json_response('{"text": "áo dài"}')
        self.assertEqual(out["text"], "áo dài")

    def test_run_parallel_keeps_order_and_survives_errors(self):
        def fn(x):
            if x == 3:
                raise ValueError("hỏng")
            return x * 2

        errors = []
        out = llm.run_parallel([1, 2, 3, 4], fn, workers=4,
                               on_error=lambda item, exc: errors.append(item))
        self.assertEqual(out, [2, 4, None, 8])
        self.assertEqual(errors, [3])

    def test_run_parallel_on_empty_list(self):
        self.assertEqual(llm.run_parallel([], lambda x: x), [])


# =============================================================================
# Prompt — dựng messages đúng hình dạng
# =============================================================================

class TestPrompts(unittest.TestCase):

    def test_step1a_messages_have_fewshot(self):
        messages = prompts.build_step1a_messages(SAMPLE_TARGET)
        self.assertEqual([m["role"] for m in messages],
                         ["system", "user", "assistant", "user"])
        self.assertIn("cardinality", messages[0]["content"])
        # Few-shot phải dạy đúng luật vague
        self.assertIn("vague", messages[2]["content"])

    def test_step1a_fewshot_output_is_valid_against_its_input(self):
        """Ví dụ few-shot phải tự vượt qua chính bộ validate của step 1a."""
        errors = s1a.validate_decomposition(
            prompts.STEP1A_FEWSHOT_OUTPUT, prompts.STEP1A_FEWSHOT_INPUT,
        )
        self.assertEqual(errors, [])

    def test_step1b_fewshot_output_is_valid_against_its_input(self):
        """Ví dụ few-shot của 1b phải tự vượt qua chính bộ kiểm tra lựa chọn."""
        self.assertEqual(subjson.validate_selection(
            prompts.STEP1B_FEWSHOT_OUTPUT, prompts.STEP1B_FEWSHOT_INPUT), [])

    def test_step1b_messages_have_fewshot(self):
        messages = prompts.build_step1b_messages(prompts.STEP1B_FEWSHOT_INPUT)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])

    def test_step1b_system_defines_levels_by_content_not_length(self):
        text = prompts.STEP1B_SYSTEM
        for phrase in ("ĐỘ PHỦ", "ĐỘ SÂU", "LOẠI THÔNG TIN", "SHORT", "MEDIUM", "LONG",
                       "NGUYÊN VĂN", "LỒNG NHAU", "CÓ BẢN SẮC", "required_subject"):
            self.assertIn(phrase, text)

    def test_step1b_fix_message_lists_every_error(self):
        message = prompts.build_step1b_fix_message(["lỗi A", "lỗi B"])
        self.assertIn("lỗi A", message)
        self.assertIn("lỗi B", message)

    def test_step2a_messages_alternate_roles(self):
        spec = {"detail_level": "short", "length_hint": "8-20 từ",
                "persona": {}, "groups": []}
        messages = prompts.build_step2a_messages(spec)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1]["role"], "user")
        for i in range(1, len(messages) - 1, 2):
            self.assertEqual(messages[i]["role"], "user")
            self.assertEqual(messages[i + 1]["role"], "assistant")

    def test_step2a_system_bans_the_right_things(self):
        text = prompts.STEP2A_SYSTEM
        for banned in ["liệt kê", "góc trên bên trái", "văn dịch", "thứ tự"]:
            self.assertIn(banned, text)

    def test_step2a_system_requires_full_coverage_of_given_facts(self):
        """Chốt chặn cho lỗ hổng thật: STEP2A_SYSTEM chỉ cấm THÊM tin, chưa từng nói
        rõ phải nhắc ĐỦ mọi mệnh đề được cho -- model có thể tự ý bỏ bớt ánh sáng/
        bối cảnh mà không phạm luật nào, khớp với hiện tượng anh_sang/boi_canh vẫn
        thiếu ngay cả ở mức long (dư từ, không phải do hết ngân sách)."""
        text = prompts.STEP2A_SYSTEM
        self.assertIn("ĐẦY ĐỦ", text)
        self.assertIn("bỏ sót", text)

    def test_step2a_fewshot_examples_fully_cover_their_own_facts(self):
        """Few-shot phải LÀM MẪU đúng luật 'nhắc đủ mọi mệnh đề' -- nếu ví dụ tự mâu thuẫn
        với luật thì model học sai theo ví dụ, bất kể system prompt viết gì."""
        stopwords = {"từ", "trên", "trong", "chụp", "và", "mặt", "là", "những", "chiếc",
                     "màu", "bằng", "đang", "một"}
        for spec, answer in prompts.STEP2A_FEWSHOT:
            items = [f for g in spec["groups"] for f in g["facts"]]
            items += spec.get("boi_canh", []) + spec.get("phong_cach", [])
            answer_low = answer.lower()
            for item in items:
                if item.startswith("số lượng"):
                    continue
                for kw in [w for w in item.lower().split() if w not in stopwords]:
                    self.assertIn(kw, answer_low, "few-shot {}: không nhắc '{}' (từ '{}')".format(
                        spec["detail_level"], kw, item))

    def test_step2a_fewshot_specs_use_build_spec_shape(self):
        allowed = {"detail_level", "length_hint", "persona", "groups", "boi_canh", "phong_cach"}
        for spec, _ in prompts.STEP2A_FEWSHOT:
            self.assertLessEqual(set(spec), allowed)
            for group in spec["groups"]:
                self.assertLessEqual(set(group), {"facts", "so_nhieu"})

    def test_step2a_fewshot_lengths_match_their_level(self):
        bounds = {"short": (5, 25), "medium": (25, 70), "long": (55, 220)}
        for spec, answer in prompts.STEP2A_FEWSHOT:
            lo, hi = bounds[spec["detail_level"]]
            n = len(answer.split())
            self.assertTrue(lo <= n <= hi,
                            "few-shot {} dài {} từ, ngoài khoảng {}-{}".format(
                                spec["detail_level"], n, lo, hi))

    def test_step2a_system_requires_correct_spelling(self):
        self.assertIn("ĐÚNG CHÍNH TẢ", prompts.STEP2A_SYSTEM)

    def test_step2a_fewshot_answers_are_properly_accented(self):
        """Không còn ví dụ mất dấu: dữ liệu nhiễu thuộc mô-đun correction riêng."""
        accented = set("ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
        for spec, answer in prompts.STEP2A_FEWSHOT:
            if "tiếng Việt" not in spec["persona"].get("ngôn ngữ", "tiếng Việt"):
                continue
            self.assertTrue(accented & set(answer.lower()),
                            "few-shot thiếu dấu tiếng Việt: {}".format(answer))

    def test_step2a_fewshot_personas_have_no_sloppiness_axis(self):
        for spec, _ in prompts.STEP2A_FEWSHOT:
            self.assertNotIn("độ cẩu thả", spec["persona"])

    def test_step2b_messages_embed_checklist_and_prompt(self):
        messages = prompts.build_step2b_messages(["áo dài đỏ"], "ảnh áo dài")
        self.assertIn("áo dài đỏ", messages[-1]["content"])
        self.assertIn("ảnh áo dài", messages[-1]["content"])

    def test_step2b_tolerates_missing_subject_repeat_in_position_facts(self):
        """Chốt chặn cho hướng dẫn khoan dung: mệnh đề vị trí lặp tên chủ thể
        ('ở tai phải thanh đồng') không bắt buộc prompt phải lặp lại tên đó."""
        text = prompts.STEP2B_SYSTEM
        self.assertIn("KHÔNG lặp", text)
        self.assertIn("KHÔNG được tính là thiếu", text)

    def test_step2b_still_strict_about_actual_position_value(self):
        """Nhưng vị trí CỤ THỂ (trái/phải...) sai thì vẫn phải bắt được, không
        được khoan dung tới mức bỏ qua luôn cả nội dung vị trí."""
        text = prompts.STEP2B_SYSTEM
        self.assertIn("tai trái", text)   # ví dụ minh hoạ trường hợp sai vị trí
        self.assertIn("SAI vị trí", text)
        self.assertIn("vẫn tính là THIẾU", text)

    def test_step2b_language_agnostic_except_cultural_terms(self):
        text = prompts.STEP2B_SYSTEM
        self.assertIn("Ngôn ngữ không quan trọng", text)
        self.assertIn("NGOẠI LỆ DUY NHẤT", text)
        self.assertIn("thuật ngữ văn hoá Việt Nam", text)


# =============================================================================
# Config & io_utils
# =============================================================================

class TestConfig(unittest.TestCase):

    def test_parses_quotes_comments_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                '# comment\n'
                'export LLM_BASE_URL="http://x:1"\n'
                "LLM_MODEL='m1'\n"
                "\n"
                "EMPTY=\n",
                encoding="utf-8",
            )
            parsed = config._parse_env_file(path)
        self.assertEqual(parsed["LLM_BASE_URL"], "http://x:1")
        self.assertEqual(parsed["LLM_MODEL"], "m1")
        self.assertEqual(parsed["EMPTY"], "")

    def test_require_raises_helpful_message(self):
        os.environ.pop("DEFINITELY_NOT_SET_VAR", None)
        with self.assertRaises(RuntimeError) as ctx:
            config.require("DEFINITELY_NOT_SET_VAR")
        self.assertIn(".env", str(ctx.exception))

    def test_no_endpoint_hardcoded_in_source(self):
        """Bảo hiểm: không được lỡ tay commit endpoint nội bộ vào code.

        Ghép chuỗi cần tìm từ mảnh để chính file test này không khớp với nó.
        """
        needles = ["viettel" + "ai", "llm-" + "h200", "deepseek-v4-" + "datagen"]
        root = Path(__file__).resolve().parent.parent
        scanned = 0
        for path in root.rglob("*.py"):
            if "tests" in path.parts:
                continue
            text = path.read_text(encoding="utf-8").lower()
            scanned += 1
            for needle in needles:
                self.assertNotIn(needle, text, "{} chứa endpoint nội bộ".format(path))
        self.assertGreater(scanned, 5, "không quét được file nào - đường dẫn sai?")

    def test_env_example_has_no_real_secret(self):
        example = Path(__file__).resolve().parent.parent / ".env.example"
        text = example.read_text(encoding="utf-8").lower()
        for needle in ["viettel" + "ai", "llm-" + "h200"]:
            self.assertNotIn(needle, text)
        self.assertIn("LLM_BASE_URL", example.read_text(encoding="utf-8"))

    def test_package_gitignore_blocks_env_and_outputs(self):
        ignore = (Path(__file__).resolve().parent.parent / ".gitignore").read_text(encoding="utf-8")
        # Bí mật + mọi thư mục output, kể cả khi người dùng tự đặt tên qua --out_dir
        for entry in (".env", "output/", "test_*/", "run_*/"):
            self.assertIn(entry, ignore)

    def test_repo_gitignore_blocks_raw_data(self):
        root = Path(__file__).resolve().parents[3] / ".gitignore"
        if not root.is_file():          # repo chưa init thì bỏ qua
            self.skipTest("chưa có .gitignore ở gốc repo")
        ignore = root.read_text(encoding="utf-8")
        self.assertIn("/data/", ignore)
        self.assertIn("**/output/", ignore)


class TestOutRoot(unittest.TestCase):
    """Mọi step ghi vào thư mục con riêng bên trong một out_root chung."""

    def test_every_step_has_its_own_subdir(self):
        names = list(io_utils.STEP_DIRS.values())
        self.assertEqual(len(names), len(set(names)), "tên thư mục step bị trùng")
        self.assertEqual(
            set(io_utils.STEP_DIRS),
            {"step0", "step1a", "step1b", "step2a", "step2b", "step2b2", "step2c", "step2d"})

    def test_step_dir_nests_under_out_root(self):
        self.assertEqual(str(io_utils.step_dir("test_1", "step0")), "test_1/step0_normalized")
        self.assertEqual(str(io_utils.step_dir("test_1", "step2c")), "test_1/step2c_split")

    def test_step_dir_rejects_unknown_step(self):
        with self.assertRaises(KeyError):
            io_utils.step_dir("test_1", "step9")

    def test_resolve_derives_path_from_out_root(self):
        self.assertEqual(str(io_utils.resolve(None, "test_1", "step0", "targets.jsonl")),
                         "test_1/step0_normalized/targets.jsonl")

    def test_resolve_prefers_explicit_path(self):
        self.assertEqual(str(io_utils.resolve("/tuy/chinh.jsonl", "test_1", "step0", "targets.jsonl")),
                         "/tuy/chinh.jsonl")

    def test_steps_chain_through_the_same_out_root(self):
        """Đầu ra step trước phải là đầu vào mặc định của step sau."""
        root = "run_x"
        chain = [("step0", "targets.jsonl", "step1a"),
                 ("step1b", "subjson.jsonl", "step2a"),
                 ("step2a", "prompts.jsonl", "step2b"),
                 ("step2b", "passed.jsonl", "step2c")]
        for producer, filename, consumer in chain:
            produced = io_utils.step_dir(root, producer) / filename
            consumed = io_utils.resolve(None, root, producer, filename)
            self.assertEqual(produced, consumed,
                             "{} -> {} không khớp".format(producer, consumer))


class TestJudgeFallback(unittest.TestCase):
    """Chốt chặn cho cấu hình thực tế: key rỗng + step 2b dùng lại model chính."""

    def _load(self, text):
        tmp = Path(tempfile.mkdtemp()) / ".env"
        tmp.write_text(text, encoding="utf-8")
        for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL",
                    "JUDGE_BASE_URL", "JUDGE_API_KEY", "JUDGE_MODEL"):
            os.environ.pop(key, None)
        config.load_env(str(tmp), force=True)

    def test_empty_judge_config_falls_back_to_main_model(self):
        self._load("LLM_BASE_URL=http://h:30080\nLLM_API_KEY=\nLLM_MODEL=M1\n"
                   "JUDGE_BASE_URL=\nJUDGE_API_KEY=\nJUDGE_MODEL=\n")
        client = s2b.make_client()
        self.assertEqual(client.endpoint, "http://h:30080/v1/chat/completions")
        self.assertEqual(client.model, "M1")

    def test_empty_api_key_is_allowed(self):
        self._load("LLM_BASE_URL=http://h:30080\nLLM_API_KEY=\nLLM_MODEL=M1\n")
        client = s2b.make_client()
        self.assertEqual(client.api_key, "")
        # Khớp đúng lệnh curl: header có mặt nhưng phần token để trống
        self.assertEqual(llm.build_headers(client.api_key)["Authorization"], "Bearer ")

    def test_judge_override_is_used_when_set(self):
        self._load("LLM_BASE_URL=http://a:1\nLLM_MODEL=M1\n"
                   "JUDGE_BASE_URL=http://b:2\nJUDGE_MODEL=M2\n")
        client = s2b.make_client()
        self.assertEqual(client.endpoint, "http://b:2/v1/chat/completions")
        self.assertEqual(client.model, "M2")


class TestIoUtils(unittest.TestCase):

    def test_jsonl_roundtrip_preserves_vietnamese(self):
        rows = [{"id": "a", "t": "áo dài"}, {"id": "b", "t": "khăn xếp"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            io_utils.write_jsonl(path, rows)
            self.assertIn("áo dài", path.read_text(encoding="utf-8"))
            self.assertEqual(io_utils.read_jsonl(path), rows)

    def test_sample_for_test_is_seeded_and_capped(self):
        items = list(range(100))
        a = io_utils.sample_for_test(items, 20, seed=1)
        b = io_utils.sample_for_test(items, 20, seed=1)
        c = io_utils.sample_for_test(items, 20, seed=2)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(a), 20)
        self.assertEqual(len(io_utils.sample_for_test(items, 500)), 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
