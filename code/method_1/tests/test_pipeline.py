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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import config, io_utils, llm, prompts, schema  # noqa: E402
import step1b_build_subjson as s1b  # noqa: E402
import step2a_verbalize as s2a  # noqa: E402
import step2b_filter as s2b  # noqa: E402
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
}


# =============================================================================
# STEP 0 — schema
# =============================================================================

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


# =============================================================================
# STEP 1b — nén 2 trục
# =============================================================================

class TestBuildSubJson(unittest.TestCase):

    def build(self, level, seed="s"):
        rng = random.Random(seed)
        return s1b.build_subjson("img1", SAMPLE_TARGET, SAMPLE_DECOMP, level, rng)

    def test_short_is_shallower_and_narrower_than_long(self):
        short = self.build("short")
        long_ = self.build("long")
        self.assertLessEqual(len(short["groups"]), len(long_["groups"]))
        self.assertLess(len(short["checklist"]), len(long_["checklist"]))

    def test_cultural_group_always_kept_even_in_short(self):
        short = self.build("short")
        names = [g["name"] for g in short["groups"]]
        self.assertIn("Lăng Bác", names)

    def test_short_drops_background_but_long_keeps_it(self):
        self.assertNotIn("boi_canh", self.build("short")["scene"])
        self.assertIn("boi_canh", self.build("long")["scene"])

    def test_catchall_group_with_one_text_element_is_trimmed_to_that_element(self):
        """Chốt chặn cho bug thật (server test_3): 'cảnh nền' gom 13 element không
        liên quan, tình cờ chứa 1 element chữ (biển hiệu) -- giữ nguyên CẢ nhóm chỉ
        vì có chữ khiến short/medium phình to. Phải CHỈ giữ đúng element chữ đó."""
        target = {
            "high_level_description": "Một cảnh chợ.",
            "style_description": {"medium": "photograph", "photo": "wide"},
            "compositional_deconstruction": {
                "background": "Chợ.",
                "elements": (
                    [{"id": 0, "type": "obj", "desc": "Chủ thể chính."}]
                    + [{"id": i, "type": "obj", "desc": "Vật nền {}.".format(i)}
                      for i in range(1, 6)]
                    + [{"id": 6, "type": "text", "text": "ABC", "desc": "Biển hiệu chữ ABC."}]
                ),
            },
        }
        decomp = {
            "concept_groups": [
                {"name": "chủ thể chính", "member_ids": [0], "cardinality": "exact:1", "cultural": False},
                {"name": "cảnh nền", "member_ids": [1, 2, 3, 4, 5, 6], "cardinality": "vague", "cultural": False},
            ],
            "facts": {
                "0": [{"rank": 0, "kind": "subject", "text": "chủ thể chính"}],
                **{str(i): [{"rank": 0, "kind": "subject", "text": "vật nền {}".format(i)},
                           {"rank": 1, "kind": "attribute", "text": "chi tiết {}".format(i)}]
                  for i in range(1, 6)},
                "6": [{"rank": 0, "kind": "subject", "text": "biển hiệu ABC"},
                      {"rank": 1, "kind": "color", "text": "chữ đen"}],
            },
        }
        sub = s1b.build_subjson("x", target, decomp, "short", random.Random("s"))
        bg = next(g for g in sub["groups"] if g["name"] == "cảnh nền")
        self.assertIn("biển hiệu ABC", bg["facts"])
        for i in range(1, 6):
            self.assertNotIn("vật nền {}".format(i), bg["facts"],
                             "kéo theo vật nền không liên quan chỉ vì nhóm có chữ")

    def test_exact_cardinality_emits_count_constraint(self):
        checklist = self.build("long")["checklist"]
        self.assertTrue(any(c.startswith("số lượng: 3") for c in checklist))

    def test_vague_cardinality_never_emits_count_constraint(self):
        # "cảnh nền" là vague — không được sinh ra ràng buộc số lượng cho nó
        for level in ("short", "medium", "long"):
            for c in self.build(level)["checklist"]:
                self.assertNotIn("cảnh nền", c.replace("số lượng: ", "")) if c.startswith(
                    "số lượng:") else None

    def test_collective_noun_group_name_suppresses_count_constraint(self):
        """Chốt chặn cho bug thật (server test_6): tên nhóm là danh từ tập hợp/cặp
        (vd 'đôi đũa', 'đôi bàn tay') đã tự mang nghĩa 'một cặp' -- ghép thêm số n
        (đếm số member nguyên tử) phía trước sẽ ra khẳng định số lượng SAI
        ('đôi đũa' + n=2 -> 'số lượng: 2 đôi đũa' = 4 chiếc, trong khi ảnh chỉ có
        1 đôi/2 chiếc). Không ai đặt hàng ảnh lại nói câu này, và nó còn sai."""
        target = {
            "high_level_description": "x",
            "style_description": {"medium": "photograph", "photo": "wide"},
            "compositional_deconstruction": {"background": "", "elements": [
                {"id": 0, "type": "obj", "desc": "a"},
                {"id": 1, "type": "obj", "desc": "b"},
            ]},
        }
        decomp = {
            "concept_groups": [
                {"name": "đôi đũa", "member_ids": [0, 1],
                 "cardinality": "exact:2", "cultural": True},
            ],
            "facts": {
                "0": [{"rank": 0, "kind": "subject", "text": "chiếc đũa"}],
                "1": [{"rank": 0, "kind": "subject", "text": "chiếc đũa"}],
            },
        }
        sub = s1b.build_subjson("x", target, decomp, "long", random.Random("s"))
        for c in sub["checklist"]:
            self.assertFalse(c.startswith("số lượng:"), c)

    def test_group_with_many_members_is_capped_in_short(self):
        """Chốt chặn cho bug thật (server test_3): nhóm chính nhiều thành viên
        (vd '3 người trên thuyền') từng gộp hết vào short, ra hàng chục mệnh đề
        cho một mức lẽ ra chỉ 8-20 từ."""
        target = {
            "high_level_description": "Năm người bạn.",
            "style_description": {"medium": "photograph", "photo": "wide"},
            "compositional_deconstruction": {
                "background": "Nền.",
                "elements": [{"id": i, "type": "obj", "desc": "Người thứ {}.".format(i)}
                            for i in range(5)],
            },
        }
        decomp = {
            "concept_groups": [{"name": "nhóm bạn", "member_ids": [0, 1, 2, 3, 4],
                                "cardinality": "exact:5", "cultural": False}],
            "facts": {str(i): [{"rank": 0, "kind": "subject", "text": "người {}".format(i)},
                               {"rank": 1, "kind": "attribute", "text": "đặc điểm {}".format(i)}]
                     for i in range(5)},
        }
        short = s1b.build_subjson("x", target, decomp, "short", random.Random("s"))
        long_ = s1b.build_subjson("x", target, decomp, "long", random.Random("s"))
        self.assertLess(len(short["groups"][0]["facts"]), len(long_["groups"][0]["facts"]))
        # short chỉ được giữ vài người, không phải cả 5
        n_subjects_short = sum(1 for f in short["groups"][0]["facts"] if f.startswith("người "))
        self.assertLess(n_subjects_short, 5)

    def test_too_many_cultural_groups_are_capped_in_short(self):
        """Chốt chặn cho bug thật (server test_3): 4 nhóm cùng cultural=True (vd 4
        món trong 1 mâm cỗ) từng ĐỀU bị ép giữ ở short, phá vỡ ngân sách 8-20 từ."""
        target = {
            "high_level_description": "Mâm cỗ.",
            "style_description": {"medium": "photograph", "photo": "wide"},
            "compositional_deconstruction": {
                "background": "Bàn gỗ.",
                "elements": [{"id": i, "type": "obj", "desc": "Món {}.".format(i)}
                            for i in range(4)],
            },
        }
        decomp = {
            "concept_groups": [
                {"name": "món {}".format(i), "member_ids": [i],
                 "cardinality": "exact:1", "cultural": True}
                for i in range(4)
            ],
            "facts": {str(i): [{"rank": 0, "kind": "subject", "text": "món {}".format(i)}]
                     for i in range(4)},
        }
        short = s1b.build_subjson("x", target, decomp, "short", random.Random("s"))
        long_ = s1b.build_subjson("x", target, decomp, "long", random.Random("s"))
        self.assertLess(len(short["groups"]), 4, "vẫn giữ cả 4 nhóm cultural ở short")
        self.assertEqual(len(long_["groups"]), 4, "long không nên bị giới hạn số nhóm cultural")

    def test_depth_axis_actually_cuts_facts(self):
        """Chốt chặn cho lỗi thiết kế cũ: chỉ cắt bề rộng là chưa đủ."""
        short = self.build("short")
        group = next(g for g in short["groups"] if g["name"] == "nhóm bạn")
        # 3 element × 3 mệnh đề = 9; mức short phải cắt còn ít hơn hẳn
        self.assertLess(len(group["facts"]), 9)

    def test_is_deterministic_for_same_seed(self):
        a = s1b.build_subjson("img1", SAMPLE_TARGET, SAMPLE_DECOMP, "long", random.Random("k"))
        b = s1b.build_subjson("img1", SAMPLE_TARGET, SAMPLE_DECOMP, "long", random.Random("k"))
        self.assertEqual(a, b)

    def test_checklist_is_flattened_group_facts(self):
        sub = self.build("medium")
        flat = [f for g in sub["groups"] for f in g["facts"]]
        for fact in flat:
            self.assertIn(fact, sub["checklist"])

    def test_checklist_includes_scene_values(self):
        """Chốt chặn cho bug thật: 2a được cấp scene để viết prompt, nên checklist ở
        2b PHẢI chứa scene — thiếu nó thì mọi câu tả ánh sáng/bối cảnh bị chấm oan
        là 'thêm tin' dù model chỉ đang tả đúng phần được cấp.

        NGOẠI LỆ: khong_khi (aesthetics) chỉ là gợi ý văn phong cho 2a, không bắt
        buộc trong checklist chấm điểm của 2b (xem test_khong_khi_is_hint_only_...)."""
        for level in ("short", "medium", "long"):
            sub = self.build(level)
            for key, value in sub["scene"].items():
                if value and key != "khong_khi":
                    self.assertIn(value, sub["checklist"],
                                 "{}: thiếu '{}' trong checklist".format(level, value))

    def test_khong_khi_is_hint_only_not_in_checklist(self):
        """Chốt chặn cho bug thật (server test_6): aesthetics là danh sách tag mood
        tiếng Anh rời rạc (vd 'elegant, historical, serene'), không phải câu mô tả
        -- bắt buộc trong checklist gây 71% prompt long bị loại oan. Giữ làm gợi ý
        cho 2a viết nhưng KHÔNG chấm điểm ở 2b."""
        sub = self.build("long")
        self.assertIn("khong_khi", sub["scene"])
        self.assertNotIn(sub["scene"]["khong_khi"], sub["checklist"])

    def test_khong_khi_keeps_only_first_tag(self):
        target = dict(SAMPLE_TARGET)
        target["style_description"] = dict(SAMPLE_TARGET["style_description"])
        target["style_description"]["aesthetics"] = "elegant, historical, serene"
        scene = s1b.build_scene(target, "long")
        self.assertEqual(scene["khong_khi"], "elegant")

    def test_first_clause_does_not_leave_dangling_preposition(self):
        """Chốt chặn cho bug thật (server test_6): cắt cứng theo số từ từng để lại
        cụm cụt lửng ('...on the'), không ai viết prompt lại chèn một câu bị cắt
        cụt như vậy."""
        text = ("A dark interior space framed by a heavy brown wooden door frame "
                "on the left")
        clause = s1b.first_clause(text, 14)
        self.assertFalse(clause.split()[-1].lower() in s1b._DANGLING_TAIL_WORDS)
        self.assertEqual(clause, "A dark interior space framed by a heavy brown wooden door frame")

    def test_default_medium_photograph_excluded_from_scene(self):
        """Chốt chặn cho bug thật (server test_5): medium=='photograph' chiếm 994/1000
        mẫu -- gần như hằng số, không mang thông tin phân biệt. Bắt buộc checklist
        phải có 'photograph' khiến 44/98 lượt loại ở test_5 là oan (model không sai,
        chỉ đơn giản không ai đặt hàng ảnh lại nói 'đây là một bức ảnh')."""
        target = dict(SAMPLE_TARGET)
        target["style_description"] = dict(SAMPLE_TARGET["style_description"])
        target["style_description"]["medium"] = "photograph"
        target["style_description"]["photo"] = "eye-level medium shot, sharp focus"
        scene = s1b.build_scene(target, "long")
        self.assertNotIn("loai_anh", scene)

    def test_non_default_medium_kept_in_scene(self):
        target = dict(SAMPLE_TARGET)
        target["style_description"] = dict(SAMPLE_TARGET["style_description"])
        target["style_description"]["medium"] = "illustration"
        scene = s1b.build_scene(target, "long")
        self.assertEqual(scene.get("loai_anh"), "illustration")

    def test_default_camera_angle_excluded_from_scene(self):
        """Chốt chặn cho bug thật (server test_5): 'eye-level medium shot' là góc máy
        mặc định của phần lớn ảnh -- STEP2A_SYSTEM cấm dùng ngôn ngữ kỹ thuật khung
        hình nên model đúng khi bỏ qua, nhưng checklist cũ vẫn bắt phải nhắc tới,
        gây loại oan hàng loạt."""
        target = dict(SAMPLE_TARGET)
        target["style_description"] = dict(SAMPLE_TARGET["style_description"])
        target["style_description"]["photo"] = "eye-level medium shot, sharp focus on the subject"
        scene = s1b.build_scene(target, "long")
        self.assertNotIn("goc_chup", scene)

    def test_notable_camera_angle_kept_in_scene(self):
        target = dict(SAMPLE_TARGET)
        target["style_description"] = dict(SAMPLE_TARGET["style_description"])
        target["style_description"]["photo"] = "high-angle wide shot, deep focus"
        scene = s1b.build_scene(target, "long")
        self.assertIn("high-angle", scene.get("goc_chup", ""))

    def test_boring_eye_level_stripped_from_notable_angle_clause(self):
        """Chốt chặn cho bug thật (server test_6): dữ liệu gốc không nhất quán --
        có ảnh viết 'eye-level wide shot' dính liền (không dấu phẩy tách). Nếu giữ
        nguyên cả cụm thì 'eye-level' (mặc định, bị STEP2A_SYSTEM cấm) vẫn lọt vào
        checklist cùng từ khoá đáng chú ý, khiến model đúng luật (bỏ qua) nhưng bị
        chấm oan là thiếu 'eye-level wide shot'."""
        target = dict(SAMPLE_TARGET)
        target["style_description"] = dict(SAMPLE_TARGET["style_description"])
        target["style_description"]["photo"] = "eye-level wide shot, deep focus"
        scene = s1b.build_scene(target, "long")
        self.assertEqual(scene.get("goc_chup"), "wide shot")

    def test_duplicate_rank0_subjects_are_not_dropped_across_members(self):
        """Chốt chặn cho bug thật (server test_3): 2 người phụ nữ trong cùng một
        nhóm cùng bắt đầu bằng 'người phụ nữ' -- dedup toàn nhóm từng xoá mất
        chủ ngữ người thứ hai, khiến model tự suy luận đúng theo cardinality
        nhưng bị 2b chấm oan là 'thêm tin' vì checklist không còn dấu vết."""
        target = {
            "high_level_description": "Ba du khách trên thuyền.",
            "style_description": {"medium": "photograph", "photo": "wide"},
            "compositional_deconstruction": {
                "background": "Sông nước.",
                "elements": [
                    {"id": 0, "type": "obj", "desc": "Người đàn ông tóc đen."},
                    {"id": 1, "type": "obj", "desc": "Người phụ nữ tóc vàng buộc sau."},
                    {"id": 2, "type": "obj", "desc": "Người phụ nữ tóc vàng khác."},
                ],
            },
        }
        decomp = {
            "concept_groups": [{"name": "nhóm du khách", "member_ids": [0, 1, 2],
                                "cardinality": "exact:3", "cultural": False}],
            "facts": {
                "0": [{"rank": 0, "kind": "subject", "text": "người đàn ông"},
                      {"rank": 1, "kind": "color", "text": "tóc đen"}],
                "1": [{"rank": 0, "kind": "subject", "text": "người phụ nữ"},
                      {"rank": 1, "kind": "attribute", "text": "tóc vàng buộc sau"}],
                "2": [{"rank": 0, "kind": "subject", "text": "người phụ nữ"},
                      {"rank": 1, "kind": "attribute", "text": "tóc vàng khác"}],
            },
        }
        sub = s1b.build_subjson("x", target, decomp, "long", random.Random("s"))
        facts = sub["groups"][0]["facts"]
        self.assertEqual(facts.count("người phụ nữ"), 2,
                         "chủ ngữ người phụ nữ thứ hai bị xoá mất: {}".format(facts))
        self.assertIn("tóc vàng khác", facts)

    def test_cultural_group_name_in_checklist_even_with_facts(self):
        """Chốt chặn cho bug thật (server test_3): tên nhóm văn hoá trước đây chỉ
        được thêm vào checklist khi nhóm KHÔNG có facts nào. 'thanh đồng trong lễ
        Hầu đồng' có 8 facts (áo, mũ, quạt...) nên tên nhóm bị bỏ sót hoàn toàn,
        khiến model viết đúng tên văn hoá vẫn bị 2b chấm là 'thêm tin'."""
        target = {
            "high_level_description": "Một thanh đồng trong lễ Hầu đồng.",
            "style_description": {"medium": "photograph", "photo": "wide"},
            "compositional_deconstruction": {
                "background": "Đền thờ.",
                "elements": [{"id": 0, "type": "obj", "desc": "Áo thụng lụa xanh."}],
            },
        }
        decomp = {
            "concept_groups": [{"name": "thanh đồng trong lễ Hầu đồng", "member_ids": [0],
                                "cardinality": "exact:1", "cultural": True}],
            "facts": {"0": [{"rank": 0, "kind": "subject", "text": "áo thụng lụa xanh"}]},
        }
        sub = s1b.build_subjson("x", target, decomp, "long", random.Random("s"))
        self.assertIn("thanh đồng trong lễ Hầu đồng", sub["checklist"])

    def test_first_clause_compression(self):
        self.assertEqual(
            s1b.first_clause("Quảng trường Ba Đình rộng; bầu trời trong xanh"),
            "Quảng trường Ba Đình rộng",
        )
        self.assertEqual(s1b.first_clause("a b c d e f", max_words=3), "a b c")


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
        sub = s1b.build_subjson("i", SAMPLE_TARGET, SAMPLE_DECOMP, "short", random.Random("x"))
        spec = s2a.build_spec(sub, {"vai": "v", "giọng": "g", "ngôn ngữ": "n"})
        # Không được rò rỉ target_json hay checklist đầy đủ vào prompt của model
        self.assertNotIn("target_json", json.dumps(spec, ensure_ascii=False))
        self.assertEqual(len(spec["groups"]), len(sub["groups"]))


# =============================================================================
# STEP 2b — quy tắc đạt/loại
# =============================================================================

class _Args:
    max_missing = 0


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

    def test_step2a_messages_alternate_roles(self):
        spec = {"detail_level": "short", "length_hint": "8-20 từ",
                "persona": {}, "scene": "", "groups": []}
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
        self.assertEqual(set(io_utils.STEP_DIRS),
                         {"step0", "step1a", "step1b", "step2a", "step2b", "step2c"})

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
