"""Test thuần logic cho code/train/prompts.py -- không cần torch/transformers/GPU.

train_prompt_enhancer_qwen36.py import torch ngay ở đầu file nên không thể unit-test
trực tiếp trên máy không có các thư viện đó (đúng quy ước của repo: phần liên quan
model/GPU chỉ kiểm tra được trên server). prompts.py tách riêng chính vì lý do này --
system prompt là logic thuần (ghép chuỗi), nên phải test được ở đây.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prompts  # noqa: E402


class TestBuildSystemPrompt(unittest.TestCase):

    def test_all_three_levels_build_without_error(self):
        for level in prompts.LEVELS:
            text = prompts.build_system_prompt(level)
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 0)

    def test_invalid_level_raises(self):
        with self.assertRaises(ValueError):
            prompts.build_system_prompt("medium_long_extra_ultra")

    def test_each_level_prompt_is_distinct(self):
        """Chốt chặn cho đúng quyết định thiết kế: mỗi mức phải có system prompt
        RIÊNG, không phải cùng một prompt kèm một tag chung."""
        texts = {level: prompts.build_system_prompt(level) for level in prompts.LEVELS}
        self.assertEqual(len(set(texts.values())), len(prompts.LEVELS))

    def test_every_level_shares_the_same_base_schema(self):
        """Phần schema/luật chung (cultural preservation, JSON schema) phải giống
        nhau ở cả 3 mức -- chỉ phần hướng dẫn theo mức là khác."""
        for level in prompts.LEVELS:
            text = prompts.build_system_prompt(level)
            self.assertIn("áo dài", text)
            self.assertIn("bbox", text)
            self.assertIn("high_level_description", text)

    def test_short_tells_model_to_expand_not_stay_sparse(self):
        text = prompts.build_system_prompt("short")
        self.assertIn("SHORT", text)
        self.assertIn("EXPAND", text)

    def test_long_tells_model_to_stay_faithful_not_invent(self):
        text = prompts.build_system_prompt("long")
        self.assertIn("LONG", text)
        self.assertIn("FAITHFULLY STRUCTURE", text)

    def test_level_guidance_keys_match_levels_tuple(self):
        self.assertEqual(set(prompts.LEVEL_GUIDANCE.keys()), set(prompts.LEVELS))

    def test_schema_never_shows_an_id_field(self):
        """`id` là tay cầm nội bộ của pipeline sinh dữ liệu, KHÔNG thuộc schema
        Ideogram 4 -- CaptionVerifier báo "unknown keys ['id']". Nhãn Y đã được
        step2d bóc `id`, nên system prompt cũng không được dạy model sinh ra nó."""
        for level in prompts.LEVELS:
            text = prompts.build_system_prompt(level)
            self.assertNotIn('"id"', text)
            self.assertIn("no id field", text)

    def test_levels_are_defined_by_content_not_word_counts(self):
        """Số từ thực tế lệch xa mọi khoảng cố định (long 54-402 từ) -- mô tả mức theo
        NỘI DUNG giống định nghĩa ở step 1b, không ghi khoảng số từ."""
        import re
        for level in prompts.LEVELS:
            text = prompts.build_system_prompt(level)
            self.assertIsNone(re.search(r"\d+\s*-\s*\d+\s*words", text), level)

    def test_lists_medium_values_seen_in_data(self):
        for level in prompts.LEVELS:
            text = prompts.build_system_prompt(level)
            for medium in ("photograph", "illustration", "3d_render"):
                self.assertIn('"{}"'.format(medium), text)

    def test_input_may_be_vietnamese_or_english(self):
        for level in prompts.LEVELS:
            text = prompts.build_system_prompt(level)
            self.assertIn("Vietnamese or in English", text)

    def test_output_contract_is_single_minified_object_without_extra_keys(self):
        for level in prompts.LEVELS:
            text = prompts.build_system_prompt(level)
            self.assertIn("minified JSON on a single line", text)
            for key in ("bbox", "color_palette", "aspect_ratio"):
                self.assertIn("no " + key, text)

    def test_vietnamese_terms_kept_with_optional_gloss(self):
        """Nhãn Y giữ tên Việt, 88/199 ảnh kèm gloss tiếng Anh trong ngoặc (76 ảnh ở HLD)."""
        text = prompts.BASE_SYSTEM_PROMPT
        self.assertIn("full diacritics", text)
        self.assertIn("Áo ngũ thân (traditional five-panel tunic)", text)
        self.assertIn("Name the term in high_level_description", text)

    def test_activity_request_keeps_the_activity_as_main_subject(self):
        self.assertIn("making pottery", prompts.BASE_SYSTEM_PROMPT)
        self.assertIn("not merely list", prompts.BASE_SYSTEM_PROMPT)

    def test_vague_counts_stay_vague(self):
        self.assertIn("Never invent a precise number", prompts.BASE_SYSTEM_PROMPT)

    def test_does_not_import_creative_rules_that_contradict_real_captions(self):
        """Luật sáng tác của magic prompt v1 trái với caption ảnh thật (Y có 'warm' ở ~20%
        dòng, hiếm text element, tách phụ kiện thành element) -- không được đưa vào."""
        text = prompts.BASE_SYSTEM_PROMPT.lower()
        for banned in ("iphone", "text everywhere", "never use warm", "transparent background"):
            self.assertNotIn(banned, text)

    def test_prompt_stays_compact_for_sequence_budget(self):
        """Prompt lặp lại ở mọi mẫu: giữ dưới ~1200 từ để mẫu dài nhất (X+Y ~5.1k ký tự)
        vẫn vừa max_seq_length 4096."""
        for level in prompts.LEVELS:
            self.assertLess(len(prompts.build_system_prompt(level).split()), 1200, level)

    def test_schema_documents_both_style_key_orders(self):
        """Ideogram dùng hai thứ tự khác nhau: ảnh chụp thì `photo` TRƯỚC `medium`,
        còn lại thì `art_style` SAU `medium`. Gộp làm một là sai một nửa."""
        for level in prompts.LEVELS:
            text = prompts.build_system_prompt(level)
            self.assertIn('"photo": "...", "medium"', text)
            self.assertIn('"medium": "illustration", "art_style"', text)


if __name__ == "__main__":
    unittest.main()
