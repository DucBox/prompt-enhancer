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


if __name__ == "__main__":
    unittest.main()
