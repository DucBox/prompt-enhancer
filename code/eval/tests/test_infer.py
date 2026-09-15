"""Test phần logic thuần của infer.py (chọn biến thể, system prompt, parse JSON, gộp) -- không cần GPU."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import infer  # noqa: E402

ROW = {"id": "ao_dai_1", "mode": "short", "user_prompt": "cô gái mặc áo dài",
       "target_json": {"high_level_description": "x"}, "sub_json": {"checklist": ["cô gái"]}}


def args_for(*extra, adapter=None):
    base = ["--model_name", "m", "--test_file", "t.jsonl", "--output_dir", "o"]
    if adapter:
        base += ["--adapter_dir", adapter]
    return infer.parse_args(base + list(extra))


class TestVariantFlags(unittest.TestCase):

    def test_requires_at_least_one_flag(self):
        with self.assertRaises(SystemExit):
            args_for()

    def test_adapter_flag_requires_adapter_dir(self):
        with self.assertRaises(SystemExit):
            args_for("--run_adapter")

    def test_only_baseline(self):
        self.assertEqual(infer.selected_variants(args_for("--run_baseline")), ["baseline"])

    def test_only_adapter(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(infer.selected_variants(args_for("--run_adapter", adapter=d)), ["adapter"])

    def test_both_runs_adapter_first(self):
        with tempfile.TemporaryDirectory() as d:
            args = args_for("--run_baseline", "--run_adapter", adapter=d)
            self.assertEqual(infer.selected_variants(args), ["adapter", "baseline"])

    def test_4bit_default_and_can_disable(self):
        self.assertTrue(args_for("--run_baseline").load_in_4bit)
        self.assertFalse(args_for("--run_baseline", "--no-load_in_4bit").load_in_4bit)


class TestSystemPrompts(unittest.TestCase):

    def test_prefers_files_saved_with_adapter(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "system_prompt_short.txt").write_text("PROMPT SHORT LÚC TRAIN", encoding="utf-8")
            prompts, sources = infer.load_system_prompts(d)
            self.assertEqual(prompts["short"], "PROMPT SHORT LÚC TRAIN")
            self.assertTrue(sources["short"].endswith("system_prompt_short.txt"))
            self.assertEqual(prompts["long"], infer.build_system_prompt("long"))
            self.assertEqual(infer.prompt_drift(prompts), ["short"])

    def test_falls_back_to_prompts_py(self):
        prompts, sources = infer.load_system_prompts(None)
        self.assertEqual(set(sources.values()), {"code/train/prompts.py"})
        self.assertEqual(infer.prompt_drift(prompts), [])


class TestRows(unittest.TestCase):

    def test_select_by_mode_then_limit(self):
        rows = [{"id": i, "mode": m} for i, m in enumerate(["short", "long", "short", "medium", "short"])]
        picked = infer.select_rows(rows, ["short"], 2)
        self.assertEqual([r["id"] for r in picked], [0, 2])

    def test_shards_cover_all_rows_once(self):
        rows = list(range(11))
        shards = [infer.shard(rows, r, 4) for r in range(4)]
        self.assertEqual(sorted(x for s in shards for x in s), rows)

    def test_messages_match_training_layout(self):
        msgs = infer.build_messages("SYS", "USER")
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])


class TestExtractJson(unittest.TestCase):

    def test_plain(self):
        self.assertEqual(infer.extract_json('{"a": 1}'), ({"a": 1}, None))

    def test_fenced_and_think(self):
        text = "<think>\n\n</think>\n\n```json\n{\"a\": [1, 2]}\n```"
        self.assertEqual(infer.extract_json(text), ({"a": [1, 2]}, None))

    def test_extra_text_around(self):
        self.assertEqual(infer.extract_json('Here you go: {"a": 1} hope it helps')[0], {"a": 1})

    def test_truncated_json_reports_error(self):
        obj, err = infer.extract_json('{"a": {"b": 1')
        self.assertIsNone(obj)
        self.assertTrue(err)

    def test_non_object(self):
        self.assertEqual(infer.extract_json("[1, 2]"), (None, "JSON không phải object"))
        self.assertEqual(infer.extract_json("xin lỗi, tôi không làm được"), (None, "không tìm thấy object JSON"))


class TestRecordsAndMerge(unittest.TestCase):

    def test_record_fields(self):
        rec = infer.build_record(ROW, "adapter", '{"high_level_description": "y"}', 900, 4096, 3.2, 4096)
        self.assertEqual(rec["pred_json"], {"high_level_description": "y"})
        self.assertIsNone(rec["parse_error"])
        self.assertTrue(rec["hit_max_new_tokens"])
        self.assertEqual(rec["sub_json"], ROW["sub_json"])
        self.assertEqual(rec["target_json"], ROW["target_json"])

    def test_merge_keeps_test_order_and_counts(self):
        rows = [dict(ROW, id="b"), dict(ROW, id="a"), dict(ROW, id="c")]
        with tempfile.TemporaryDirectory() as d:
            for row, raw in ((rows[1], '{"k": 1}'), (rows[0], "hỏng")):
                path = infer.cache_path(d, "baseline", row)
                path.parent.mkdir(parents=True, exist_ok=True)
                rec = infer.build_record(row, "baseline", raw, 10, 5, 1.0, 4096)
                path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            stats = infer.merge_predictions(d, "baseline", rows)
            self.assertEqual(stats, {"written": 2, "missing": 1, "parse_failed": 1, "truncated": 0})
            lines = Path(d, "predictions_baseline.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(x)["id"] for x in lines], ["b", "a"])


if __name__ == "__main__":
    unittest.main()
