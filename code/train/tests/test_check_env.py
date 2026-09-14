"""Test phần logic thuần của check_env.py (gom lệnh cài, báo cáo) -- không cần GPU."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_env as ce  # noqa: E402


class TestInstallCommands(unittest.TestCase):

    def test_nothing_missing_gives_no_command(self):
        self.assertEqual(ce.install_commands([ce.Check("torch", ce.OK, "2.8")]), [])

    def test_groups_cuda_bound_regular_and_kernels(self):
        checks = [
            ce.Check("torch", ce.FAIL, pip=["torch"]),
            ce.Check("peft", ce.FAIL, pip=["peft"]),
            ce.Check("transformers: kiến trúc qwen3_5", ce.FAIL, pip=["transformers"]),
            ce.Check("transformers: AutoModelForMultimodalLM", ce.FAIL, pip=["transformers"]),
            ce.Check("flash-linear-attention", ce.WARN, pip=["flash-linear-attention"]),
        ]
        commands = ce.install_commands(checks)
        self.assertEqual(len(commands), 3)
        self.assertTrue(commands[0].startswith("pip install -U torch"))
        self.assertEqual(commands[1], "pip install -U peft transformers")  # không lặp transformers
        self.assertIn("flash-linear-attention", commands[2])

    def test_missing_package_is_reported_with_pip_name(self):
        check = ce.check_package("khong_ton_tai_abc", "goi-khong-ton-tai")
        self.assertEqual(check.status, ce.FAIL)
        self.assertEqual(check.pip, ["goi-khong-ton-tai"])
        optional = ce.check_package("khong_ton_tai_abc", "goi-khong-ton-tai", required=False)
        self.assertEqual(optional.status, ce.WARN)


class TestReport(unittest.TestCase):

    def run_report(self, checks):
        lines = []
        ok = ce.report(checks, printer=lines.append)
        return ok, "\n".join(lines)

    def test_fail_means_not_ready(self):
        ok, text = self.run_report([ce.Check("CUDA", ce.FAIL, "không thấy GPU")])
        self.assertFalse(ok)
        self.assertIn("CHƯA ĐỦ", text)

    def test_warnings_only_is_ready(self):
        ok, text = self.run_report([ce.Check("torch", ce.OK), ce.Check("fla", ce.WARN, pip=["flash-linear-attention"])])
        self.assertTrue(ok)
        self.assertIn("ĐỦ để train", text)
        self.assertIn("flash-linear-attention", text)


if __name__ == "__main__":
    unittest.main()
