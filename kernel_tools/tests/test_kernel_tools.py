"""Unit tests that never touch Modal. Run: python -m pytest kernel_tools/tests  (or unittest)."""
import unittest
from unittest import mock

from kernel_tools import evaluate, profile
from kernel_tools.runner import ScriptRun
from kernel_tools.summarize import compact_json, summarize_evaluation, summarize_profile, tool_text

GOOD_SOURCE = "import torch\nclass ModelNew(torch.nn.Module):\n    pass\n"


class ValidationTests(unittest.TestCase):
    def test_rejects_bad_requests_without_calling_modal(self):
        with mock.patch.object(evaluate, "run_modal_script") as run:
            self.assertIn("ModelNew", evaluate.evaluate_kernel(1, 1, "print('hi')")["error"])
            self.assertIn("level", evaluate.evaluate_kernel(9, 1, GOOD_SOURCE)["error"])
            self.assertIn("task_id", evaluate.evaluate_kernel(1, 0, GOOD_SOURCE)["error"])
            self.assertIn("empty", profile.profile_kernel(1, 1, "   ")["error"])
            run.assert_not_called()

    def test_passes_arguments_and_marks_ok(self):
        payload = {"level": 1, "task_id": 1, "task_name": "Square_matrix_multiplication_",
                   "gpu": "NVIDIA L4", "compiled": True, "correct": True, "errors": 0,
                   "latency_us": {"mean": 900.0, "median": 890.0, "std": 5.0, "runs": 3, "expected": 3},
                   "reference_latency_us": {"mean": 450.0}, "speedup": 0.5,
                   "feedback": "correct; 900.0 us vs torch 450.0 us (0.50x)", "time_us": [1, 2, 3]}
        with mock.patch.object(evaluate, "run_modal_script",
                               return_value=ScriptRun(True, payload, "", "", 0)) as run:
            result = evaluate.evaluate_kernel(1, 1, GOOD_SOURCE, repeats=5)
        args = run.call_args
        self.assertEqual(args.args[0], "pygpubench/evaluate_pygpubench.py")
        self.assertEqual(args.args[1], ["--task-id", "1", "--level", "1", "--repeats", "5"])
        self.assertTrue(result["ok"])
        self.assertIn("CORRECT", result["summary"])
        self.assertIn("0.50x", result["summary"])
        self.assertNotIn("time_us", compact_json(result))

    def test_modal_failure_is_reported(self):
        with mock.patch.object(profile, "run_modal_script",
                               return_value=ScriptRun(False, None, "", "boom", 1, "Modal run exited with code 1")):
            result = profile.profile_kernel(1, 1, GOOD_SOURCE)
        self.assertFalse(result["ok"])
        self.assertIn("exited with code 1", result["summary"])


class SummaryTests(unittest.TestCase):
    def test_incorrect_candidate_summary(self):
        text = summarize_evaluation({"ok": True, "level": 1, "task_id": 1, "task_name": "x", "gpu": "L4",
                                     "compiled": True, "correct": False,
                                     "feedback": "KernelBench correctness check failed: mismatch"})
        self.assertTrue(text.startswith("COMPILED but incorrect"))
        self.assertIn("mismatch", text)

    def test_profile_summary_and_tool_text(self):
        result = {"ok": True, "level": 1, "task_id": 1, "task_name": "x", "gpu": "L4", "profiled": True,
                  "summary": "k | 10.0 us | SM 70% | compute-bound", "feedback": "profiled 1 kernel launch(es) on L4"}
        text = summarize_profile(result)
        self.assertIn("PROFILED", text)
        self.assertIn("compute-bound", text)
        self.assertIn("\n\nJSON:\n{", tool_text(text, result))


if __name__ == "__main__":
    unittest.main()
