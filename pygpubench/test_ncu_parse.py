"""Unit tests for the Nsight Compute CSV reducer. Run: python -m unittest pygpubench.test_ncu_parse"""
import unittest

from ncu_parse import classify, parse_ncu_raw_csv, summarize_kernels

SAMPLE = '''==PROF== Disconnected from process 1234
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","gpu__time_duration.sum","sm__throughput.avg.pct_of_peak_sustained_elapsed","gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed","sm__warps_active.avg.pct_of_peak_sustained_active","launch__registers_per_thread","launch__shared_mem_per_block_static","launch__shared_mem_per_block_dynamic","smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio","smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio","smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"
"","","","","","","","","","","","usecond","%","%","%","register/thread","byte","byte","","",""
"0","1234","python","host","matmul_tiled_kernel(const float *, const float *, float *, int)","1","7","(16, 16, 1)","(128, 128, 1)","0","8.9","1,234.50","71.20","38.10","92.00","32","2,048","0","3.40","1.10","0.20"
"1","1234","python","host","copy_kernel","1","7","(256, 1, 1)","(64, 1, 1)","0","8.9","12.00","8.00","81.00","15.00","16","0","0","0.90","0.00","0.10"
'''


class ParseTests(unittest.TestCase):
    def test_parses_two_kernels_with_units_and_stalls(self):
        kernels = parse_ncu_raw_csv(SAMPLE)
        self.assertEqual(len(kernels), 2)
        first = kernels[0]
        self.assertTrue(first["name"].startswith("matmul_tiled_kernel"))
        self.assertAlmostEqual(first["duration_us"], 1234.5)
        self.assertEqual(first["registers_per_thread"], 32)
        self.assertEqual(first["shared_mem_bytes"], 2048)
        self.assertEqual(first["top_stalls"][0]["reason"], "long_scoreboard")
        self.assertEqual(first["verdict"], "compute-bound")
        self.assertEqual(kernels[1]["verdict"], "memory-bound")
        self.assertEqual(classify({"sm_throughput_pct": 8, "memory_throughput_pct": 10,
                                   "achieved_occupancy_pct": 15}), "latency-bound (low occupancy)")

    def test_memory_bound_when_memory_pipe_dominates(self):
        self.assertEqual(classify({"sm_throughput_pct": 20, "memory_throughput_pct": 85}), "memory-bound")
        self.assertEqual(classify({}), "unknown")

    def test_ignores_text_without_a_table(self):
        self.assertEqual(parse_ncu_raw_csv("==ERROR== ERR_NVGPUCTRPERM"), [])
        self.assertEqual(summarize_kernels([]), "no kernels were profiled")

    def test_summary_is_one_line_per_kernel(self):
        lines = summarize_kernels(parse_ncu_raw_csv(SAMPLE)).splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("SM 71%", lines[0])
        self.assertIn("long_scoreboard", lines[0])


if __name__ == "__main__":
    unittest.main()
