import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("server_risk_checks", Path(__file__).with_name("server-risk-checks.py"))
assert spec is not None and spec.loader is not None
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


class TestServerRiskChecks(unittest.TestCase):
    def test_recall_repeats_short_corpus_to_exact_depth_and_detects_missing_needle(self):
        lengths = []

        def request(server, path, body):
            if path == "/tokenize":
                return {"tokens": list(body["content"].encode())}
            lengths.append(len(body["prompt"]))
            self.assertIn(b"BLUE-HARBOR-7342", bytes(body["prompt"]))
            return {
                "tokens_evaluated": len(body["prompt"]), "tokens_predicted": 1,
                "stop_type": "eos", "content": "",
            }

        with patch.object(checks, "request", side_effect=request), contextlib.redirect_stdout(io.StringIO()) as output:
            failed = checks.check_recall("http://localhost", [65, 66], [200, 300])
        self.assertTrue(failed)
        self.assertEqual(lengths, [200, 300])
        self.assertTrue(all(json.loads(line)["silent_eos"] for line in output.getvalue().splitlines()))

    def test_depth_mismatch_is_not_a_passing_check(self):
        with patch.object(checks, "request", return_value={"tokens_evaluated": 2}):
            with self.assertRaisesRegex(ValueError, "depth mismatch"):
                checks.complete("http://localhost", [1, 2, 3], 32)

    def test_parallel_reports_each_slot_and_flags_zero_acceptance(self):
        def complete(server, prompt, n_predict, slot):
            return {
                "tokens_evaluated": len(prompt), "tokens_predicted": 32,
                "timings": {"draft_n": 30, "draft_n_accepted": 10 if slot == 0 else 0},
            }

        with patch.object(checks, "tokenize", return_value=[3]), patch.object(checks, "complete", side_effect=complete):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertTrue(checks.check_parallel("http://localhost", [1, 2], 2, 4096, [0, 1]))
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([(row["round"], row["slot"]) for row in rows], [(1, 0), (1, 1), (2, 0), (2, 1)])
        self.assertEqual([row["zero_acceptance"] for row in rows], [False, True, False, True])

    def test_parallel_short_generation_is_inconclusive_and_worker_errors_propagate(self):
        result = {"tokens_predicted": 1, "timings": {"draft_n": 0, "draft_n_accepted": 0}}
        with patch.object(checks, "tokenize", return_value=[3]), patch.object(checks, "complete", return_value=result):
            with self.assertRaisesRegex(ValueError, "insufficient"):
                checks.check_parallel("http://localhost", [1, 2], 1, 4096, [0, 1])
        with patch.object(checks, "tokenize", return_value=[3]), patch.object(checks, "complete", side_effect=OSError("offline")):
            with self.assertRaisesRegex(OSError, "offline"):
                checks.check_parallel("http://localhost", [1, 2], 1, 4096, [0, 1])


if __name__ == "__main__":
    unittest.main()
