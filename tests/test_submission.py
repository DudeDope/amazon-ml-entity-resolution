import csv
import tempfile
import unittest
from pathlib import Path

from amazon_er.submission import SubmissionWriter


class SubmissionTests(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            with SubmissionWriter(Path(d)) as writer:
                writer.write("S1-1", [], [])
                writer.write("S1-2", ["S3-2"], ["S3-2", "S2-1"])
                with self.assertRaises(ValueError):
                    writer.write("S1-2", [], [])
                with self.assertRaises(ValueError):
                    writer.write("S1-3", ["S2-3"], [])
            with open(Path(d) / "matching_results.tsv", encoding="utf-8", newline="") as f:
                data = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual(data[0]["matched_entity_ids"], "")
            self.assertEqual(data[1]["matched_entity_ids"], "S3-2")
