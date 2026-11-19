import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.evaluate_public_outputs import evaluate


class PublicEvaluationTests(unittest.TestCase):
    def write_inputs(self, base, packages, findings, ranks):
        catalog = base / "catalog.json"
        findings_path = base / "findings.jsonl"
        ranking = base / "ranking.csv"
        catalog.write_text(
            json.dumps({"provenance": {"source": "test"}, "packages": packages}),
            encoding="utf-8",
        )
        findings_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in findings),
            encoding="utf-8",
        )
        with ranking.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["rank", "datasetId"])
            writer.writeheader()
            for rank, dataset_id in ranks:
                writer.writerow({"rank": rank, "datasetId": dataset_id})
        return findings_path, ranking, catalog

    def test_clean_package_without_findings_is_an_exact_match(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            paths = self.write_inputs(
                base,
                [{"randomizedDir": "family_clean", "datasetId": "family", "tier": "clean", "defectClassIds": []}],
                [],
                [(1, "family_clean")],
            )
            result = evaluate(*paths)
            self.assertEqual(result["datasetsEvaluated"], 1)
            self.assertTrue(result["packages"][0]["classPresenceExactMatch"])
            self.assertEqual(result["classPresenceMicro"]["falsePositive"], 0)

    def test_predicted_defect_on_clean_is_a_false_positive(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            paths = self.write_inputs(
                base,
                [{"randomizedDir": "family_clean", "datasetId": "family", "tier": "clean", "defectClassIds": []}],
                [{"datasetId": "family_clean", "status": "不通过", "defectType": "L0-03"}],
                [(1, "family_clean")],
            )
            result = evaluate(*paths)
            self.assertEqual(result["classPresenceMicro"]["falsePositive"], 1)
            self.assertEqual(result["packages"][0]["falsePositiveClasses"], ["L0-03"])

    def test_two_tiers_produce_one_correct_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            packages = [
                {"randomizedDir": "family_clean", "datasetId": "family", "tier": "clean", "defectClassIds": []},
                {"randomizedDir": "family_p1", "datasetId": "family", "tier": "P1", "defectClassIds": ["L0-03"]},
            ]
            paths = self.write_inputs(
                base,
                packages,
                [{"datasetId": "family_p1", "status": "不通过", "defectType": "L0-03"}],
                [(1, "family_clean"), (2, "family_p1")],
            )
            result = evaluate(*paths)
            self.assertEqual(result["tierPairwise"]["comparablePairs"], 1)
            self.assertEqual(result["tierPairwise"]["correctPairs"], 1)
            self.assertEqual(result["tierPairwise"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
