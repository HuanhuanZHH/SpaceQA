import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("demo_subset_export", ROOT / "scripts" / "export_demo_subset.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DemoSubsetExportTests(unittest.TestCase):
    def test_epb_geometry_keeps_polygon_vertices(self):
        label = {
            "science_annotation": {
                "mask": {"shapes": [{"label": "EPB", "shape_type": "polygon", "points": [[1, 1], [8, 2], [4, 9]]}]}
            }
        }
        geometry = MODULE._collect_geometry(label)
        self.assertEqual(len(geometry), 1)
        self.assertEqual(geometry[0]["kind"], "polygon")
        self.assertEqual(geometry[0]["points"], [[1.0, 1.0], [8.0, 2.0], [4.0, 9.0]])

    def test_aggregate_returns_eight_codes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            path.write_text(
                "variant,family,tier,code,records,known,unknown,positives,TP,FP,FN,TN\n"
                "selected_eight,AIA_AR_sample,P1,L0-01,2,2,0,1,1,0,0,1\n",
                encoding="utf-8",
            )
            result = MODULE._aggregate(path, "selected_eight")
        self.assertEqual(set(result["AIA_AR_sample"]), set(MODULE.CLASSES))
        self.assertEqual(result["AIA_AR_sample"]["L0-01"]["code"], "L0-01")


if __name__ == "__main__":
    unittest.main()
