"""Leakage, coordinate and review-scope guards for the optional localizer pilot."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SOURCE = Path(os.environ.get("SPATIAL_LOCALIZER_SOURCE", str(Path(__file__).resolve().parents[1] / "experiments/research_optimization_20261103/spatial_localizer.py")))
spec = importlib.util.spec_from_file_location("spatial_localizer_pilot", SOURCE)
localizer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = localizer
spec.loader.exec_module(localizer)


def check_incomplete_objects_are_not_silently_background():
    objects = [{"geometry": {"kind": "polygon", "valid": False}},
               {"geometry": {"kind": "rectangle", "valid": True, "points": [[1, 2], [4, 5]]}}]
    geometry, errors = localizer.extract_geometry(objects, 10, 10)
    assert len(geometry) == 1
    assert errors == [{"index": 0, "reason": "unavailable_pixel_geometry"}]


def check_nonfinite_and_outside_geometry_excluded():
    objects = [{"geometry": {"kind": "rectangle", "valid": True, "points": [[-1, 2], [4, 5]]}},
               {"geometry": {"kind": "rectangle", "valid": True, "points": [[1, 2], [float("nan"), 5]]}}]
    geometry, errors = localizer.extract_geometry(objects, 10, 10)
    assert not geometry and len(errors) == 2


def check_crop_native_coordinates_and_reverse_rectangle():
    geometry = [{"index": 0, "kind": "rectangle", "points": [[300, 400], [200, 300]]}]
    full = localizer.region_mask(geometry, 1000, 1000, 100)
    crop = localizer.region_mask(geometry, 1000, 1000, 100, (200, 300, 400, 500))
    assert full[30:41, 20:31].all()
    assert crop[:51, :51].all()
    assert not crop[60:, :].any()


def check_group_and_content_leakage_rejected(key):
    train = [{"training_group_id": "g1", "base_sample_id": "b1", "data_sha256": "s1"}]
    evaluation = [{"training_group_id": "g2", "base_sample_id": "b2", "data_sha256": "s2"}]
    evaluation[0][key] = train[0][key]
    with unittest.TestCase().assertRaisesRegex(ValueError, key):
        localizer.assert_disjoint(train, evaluation)


def check_independent_components_native_mapping_and_abstention():
    pred = np.zeros((100, 100), bool)
    pred[10:30, 20:40] = True
    components = localizer.instances(pred, pred.astype(float))
    geometry = [{"index": 7, "kind": "rectangle", "points": [[400, 100], [800, 300]]},
                {"index": 8, "kind": "rectangle", "points": [[1200, 600], [1400, 700]]}]
    matched = localizer.match_instances(components, geometry, 2000, 1000, 100)
    assert matched["matches"][0]["annotation_index"] == 7
    assert matched["unmatched_annotations_abstain"] == [8]
    assert matched["scoreEligible"] is False
    row = {"id": "sample", "width": 2000, "height": 1000, "data_sha256": "frame"}
    reference = localizer.native_reference(row, components, 100, .5, "bce")
    assert reference["independent_of_annotations"] is True
    assert reference["coverage_complete"] is False
    assert reference["class_known"] is False
    assert reference["objects"][0]["label"] is None
    assert reference["objects"][0]["locator"]["bbox"] == [400, 100, 800, 300]
    assert reference["objects"][0]["locator"]["coordinate_system"]=="image_pixels"
    assert reference["objects"][0]["locator"]["image_size"]==[2000,1000]
    assert reference["complete"] is False and reference["negative_evidence_verified"] is False


def check_roi_is_fixed_grid_with_full_native_coverage():
    boxes = localizer.roi_boxes(4096, 4096)
    assert len(boxes) == 16
    assert boxes[0] == (0, 0, 1024, 1024)
    assert boxes[-1] == (3072, 3072, 4096, 4096)
    assert localizer.roi_boxes(390, 350) == []


def check_maximum_cardinality_instead_of_greedy():
    # Highest single IoU would consume target0 and hide the second valid match.
    p0=np.zeros((10,10),bool);p0[0:2,0:6]=True
    p1=np.zeros((10,10),bool);p1[0:2,6:9]=True
    components=[dict(mask=p0),dict(mask=p1)]
    geometry=[dict(index=0,kind="rectangle",points=[[0,0],[80,10]]),
              dict(index=1,kind="rectangle",points=[[0,0],[20,10]])]
    matched=localizer.match_instances(components,geometry,100,100,size=10)
    assert len(matched["matches"])==2
    assert {(m["component"],m["annotation_index"]) for m in matched["matches"]}=={(0,1),(1,0)}


class TestLocalizerGuards(unittest.TestCase):
    def test_balanced_sampler_covers_every_view_and_family(self):
        samples=[dict(family="A")]*5+[dict(family="B")]*2
        indices=localizer.balanced_epoch_indices(samples,0)
        self.assertEqual(set(indices),set(range(7)))
        self.assertEqual(sum(samples[i]["family"]=="A" for i in indices),5)
        self.assertEqual(sum(samples[i]["family"]=="B" for i in indices),5)
        self.assertEqual(indices,localizer.balanced_epoch_indices(samples,0))
    def test_unknown_schema_does_not_become_empty_target(self):
        import json
        from PIL import Image
        class Unsupported:
            def extract_annotation(self,label,family): return dict(schema_status="unsupported_structure",objects=[])
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp); image=base/"image.png"; label=base/"label.json"
            Image.new("RGB",(10,10)).save(image);label.write_text(json.dumps({}))
            row=dict(data_path=str(image),label_path=str(label),label_sha256=localizer.sha(label),family="AIA_AR_sample")
            result=localizer.inspect_row(row,Unsupported())
            self.assertFalse(result["usable"])
            self.assertEqual(result["reason"],"unsupported_or_invalid_annotation_schema")
    def test_existing_or_source_output_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            existing=root/"existing_run";existing.mkdir()
            with self.assertRaisesRegex(ValueError,"existing"):
                localizer.validate_output_path(existing,root)
            with self.assertRaisesRegex(ValueError,"overlaps"):
                localizer.validate_output_path(root/"opc1018/newpublish/new_run",root)
            localizer.validate_output_path(root/"new_safe_output",root)
    def test_actual_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);source=base/"data.bin";source.write_bytes(b"actual")
            row=dict(id="one",data_path=str(source),label_path=str(source),data_sha256="wrong",label_sha256="wrong")
            with self.assertRaisesRegex(ValueError,"Actual input hash"):
                localizer.verify_sources([row],base/"integrity.json")
    def test_incomplete(self): check_incomplete_objects_are_not_silently_background()
    def test_invalid(self): check_nonfinite_and_outside_geometry_excluded()
    def test_crop(self): check_crop_native_coordinates_and_reverse_rectangle()
    def test_group_leakage(self):
        for key in ("training_group_id", "base_sample_id", "data_sha256"):
            with self.subTest(key=key): check_group_and_content_leakage_rejected(key)
    @unittest.skipUnless(importlib.util.find_spec("scipy"), "Optional localizer scipy dependency")
    def test_components(self): check_independent_components_native_mapping_and_abstention()
    @unittest.skipUnless(importlib.util.find_spec("scipy"), "Optional localizer scipy dependency")
    def test_assignment(self): check_maximum_cardinality_instead_of_greedy()
    def test_roi(self): check_roi_is_fixed_grid_with_full_native_coverage()


if __name__ == "__main__": unittest.main()
