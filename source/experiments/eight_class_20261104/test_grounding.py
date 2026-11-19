"""Protect probability provenance and prevent empty detections certifying errors."""
import copy
import unittest
import ground_roadshow as g


class Matcher:
    @staticmethod
    def match_instances(components, geometry, width, height, size):
        return dict(matches=[], possible_omission_components=list(range(len(components))))


class GroundingSafetyTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(id='test', package='demo', data_sha256='data-sha',
                        label_path='label.json', data_path='image.png')
        self.ledger = dict(imageSize=[1024,512], dataPath='image.png', labelSha256='label-sha')
        self.signals = dict(model_candidates=['L3-06'], classes=['L3-06'], probabilities=[.12],
            investigation_triggers={'L3-06': dict(source='coarse_count_investigation', group_probability=.91)})

    def test_coarse_trigger_preserves_fine_probability_and_original_coordinates(self):
        before = copy.deepcopy(self.signals)
        component = dict(confidence=.96, bbox=[96,96,192,192], area_pixels=9216)
        findings,_ = g.candidates(self.row,self.ledger,self.signals,[component],[],[],Matcher,'model-sha')
        self.assertEqual(self.signals,before)
        self.assertEqual(len(findings),1)
        finding=findings[0]
        self.assertEqual(finding['fusion_class_probability'],.12)
        self.assertEqual(finding['fusion_group_probability'],.91)
        self.assertEqual(finding['weighted_burden'],.91)
        self.assertEqual(finding['risk_weight_source'],'GCD-C fallback')
        self.assertEqual(finding['locator']['bbox'],[256.,128.,512.,256.])
        self.assertFalse(finding['scoreEligible'])

    def test_no_detection_never_certifies_spurious_annotation(self):
        self.signals['model_candidates']=['L3-07']
        findings,_=g.candidates(self.row,self.ledger,self.signals,[],[],[],Matcher,'model-sha')
        self.assertEqual(findings,[])

    def test_invalid_annotation_cannot_be_declared_missing_target(self):
        component=dict(confidence=.99,bbox=[0,0,32,32],area_pixels=1024)
        findings,_=g.candidates(self.row,self.ledger,self.signals,[component],[],['unsupported'],Matcher,'model-sha')
        self.assertEqual(findings,[])


if __name__=='__main__': unittest.main()
