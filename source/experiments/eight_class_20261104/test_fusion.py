import copy
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jwvl_optimization_20261104'))
import fusion as f


class FusionTests(unittest.TestCase):
    def test_inference_features_ignore_answers_reference_tier_and_ids(self):
        r=dict(family='ULF_sample',features=[0.]*256,tier='P0',id='secret-clean',target=[0]*8)
        s=copy.deepcopy(r);s.update(tier='P4',id='different-polluted',target=[1]*8,clean_image='forbidden',truth_valid=[True]*8)
        rule=np.zeros((1,8),bool)
        np.testing.assert_array_equal(f.feature_current([r],rule),f.feature_current([s],rule))
        self.assertEqual(f.feature_current([r],rule).shape,(1,270))

    def test_unknown_reference_is_neither_false_negative_nor_true_negative(self):
        y=np.array([True,False]);v=np.array([False,True]);p=np.array([False,False])
        m=f.e.metric(y,v,p);self.assertEqual((m['FN'],m['TN']),(0,1));self.assertIsNone(m['recall'])

    def test_calibration_uses_known_masks_and_clean_false_positive_constraint(self):
        q=np.zeros((100,8),np.float32);rule=np.zeros((100,8),bool);y=rule.copy();v=np.ones_like(rule)
        tier=np.array([0]*20+[1]*20+[2]*20+[3]*20+[4]*20);fi=np.zeros(100,int)
        y[20:60,1]=True;q[20:60,1]=.9;q[:20,1]=.8;q[60:,1]=.1
        th,_=f.calibrate(q,rule,y,v,fi,tier)
        self.assertGreater(th[0,1],.8);self.assertLessEqual(th[0,1],.9)

    def test_all_zero_support_disables_model_head(self):
        q=f.probabilities([None]*8,np.zeros((3,270),np.float32))
        self.assertEqual(q.shape,(3,8));self.assertFalse(q.any())


if __name__=='__main__':unittest.main()
