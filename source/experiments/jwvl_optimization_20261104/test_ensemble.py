import copy
import unittest
import numpy as np
import ensemble as e

class IntegrityTests(unittest.TestCase):
    def test_same_source_and_cross_family_duplicates_stay_together(self):
        r=[{'family':f,'training_group_id':str(i//5)} for i in range(60) for f in e.FAMILIES]
        p=e.partition(r);seen={}
        for row,s in zip(r,p):
            g=row['training_group_id']
            if g in seen:self.assertEqual(seen[g],s)
            seen[g]=s
        self.assertEqual(set(p),{'fit','cal','select'})

    def test_feature_whitelist_is_independent_of_truth_and_ids(self):
        a=[{'family':e.FAMILIES[0],'features':[.25]*256,'tier':'P4','target':[1,1,1],'truth_valid':[True]*3,'id':'secret','P0_difference':9}]
        b=copy.deepcopy(a);b[0].update(tier='P0',target=[0]*3,truth_valid=[False]*3,id='other',P0_difference=-4)
        p=np.full((1,21,3),.7);r=np.array([[0,1,0]],bool)
        np.testing.assert_array_equal(e.features(a,p,r),e.features(b,p,r))

    def test_rule_cannot_be_cancelled_by_model(self):
        r=np.array([[1,0,1],[0,1,0]],bool);p=np.zeros((2,3));t=np.full((6,3),1.01)
        np.testing.assert_array_equal(e.decide(p,r,t,np.array([0,4])),r)

    def test_unknown_positive_not_in_metric(self):
        y=np.array([[1,1,0]],bool);v=np.array([[0,1,1]],bool);p=np.array([[0,1,1]],bool)
        m=e.metric(y,v,p)
        self.assertEqual((m['TP'],m['FP'],m['FN'],m['TN']),(1,1,0,0))

    def test_calibration_respects_clean_constraint(self):
        # Every family has one clean and two polluted samples. Same score for
        # clean and positives: candidate must be disabled rather than game F1.
        fam=np.repeat(np.arange(6),3);tier=np.tile([0,1,2],6)
        y=np.zeros((18,3),bool);y[tier>0,0]=True
        v=np.ones_like(y);rule=np.zeros_like(y);prob=np.full((18,3),.7)
        th=e.calibrate(prob,rule,y,v,fam,tier);pred=e.decide(prob,rule,th,fam)
        self.assertFalse(pred[tier==0].any())

if __name__=='__main__':unittest.main()
