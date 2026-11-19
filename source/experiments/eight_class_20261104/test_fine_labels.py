import copy
import unittest
import fine_labels as f


class FineLabelsTests(unittest.TestCase):
    def row(self,changes=(),**counts):
        return dict(family='EPB_seg_sample',counts=dict(ambiguous_clean=0,ambiguous_variant=0,**counts),
            object_matches=[{'confidence':'strong'}],image_comparison={'status':'identical_file_bytes'},changes=list(changes))

    def test_winding_and_vertex_origin_do_not_change_geometry(self):
        p=[[0,0],[10,0],[10,10],[0,10],[0,0]]
        q=[[10,0],[0,0],[0,10],[10,10],[10,0]]
        self.assertEqual(f.geometry_difference(p,q)['geometry'],False)
        self.assertEqual(f.geometry_difference(p,q)['coverage'],False)

    def test_shift_and_coverage_are_distinct_multilabel_targets(self):
        p=[[0,0],[10,0],[10,10],[0,10]]
        shifted=[[5,0],[15,0],[15,10],[5,10]]
        larger=[[-5,-5],[15,-5],[15,15],[-5,15]]
        self.assertEqual((f.geometry_difference(p,shifted)['geometry'],f.geometry_difference(p,shifted)['coverage']),(True,False))
        self.assertEqual((f.geometry_difference(p,larger)['geometry'],f.geometry_difference(p,larger)['coverage']),(False,True))

    def test_invalid_geometry_is_unknown(self):
        c=dict(category='geometry_changed',object_match_confidence='strong',before=[[0,0]],after=[[1,1]])
        p=f.image_diff_label(self.row([c]));self.assertFalse(p['valid'][1]);self.assertFalse(p['valid'][7])

    def test_changed_pixels_mask_positive_and_negative_visual_references(self):
        r=self.row([dict(category='annotation_removed',clean_pointer='/x/0')]);r['image_comparison']['status']='pixels_different'
        p=f.image_diff_label(r)
        for b in [1,5,6,7]:self.assertFalse(p['target'][b]);self.assertFalse(p['valid'][b])

    def test_added_and_removed_are_not_swapped(self):
        for category,bit in [('annotation_removed',5),('annotation_added',6)]:
            p=f.image_diff_label(self.row([dict(category=category,variant_pointer='/x/0')]))
            self.assertTrue(p['valid'][bit]);self.assertTrue(p['target'][bit]);self.assertFalse(p['target'][11-bit])

    def test_candidate_identity_is_not_truth(self):
        r=self.row([dict(category='class_changed',object_match_confidence='candidate',variant_pointer='/hek_event_annotation/matched_events/0/event_type')]);r['object_matches']=[dict(confidence='candidate')]
        p=f.image_diff_label(r);self.assertFalse(p['target'][2]);self.assertFalse(p['valid'][2])

    def test_ulf_unique_source_locates_time(self):
        a={'ulf_event_annotation':{'matched_events':[dict(source='x',kind='Pi2',start_dt='00',end_dt='01')]}}
        b=copy.deepcopy(a);b['ulf_event_annotation']['matched_events'][0]['end_dt']='02'
        p=f.ulf_diff_label(a,b,True);self.assertTrue(p['target'][3]);self.assertTrue(p['valid'][3]);self.assertFalse(p['target'][2])

    def test_ulf_ambiguous_events_remain_unknown(self):
        a={'ulf_event_annotation':{'matched_events':[dict(source='x',kind='Pi2',start_dt='00',end_dt='01'),dict(source='x',kind='Pi2',start_dt='02',end_dt='03')]}}
        b=copy.deepcopy(a)
        for x in b['ulf_event_annotation']['matched_events']:x['start_dt']+='changed';x['end_dt']+='changed'
        p=f.ulf_diff_label(a,b,True);self.assertFalse(p['valid'][3]);self.assertIn('ambiguous_ulf_event_identity',p['exclusions'])


if __name__=='__main__':unittest.main()
