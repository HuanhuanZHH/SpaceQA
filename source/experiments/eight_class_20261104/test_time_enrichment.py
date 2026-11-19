import unittest
from enrich_time import seconds,observation_window_changes


class TimeSupervisionTests(unittest.TestCase):
    def test_equivalent_time_zones_are_not_corruption(self):
        self.assertEqual(seconds('2026-11-04T01:00:00Z','2026-11-04T09:00:00+08:00'),0.)
        self.assertEqual(seconds('2026-11-04T01:00:00','2026-11-04T01:00:00Z'),0.)

    def test_unknown_time_is_not_zero_error_or_positive(self):
        self.assertIsNone(seconds(None,'bad'));self.assertIsNone(seconds('not_a_time','bad'))

    def test_observation_window_does_not_claim_event_onset(self):
        a={'basic_annotation':{'startTime':'2026-11-04T01:00:00Z','endTime':'2026-11-04T02:00:00Z'}}
        b={'basic_annotation':{'startTime':'2026-11-04T01:01:00Z','endTime':'2026-11-04T02:00:00Z'}}
        rows=observation_window_changes(a,b)
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['pointer'],'/basic_annotation/startTime')
        self.assertEqual(rows[0]['measurements']['error_seconds'],60.)
        self.assertEqual(rows[0]['time_scope'],'observation_window_endpoint_not_event_onset')


if __name__=='__main__':unittest.main()
