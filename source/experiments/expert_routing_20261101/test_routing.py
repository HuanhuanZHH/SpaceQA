"""Targeted behavioral tests for split integrity and true conditional routing."""
import json
import tempfile
import unittest
import sys
import types
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
import routing as r
from runtime import ExpertRouter


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.records = [dict(id=f'{f}:{g}:P{t}', family=f, tier=f'P{t}',
            training_group_id=f'{f}:{g}', features=[0.] * 256, input_missing=False)
            for f in r.FAMILIES for g in range(20) for t in range(5)]

    def test_whole_group_partition_covers_every_family_and_tier(self):
        split = r.partition(self.records)
        seen = {}
        for row, label in zip(self.records, split):
            key = row['training_group_id']
            self.assertEqual(seen.setdefault(key, label), label)
        for label in ['meta_fit', 'meta_cal', 'meta_select']:
            self.assertEqual(len({(row['family'], row['tier']) for row, s in zip(self.records, split) if s == label}), 30)
        np.testing.assert_array_equal(split, r.partition(list(self.records)))

    def test_router_features_ignore_truth_paths_ids_tier(self):
        row = self.records[0].copy()
        original = r.model_features([row], np.array([[.2, .3, .4]]), np.zeros((1, 3)))
        row.update(tier='P4', target=[1, 1, 1], truth_valid=[False] * 3, id='arbitrary', path='secret')
        np.testing.assert_array_equal(original, r.model_features([row], np.array([[.2, .3, .4]]), np.zeros((1, 3))))

    def test_unknowns_are_not_negative_and_rules_survive_disabled_expert(self):
        y = np.array([[1, 0, 0]], bool); valid = np.array([[1, 0, 0]], bool)
        m = r.metrics(y, valid, np.ones((1, 3), bool))
        self.assertEqual([m[k] for k in ['TP', 'FP', 'FN', 'TN']], [1, 0, 0, 0])
        rule = np.array([[0, 1, 0]], bool)
        out = r.decisions(np.ones((1, 3)), np.full((6, 3), 1.01), np.array([0]), rule)
        np.testing.assert_array_equal(out, rule)

    def test_true_topk_and_budget(self):
        sparse = r.top_weights(np.array([[.1, .6, .3]]), 2)
        self.assertEqual(int((sparse > 0).sum()), 2)
        self.assertAlmostEqual(float(sparse.sum()), 1.)
        scores = np.ones((6, 3, 3)); sizes = [10, r.CONFIG['weight_budget_bytes'], 10]
        self.assertNotIn(1, r.greedy_pool(scores, sizes, 0, 3))

    def test_selected_experts_only_and_missing_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = dict(kind='defect', assignment=[[0, 1, 0]] * 6, thresholds=[[.5] * 3] * 6)
            frozen = dict(experts=['a', 'b', 'unrequested'], selected_route='test', policies=dict(test=policy))
            Path(directory, 'frozen_policy.json').write_text(json.dumps(frozen))
            router = ExpertRouter(directory)
            names, plan = router.requests([0.] * 256, r.FAMILIES[0], [0, 0, 1])
            self.assertEqual(names, ['a', 'b'])
            np.testing.assert_array_equal(router.combine(plan, dict(a=[.6, .9, 0], b=[0, .1, 0])), [1, 0, 1])
            names, plan = router.requests([0.] * 256, r.FAMILIES[0], [0, 0, 1], input_available=False)
            self.assertEqual(names, [])
            np.testing.assert_array_equal(router.combine(plan, {}), [0, 0, 1])

    def test_train_and_portable_runtime_equivalence(self):
        rng = np.random.default_rng(12); n = 50
        records = [self.records[i].copy() for i in range(n)]
        probs = rng.uniform(.1, .9, (n, 2, 3)).astype(np.float32)
        yy = probs[:, 0] > .5; valid = np.ones_like(yy, bool)
        rules = np.zeros_like(yy); xx = r.model_features(records, probs[:, 0], rules)
        fit = np.arange(n) < 30; cal = ~fit
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            bundle = r.train_gates(xx, probs, yy, valid, fit, cal, np.ones(n), [0, 1], folder / 'pool', epochs=2)
            policy = dict(kind='learned', pool=[0, 1], topk=1, bundle='pool/gates.json', probe=0,
                          thresholds=[[.5] * 3] * 6, resident=[0, 1])
            frozen = dict(experts=['a', 'b'], selected_route='test', policies=dict(test=policy))
            (folder / 'frozen_policy.json').write_text(json.dumps(frozen))
            router = ExpertRouter(folder)
            prior_out = r.OUT
            try:
                r.OUT = folder
                expected, called = r.apply_policy(policy, records, probs, rules)
            finally:
                r.OUT = prior_out
            for i, record in enumerate(records):
                first, pending = router.requests(record['features'], record['family'], [0, 0, 0])
                self.assertEqual(first, ['a']); self.assertTrue(pending['awaiting_probe'])
                names, plan = router.requests(record['features'], record['family'], [0, 0, 0], probs[i, 0])
                needed = {name: probs[i, ['a', 'b'].index(name)] for name in names}
                np.testing.assert_array_equal(router.combine(plan, needed), expected[i])
                self.assertEqual(set(names), {['a', 'b'][j] for j in np.flatnonzero(called[i])})
            self.assertEqual(len(bundle['models']), 3)

    def test_complete_train_freeze_test_pipeline_on_thirty_packages(self):
        """Exercise serialization, post-training barrier and all thirty output cells."""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'base'; out = Path(directory) / 'out'
            (base / 'predictions').mkdir(parents=True)
            validation = []; test = []
            for fi, family in enumerate(r.FAMILIES):
                for split, number, target in [('val', 20, validation), ('test', 4, test)]:
                    for group in range(number):
                        for tier in range(5):
                            bits = [tier >= 1 and group % 2 == 0, tier >= 2, tier >= 3]
                            target.append(dict(id=f'{family}:{split}:{group}:P{tier}', family=family,
                                tier=f'P{tier}', training_group_id=f'{family}:{split}:{group}',
                                features=[float(group % 2)] * 256, input_missing=False,
                                input_sha256=f'{family}:{split}:{group}:P{tier}', target=bits,
                                truth_valid=[True] * 3))
            for name, values in [('val', validation), ('test', test)]:
                (base / f'{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in values))
            (base / 'training_barrier.json').write_text('{}')
            experts = [r.PROBE, 'unet']
            for j, expert in enumerate(experts):
                path = base / 'predictions' / (expert + '.jsonl')
                path.write_text(''.join(json.dumps(dict(id=row['id'], input_sha256=row['input_sha256'],
                    status='success', probabilities=[.9 if b else .1 for b in row['target']], seconds=.001)) + '\n'
                    for row in validation + test))
                (base / 'predictions' / (expert + '.complete.json')).write_text(json.dumps(dict(prediction_hash=r.sha(path))))
            overrides = dict(r.CONFIG, pools=[2], epochs=2)
            barrier = types.SimpleNamespace(verify_barrier=lambda: dict(checkpoint_hashes={}))
            prior_barrier = sys.modules.get('barrier')
            sys.modules['barrier'] = barrier
            try:
                with patch.object(r, 'BASE', base), patch.object(r, 'OUT', out), patch.object(r, 'CONFIG', overrides), \
                     patch.object(r, 'ready', lambda: []), patch.object(r, 'blind_rules', lambda rr: np.zeros((len(rr), 3), bool)), \
                     patch.object(r, 'footprint', lambda: [dict(route=e, checkpoint_bytes=100, eligible=True) for e in experts]):
                    r.run()
            finally:
                if prior_barrier is None:
                    sys.modules.pop('barrier', None)
                else:
                    sys.modules['barrier'] = prior_barrier
            complete = json.loads((out / 'complete.json').read_text())
            self.assertEqual(complete['packages'], 30); self.assertEqual(complete['policies'], 7)
            check = json.loads((out / 'validation.json').read_text())
            self.assertEqual(check['independent_confusion_checks'], 630)
            self.assertFalse(check['fitted_or_selected_using_test'])
            frozen = json.loads((out / 'frozen_policy.json').read_text())
            self.assertEqual(frozen['selected_on'], 'meta_select only')
            self.assertEqual(len(frozen['policies']), 7)


if __name__ == '__main__':
    unittest.main()
