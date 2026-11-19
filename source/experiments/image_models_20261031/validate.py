"""Independent reconstruction of splits, probabilities and tier metrics."""
import hashlib
import itertools
import json

import numpy as np

from common import OUT, PREVIOUS, SIZE, lines, dump


def main():
    checks = []
    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
    bases = lines(OUT/'bases.jsonl'); rows = lines(OUT/'dataset.jsonl')
    summary = json.loads((OUT/'dataset_summary.json').read_text())
    for name in ['bases.jsonl', 'dataset.jsonl', 'dataset_summary.json']:
        check('frozen_'+name, hashlib.sha256((OUT/name).read_bytes()).digest() == hashlib.sha256((PREVIOUS/name).read_bytes()).digest())
    check('row_count', len(rows) == 7799 == len({r['id'] for r in rows}))
    check('base_count', len(bases) == 1623)
    check('missing_records_retained', summary['skipped'] == {'missing_current_data': 316})
    for field in ['base_sample_id', 'split_group', 'raw_sha256']:
        groups = {s: {b[field] for b in bases if b['split'] == s} for s in ['train', 'val', 'test']}
        check('disjoint_'+field, all(not groups[a] & groups[b] for a, b in itertools.combinations(groups, 2)))
    check('tier_split_inherits_base', all(r['split'] == bases[r['base_index']]['split'] for r in rows))
    for task in ['detect', 'segment']:
        for split in ['train', 'val']:
            paths = list((OUT/'yolo_data'/task/'images'/split).glob('*.png'))
            check('yolo_'+task+'_'+split, {int(p.stem) for p in paths} == {i for i, b in enumerate(bases) if b['split'] == split})
    for path in (OUT/'predictions').glob('*.npy'):
        p = np.load(path, mmap_mode='r')
        check(path.stem+'_shape', p.shape == (len(bases), SIZE, SIZE))
        check(path.stem+'_probabilities', all(np.isfinite(p[i]).all() and p[i].min() >= 0 and p[i].max() <= 1 for i in range(len(p))))
    results = json.loads((OUT/'evaluation_results.json').read_text())
    y = np.array([r['target'] for r in rows], dtype=bool)
    for name, record in results.items():
        with np.load(OUT/'evaluation'/(name+'_probabilities.npz')) as z:
            probs = z['probabilities']; ids = z['ids']; targets = z['targets']
        check(name+'_identity', list(ids) == [r['id'] for r in rows] and np.array_equal(targets, y))
        check(name+'_finite', np.isfinite(probs).all() and probs.min() >= 0 and probs.max() <= 1)
        for method, cal in record['calibrations'].items():
            p = probs >= cal['thresholds']
            check(name+'_'+method+'_fpr_guard', (cal['P1_validation_guard']['clean_fpr'] or 0) <= .01)
            for cohort, m in cal['cohorts'].items():
                if not m.get('samples'):
                    continue
                split, family, tier = cohort.split(':')
                tiers = ['P1', 'P2', 'P3', 'P4'] if tier == 'P1_P4' else ['P2', 'P3', 'P4'] if tier == 'P2_P4' else [tier]
                ii = np.array([i for i, r in enumerate(rows) if r['split'] == split and (family == 'ALL' or r['family'] == family) and r['tier'] in tiers])
                tp = (y[ii] & p[ii]).sum(0); fp = (~y[ii] & p[ii]).sum(0); fn = (y[ii] & ~p[ii]).sum(0)
                check(name+'_'+method+'_'+cohort, int(tp.sum()) == m['tp'] and int(fp.sum()) == m['fp'] and int(fn.sum()) == m['fn']
                      and abs(m['micro_f1']-float(2*tp.sum()/max(2*tp.sum()+fp.sum()+fn.sum(), 1))) < 1e-12
                      and all((m['per_class'][code]['tp'], m['per_class'][code]['fp'], m['per_class'][code]['fn']) == (tp[k], fp[k], fn[k]) for k, code in enumerate(['L2-G', 'L3-06', 'L3-07'])))
    selection = json.loads((OUT/'selection.json').read_text())
    def key(name):
        r = results[name]; c = r['calibrations']['tier_balanced']
        values = [c['cohorts']['val:ALL:'+t]['micro_f1'] for t in ['P1', 'P2', 'P3', 'P4']]
        return (np.mean(values), min(values), -(c['P1_validation_guard']['clean_fpr'] or 0), -r['feature_count'], -r['weights_bytes'])
    primary = [n for n, r in results.items() if r['feature_set'] == 'primary']
    check('selection_reconstructed_from_validation', selection['primary_selected'] == max(primary, key=key))
    result = dict(passed=all(c['passed'] for c in checks), checks=len(checks), failed=[c for c in checks if not c['passed']],
                  scientific_accuracy_certified=False, details=checks)
    dump('validation.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'details'}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
