"""Validation-only calibration, with every pollution tier reported separately."""
import csv
import itertools
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from common import (OUT, ROOT, PREVIOUS, SIZE, SEED, lines, dump, emit, residual_features,
                    metrics, fit_gates, annotation_boxes, ious,
                    dense_instances, permutation)

CODES = ['L2-G', 'L3-06', 'L3-07']
TIERS = ['P1', 'P2', 'P3', 'P4']


def tier_gates(y, pp, tiers, guard_y, guard_pp):
    best = None
    for thresholds in itertools.product([.5, .7, .85, .95, .99, 1.01], repeat=3):
        prediction = pp >= thresholds
        guard = metrics(guard_y, guard_pp >= thresholds)
        if (guard['clean_fpr'] or 0) > .01:
            continue
        values = [metrics(y[tiers == t], prediction[tiers == t])['micro_f1'] for t in TIERS]
        key = (float(np.mean(values)), min(values), -(guard['clean_fpr'] or 0), sum(thresholds))
        if best is None or key > best[0]:
            best = (key, list(thresholds))
    return best[1]


def instance_features(objects, boxes):
    objects = [r for r in objects if r['confidence'] >= .25]
    pred = np.array([r['xyxy'] for r in objects], float).reshape(-1, 4)
    scores = [r['confidence'] for r in objects]
    mat = ious(boxes, pred)
    abest = mat.max(1) if len(pred) else np.zeros(len(boxes))
    pbest = mat.max(0) if len(boxes) else np.zeros(len(pred))
    def stats(a):
        return [float(a.min()), float(a.mean()), float(a.max())] if len(a) else [0., 0., 0.]
    centers = []
    ratios = []
    if len(boxes) and len(pred):
        for i, b in enumerate(boxes):
            p = pred[mat[i].argmax()]
            centers.append(float(np.linalg.norm((b[:2]+b[2:]-p[:2]-p[2:])/2)))
            ratios.append(float(abs(np.log(max(np.prod(p[2:]-p[:2]), 1e-8)/max(np.prod(b[2:]-b[:2]), 1e-8)))))
    return [len(pred), len(boxes), int((pbest < .5).sum()), int((abest < .5).sum()),
            int((pbest < .25).sum()), int((abest < .25).sum()),
            *stats(abest), *stats(pbest), float(np.mean(scores)) if scores else 0.,
            max(scores, default=0.), float(np.mean(centers)) if centers else 0.,
            float(np.mean(ratios)) if ratios else 0., len(pred)-len(boxes)]


def instance_report(bases, objects):
    result = {}
    for split in ['val', 'test']:
        for family in ['ALL', 'EPB_seg_sample', 'DMSP_SSUSI']:
            ii = [i for i, b in enumerate(bases) if b['split'] == split and (family == 'ALL' or b['family'] == family)]
            for threshold in [.25, .5]:
                tp = fp = total = 0
                scored = []
                for i in ii:
                    b = bases[i]
                    w, h = (390, 350) if b['family'] == 'DMSP_SSUSI' else (1024, 1024)
                    truth = annotation_boxes(json.loads(Path(b['reference_label_path']).read_text()), w, h)
                    total += len(truth)
                    used = set()
                    for obj in sorted(objects[i], key=lambda r: -r['confidence']):
                        if obj['confidence'] < .25:
                            continue
                        overlap = ious(np.array([obj['xyxy']]), truth)[0]
                        available = [j for j in range(len(truth)) if j not in used and overlap[j] >= threshold]
                        matched = bool(available)
                        if matched:
                            used.add(max(available, key=lambda j: overlap[j])); tp += 1
                        else:
                            fp += 1
                        scored.append((obj['confidence'], int(matched)))
                scored.sort(reverse=True)
                hits = np.cumsum([s[1] for s in scored])
                rec = hits/max(total, 1)
                precision = hits/np.arange(1, len(scored)+1)
                ap = np.mean([precision[rec >= q].max() if np.any(rec >= q) else 0. for q in np.linspace(0, 1, 101)])
                result[f'{split}:{family}:IoU{threshold}'] = dict(samples=len(ii), truth_instances=total,
                    tp=tp, fp=fp, fn=total-tp, precision=tp/max(tp+fp, 1), recall=tp/max(total, 1),
                    ap101_truncated_at_conf025=float(ap), confidence=.25, greedy_one_to_one=True)
    return result


def predict(classifiers, xx):
    pp = np.zeros((len(xx), 3))
    for k, model in enumerate(classifiers):
        classes = list(model.classes_)
        if 1 in classes:
            pp[:, k] = model.predict_proba(xx)[:, classes.index(1)]
    return np.round(pp, 12)


def cohorts(rows, y, pred):
    result = {}
    for split in ['val', 'test']:
        for family in ['ALL', 'EPB_seg_sample', 'DMSP_SSUSI']:
            for tier in ['P0', *TIERS, 'P2_P4', 'P1_P4']:
                tier_set = TIERS if tier == 'P1_P4' else TIERS[1:] if tier == 'P2_P4' else [tier]
                ii = np.array([i for i, r in enumerate(rows) if r['split'] == split and
                    (family == 'ALL' or r['family'] == family) and r['tier'] in tier_set], dtype=int)
                result[f'{split}:{family}:{tier}'] = metrics(y[ii], pred[ii])
    return result


def selection_key(record):
    cal = record['calibrations']['tier_balanced']
    values = [cal['cohorts']['val:ALL:'+t]['micro_f1'] for t in TIERS]
    return (float(np.mean(values)), min(values), -(cal['P1_validation_guard']['clean_fpr'] or 0),
            -record['feature_count'], -record.get('weights_bytes', 0))


def legacy_controls(rows, y, va, v1, vv):
    sources = [(ROOT/'runs/spatial-research-20261031', 'unet', 'legacy_unet224'),
               (ROOT/'runs/spatial-research-20261031', 'dino', 'legacy_dino224'),
               (ROOT/'runs/spatial-research-20261031', 'unet_oof', 'legacy_crossfit_unet224'),
               (ROOT/'runs/spatial-research-20261031', 'ijepa', 'legacy_ijepa224'),
               (PREVIOUS, 'dino', 'legacy_dino448')]
    result = {}
    for directory, original, name in sources:
        path = directory/(original+'_detector_probabilities.npz')
        if not path.exists():
            continue
        with np.load(path) as z:
            pp = z['probabilities']; ids = z['ids']; targets = z['targets']
        assert list(ids) == [r['id'] for r in rows] and np.array_equal(targets, y), name+': frozen manifest mismatch'
        gates = dict(P1_only=fit_gates(y[v1], pp[v1]), pooled=fit_gates(y[va], pp[va], y[v1], pp[v1]),
            tier_balanced=tier_gates(y[vv], pp[vv], np.array([rows[i]['tier'] for i in vv]), y[v1], pp[v1]))
        checkpoint = directory/(original+'.pt')
        nbytes = checkpoint.stat().st_size if checkpoint.exists() else 0
        if original in ['dino', 'ijepa']:
            modeldir = ROOT.parent/'models'/('dinov2-small' if original == 'dino' else 'ijepa_vith14_1k')
            weights = list(modeldir.glob('*.safetensors')) or list(modeldir.glob('pytorch_model*.bin'))
            nbytes += sum(p.stat().st_size for p in weights)
        record = dict(route=name, feature_set='primary', feature_count=35, weights_bytes=nbytes,
            source_directory=str(directory), source_route=original, calibrations={}, exposed_test=True,
            scientific_ground_truth=False, formal_scoring_enabled=False,
            training_residuals_are_in_sample=original != 'unet_oof',
            note='Recalibrated saved probabilities; no new permutation/localization measurement')
        for method, thresholds in gates.items():
            pred = pp >= thresholds
            cal = dict(thresholds=thresholds, P1_validation_guard=metrics(y[v1], pred[v1]), cohorts=cohorts(rows, y, pred))
            values = [cal['cohorts']['val:ALL:'+t]['micro_f1'] for t in TIERS]
            cal.update(validation_mean_tier_f1=float(np.mean(values)), validation_min_tier_f1=min(values))
            record['calibrations'][method] = cal
        fullname = name+'__primary'
        np.savez_compressed(OUT/'evaluation'/(fullname+'_probabilities.npz'), probabilities=pp, targets=y, ids=ids)
        dump('evaluation/'+fullname+'.json', record); result[fullname] = record
    return result


def main():
    start = time.monotonic()
    rows = lines(OUT/'dataset.jsonl'); bases = lines(OUT/'bases.jsonl')
    y = np.array([r['target'] for r in rows], dtype=int)
    ann = np.array([r['annotation_features'] for r in rows])
    labels = [json.loads(Path(r['label_path']).read_text()) for r in rows]
    boxes = [annotation_boxes(a, r['width'], r['height']) for a, r in zip(labels, rows)]
    current = np.load(OUT/'current.npy', mmap_mode='r')
    truth = np.load(OUT/'truth.npy', mmap_mode='r')
    tr = np.array([i for i, r in enumerate(rows) if r['split'] == 'train'])
    va = np.array([i for i, r in enumerate(rows) if r['split'] == 'val'])
    v1 = np.array([i for i in va if rows[i]['tier'] == 'P1'])
    vv = np.array([i for i in va if rows[i]['tier'] in TIERS])
    test = np.array([i for i, r in enumerate(rows) if r['split'] == 'test' and r['tier'] in TIERS])
    perm = permutation(bases)
    results = {}
    (OUT/'evaluation').mkdir(exist_ok=True)
    paths = sorted((OUT/'predictions').glob('*.npy'))
    configurations = [('annotation_only', None)] + [(p.stem, p) for p in paths]
    for route, path in configurations:
        emit('evaluating', route=route)
        names = [route+'__'+s for s in (['primary', 'instance_extended'] if path else ['primary'])]
        if all((OUT/'evaluation'/(n+'.json')).exists() for n in names):
            for name in names:
                results[name] = json.loads((OUT/'evaluation'/(name+'.json')).read_text())
            continue
        prediction = None
        if path:
            prediction = np.load(path, mmap_mode='r')
            instpath = OUT/'instances'/(route+'.jsonl')
            if instpath.exists():
                inst = lines(instpath)
                assert [r['base_index'] for r in inst] == list(range(len(bases)))
                objects = [r['instances'] for r in inst]
            else:
                objects = [dense_instances(p) for p in prediction]
            dense = np.array([residual_features(prediction[r['base_index']], current[i], a, r['width'], r['height'])
                              for i, (r, a) in enumerate(zip(rows, labels))])
            ext = np.array([instance_features(objects[r['base_index']], b) for r, b in zip(rows, boxes)])
            primary = np.concatenate([ann, dense], axis=1)
            xxperm = np.concatenate([ann[test], np.array([residual_features(prediction[perm[rows[i]['base_index']]],
                current[i], labels[i], rows[i]['width'], rows[i]['height']) for i in test])], axis=1)
            experm = np.array([instance_features(objects[perm[rows[i]['base_index']]], boxes[i]) for i in test])
            dump('evaluation/'+route+'_instances.json', instance_report(bases, objects))
        else:
            primary = ann
        for feature_set in (['primary', 'instance_extended'] if path else ['primary']):
            name = route+'__'+feature_set
            xx = primary if feature_set == 'primary' else np.concatenate([primary, ext], axis=1)
            cached = OUT/'evaluation'/(name+'.json')
            if cached.exists():
                results[name] = json.loads(cached.read_text())
                continue
            ts = time.monotonic()
            classifiers = []
            for k in range(3):
                m = ExtraTreesClassifier(n_estimators=160, min_samples_leaf=3, class_weight='balanced', random_state=SEED, n_jobs=4)
                m.fit(xx[tr], y[tr, k]); classifiers.append(m)
            pp = predict(classifiers, xx)
            gates = dict(P1_only=fit_gates(y[v1], pp[v1]), pooled=fit_gates(y[va], pp[va], y[v1], pp[v1]),
                tier_balanced=tier_gates(y[vv], pp[vv], np.array([rows[i]['tier'] for i in vv]), y[v1], pp[v1]))
            trainmeta = OUT/'training'/(route+'.json')
            meta = json.loads(trainmeta.read_text()) if trainmeta.exists() else {}
            if route == 'baseline_unet':
                oldmeta = json.loads((PREVIOUS/'unet_training.json').read_text())
                meta['weights_bytes'] = oldmeta['checkpoint_bytes']
            record = dict(route=route, feature_set=feature_set, feature_count=xx.shape[1],
                weights_bytes=meta.get('weights_bytes', 0), calibrations={}, exposed_test=True,
                scientific_ground_truth=False, formal_scoring_enabled=False,
                training_residuals_are_in_sample=True)
            pperm = predict(classifiers, xxperm if feature_set == 'primary' else np.concatenate([xxperm, experm], axis=1)) if path else None
            for method, thresholds in gates.items():
                pg = pp >= thresholds
                cal = dict(thresholds=thresholds, P1_validation_guard=metrics(y[v1], pg[v1]), cohorts=cohorts(rows, y, pg))
                vals = [cal['cohorts']['val:ALL:'+t]['micro_f1'] for t in TIERS]
                cal['validation_mean_tier_f1'] = float(np.mean(vals)); cal['validation_min_tier_f1'] = min(vals)
                if path:
                    cal['image_permutation'] = dict(matched=metrics(y[test], pg[test]), permuted=metrics(y[test], pperm >= thresholds),
                        changed_samples=int(np.any(pg[test] != (pperm >= thresholds), axis=1).sum()))
                    loc = {}
                    for tier in [*TIERS, 'P2_P4']:
                        ii = [i for i in test if rows[i]['tier'] in (TIERS[1:] if tier == 'P2_P4' else [tier])]
                        for k, code in enumerate(CODES):
                            values = []
                            for i in ii:
                                if not (pg[i, k] and y[i, k]):
                                    continue
                                res = (prediction[rows[i]['base_index']] > .5) ^ (current[i] > 0)
                                target = truth[i] > 0
                                values.append(float((res & target).sum()/max(1, (res | target).sum())))
                            loc[tier+':'+code] = dict(correct_class_samples=len(values),
                                mean_residual_iou=float(np.mean(values)) if values else None,
                                samples_iou_at_least_05=int(np.sum(np.array(values) >= .5)), official_location_credit=False)
                    cal['localization_proxy'] = loc
                record['calibrations'][method] = cal
            record['fit_evaluate_seconds'] = time.monotonic()-ts
            with (OUT/'evaluation'/(name+'_classifiers.pkl')).open('wb') as f:
                pickle.dump(dict(classifiers=classifiers, gates=gates, feature_count=xx.shape[1], codes=CODES), f, protocol=4)
            np.savez_compressed(OUT/'evaluation'/(name+'_probabilities.npz'), probabilities=pp, targets=y,
                ids=np.array([r['id'] for r in rows]), test_indices=test,
                permuted_test=pperm if pperm is not None else np.zeros((0, 3)))
            dump('evaluation/'+name+'.json', record); results[name] = record
            dump('evaluation_results.json', results)
            emit('evaluated', name=name, validation_key=selection_key(record),
                 test_tier_f1={t: record['calibrations']['tier_balanced']['cohorts']['test:ALL:'+t]['micro_f1'] for t in TIERS})
    results.update(legacy_controls(rows, y, va, v1, vv))
    primary = [n for n, r in results.items() if r['feature_set'] == 'primary']
    extended = [n for n, r in results.items() if r['feature_set'] == 'instance_extended']
    adapted = ['yolo11_detect', 'yolo11_segment', 'yolov8_segment', 'attention_unet', 'unetpp', 'resnet18_unet', 'deeplab_mobilenet']
    eligible = [n for n in primary if results[n]['route'] in adapted]
    ranking = sorted(eligible, key=lambda n: selection_key(results[n]), reverse=True)
    selected = max(primary, key=lambda n: selection_key(results[n]))
    selection = dict(primary_selected=selected, extended_selected=max(extended, key=lambda n: selection_key(results[n])) if extended else None,
        initial_architecture_ranking=ranking, tuning_architectures=[results[n]['route'] for n in ranking[:2]],
        criterion='Validation equal-tier mean F1, worst tier F1, P1 clean FPR, fewer features, smaller weights',
        never_use_test_to_select=True, never_switch_model_using_known_pollution_tier=True,
        formal_scoring_enabled=False, total_evaluation_seconds=time.monotonic()-start,
        limitations=['EPB/DMSP only; exposed exploratory test; technical clean-derived labels.',
                    'P1 validation omission support zero; test omission support one.',
                    'In-sample training residual features; cross-fitting remains a separate control.',
                    'No official class-and-location score or A10 Docker acceptance yet.'])
    dump('selection.json', selection); dump('evaluation_results.json', results)
    table = []
    for name, r in results.items():
        for method, cal in r['calibrations'].items():
            for cohort, m in cal['cohorts'].items():
                split, family, tier = cohort.split(':')
                if not m.get('samples'):
                    continue
                table.append(dict(configuration=name, calibration=method, split=split, family=family, tier=tier,
                    samples=m['samples'], f1=m['micro_f1'], precision=m['precision'], recall=m['recall'],
                    clean_fpr=m['clean_fpr'], tp=m['tp'], fp=m['fp'], fn=m['fn'],
                    geometry_support=m['per_class']['L2-G']['support'], omission_support=m['per_class']['L3-06']['support'],
                    spurious_support=m['per_class']['L3-07']['support']))
    with (OUT/'tier_comparison.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    emit('selection', **selection)


if __name__ == '__main__':
    import fcntl
    with (OUT/'evaluation.lock').open('w') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        main()
