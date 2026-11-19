"""Replay a frozen CPU tree model on roadshow current fields; no B truth or fit."""
import collections
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np

from train_gbdt import FAMILIES, GROUPS, ROUTE, features, probabilities, sha, write, emit


def main():
    root = Path('/staff/tangjiarui/OPC/inference_prior_20261102/gbdt')
    source = Path('/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101/packages')
    complete = json.loads((root / 'complete.json').read_text())
    training = json.loads((root / 'training.json').read_text())
    barrier = json.loads((root / 'training_barrier.json').read_text())
    assert complete['status'] == 'complete'
    assert complete['training_barrier_sha256'] == sha(root / 'training_barrier.json')
    assert barrier['training_sha256'] == sha(root / 'training.json')
    assert barrier['vectorizer_sha256'] == sha(root / 'vectorizer.pkl')
    models = []
    for index, group in enumerate(GROUPS):
        path = root / ('model_' + group + '.pkl')
        assert sha(path) == training['heads'][index]['model_sha256']
        models.append(pickle.loads(path.read_bytes()))
    vectorizer = pickle.loads((root / 'vectorizer.pkl').read_bytes())
    thresholds = np.asarray(training['thresholds'])
    reports = []
    start = time.monotonic()
    for inventory in sorted(source.glob('*/inventory.jsonl')):
        package_start = time.monotonic()
        package = inventory.parent.name
        marker = root / 'roadshow_predictions' / (package + '.complete.json')
        if marker.exists():
            previous = json.loads(marker.read_text())
            prediction = marker.with_name(package + '.jsonl')
            assert sha(prediction) == previous['prediction_sha256']
            assert sha(inventory) == previous['inventory_sha256']
            assert sha(root / 'training.json') == previous['frozen_training_sha256']
            reports.append(previous)
            continue
        write(root / 'roadshow_state.json', dict(status='replaying', package=package, pid=os.getpid()))
        before = sha(inventory)
        records = [json.loads(line) for line in inventory.open()]
        ff = []
        present = []
        for index, record in enumerate(records):
            if record['input_missing']:
                continue
            # Roadshow packages currently contain AIA/EPB/ULF. The existing
            # summary fully defines the permitted inputs; no labels are fitted.
            ff.append(features(record))
            present.append(index)
        matrix = vectorizer.transform(ff)
        pp = probabilities(models, matrix)
        probabilities_by_index = dict(zip(present, pp))
        prediction = root / 'roadshow_predictions' / (package + '.jsonl')
        prediction.parent.mkdir(exist_ok=True)
        counts = collections.Counter()
        with prediction.open('w', encoding='utf-8') as stream:
            for index, record in enumerate(records):
                p = probabilities_by_index.get(index)
                bits = (p >= thresholds[FAMILIES.index(record['family'])]).astype(int).tolist() if p is not None else None
                if bits:
                    counts.update({name:int(value) for name,value in zip(GROUPS,bits)})
                stream.write(json.dumps(dict(id=record['id'], input_sha256=record['input_sha256'],
                    probabilities=p.tolist() if p is not None else None, candidate_bits=bits,
                    status='success' if p is not None else record.get('status','input_missing'),
                    verdict='undetermined', ground_truth_available=False,
                    reason='Frozen A-trained model candidate only; roadshow has no independent defect truth')) + '\n')
        assert before == sha(inventory)
        report = dict(status='complete', package=package, route=ROUTE, records=len(records), inferred=len(present),
            candidate_counts=dict(counts), inventory_sha256=before, prediction_sha256=sha(prediction),
            frozen_training_sha256=sha(root / 'training.json'), ground_truth_available=False,
            F1=None, precision=None, recall=None, seconds=time.monotonic()-package_start,
            use_B_for_training_or_thresholds=False, GPU_used=False)
        write(marker, report)
        reports.append(report)
        emit('roadshow_package_complete', **report)
    assert len(reports) == 12
    final = dict(status='complete_without_ground_truth', route=ROUTE, packages=12,
        records=sum(r['records'] for r in reports), inferred=sum(r['inferred'] for r in reports),
        package_reports=reports, elapsed_seconds=time.monotonic()-start,
        frozen_training_sha256=sha(root / 'training.json'), source_sha256=sha(__file__),
        B_fit_or_calibration=False, F1=None, precision=None, recall=None, GPU_used=False)
    write(root / 'roadshow_complete.json', final)
    write(root / 'roadshow_state.json', final)
    emit('roadshow_complete', **final)


if __name__ == '__main__':
    main()
