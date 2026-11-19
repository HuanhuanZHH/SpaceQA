"""Frozen epoch-one adapters x inference guidance, balanced paired pilot.

The original two-epoch training continues elsewhere. No weights, thresholds,
production services, or source records are changed by this evaluation.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/staff/tangjiarui/OPC')
BASE = ROOT / 'opc-agent/runs/all-family-training-20261031'
TRAINING = ROOT / 'jwvl_optimization_20261104'
FAMILIES = ['AIA_AR_sample', 'HMI_AR_sample', 'CME_C2_sample',
            'EPB_seg_sample', 'DMSP_SSUSI', 'ULF_sample']
TIERS = ['P0', 'P1', 'P2', 'P3', 'P4']
ARMS = [('control_off', 'control', False), ('control_on', 'control', True),
        ('domain_off', 'domain', False), ('domain_on', 'domain', True)]
SEED = 'domain-pilot-20261105-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fp:
        for block in iter(lambda: fp.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as fp:
        return [json.loads(line) for line in fp if line.strip()]


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')
    temp.replace(path)


def progress(out, **values):
    values.update(updated_at=dt.datetime.now().astimezone().isoformat(),
                  pid=os.getpid(), host=os.uname().nodename)
    write(out / 'state.json', values)
    print(json.dumps(values, ensure_ascii=False), flush=True)


def prepare(out, groups_per_family):
    out.mkdir(parents=True, exist_ok=False)
    (out / 'snapshots').mkdir()
    configs = {variant: read(TRAINING / f'qwen8b_{variant}_v1/config.json')
               for variant in ['control', 'domain']}
    assert {k: v for k, v in configs['control'].items() if k != 'variant'} == {
        k: v for k, v in configs['domain'].items() if k != 'variant'}
    progress(out, status='preparing', target_groups_per_family=groups_per_family)
    training = {}
    for variant in ['control', 'domain']:
        origin = TRAINING / f'qwen8b_{variant}_v1'
        state = read(origin / 'state.json')
        assert state.get('epoch', 0) == 2 and state['status'] == 'training', state
        # While epoch two is still training, best_adapter contains the completed
        # epoch-one checkpoint. Capture its epoch-complete receipt as evidence.
        receipts = []
        with (TRAINING / f'qwen8b_{variant}_v1.log').open() as fp:
            for line in fp:
                try:
                    record = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if record.get('status') == 'epoch_complete':
                    receipts.append(record)
        assert len(receipts) == 1 and receipts[0]['epoch'] == 1, receipts
        source = origin / 'best_adapter'
        original_hash = sha(source / 'adapter_model.safetensors')
        destination = out / 'snapshots' / variant
        shutil.copytree(source, destination)
        assert sha(destination / 'adapter_model.safetensors') == original_hash
        assert sha(source / 'adapter_model.safetensors') == original_hash
        training[variant] = dict(config=configs[variant], state_at_snapshot=state,
                                 completed_epoch_receipt=receipts[0],
                                 completed_continuation_epochs=1,
                                 snapshot_adapter_sha256=original_hash)
    source = BASE / 'test.jsonl'
    assert sha(source) == configs['control']['split_hashes']['test']
    all_rows = rows(source)
    groups = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in all_rows:
        if r['input_missing'] or not any(r['truth_valid']):
            continue
        if not Path(r['image']).is_file():
            continue
        groups[(r['family'], r['training_group_id'])][r['base_sample_id']][r['tier']] = r
    eligible = collections.defaultdict(list)
    for (family, group_id), bases in groups.items():
        complete = [base for base, versions in bases.items() if set(versions) == set(TIERS)]
        if complete:
            # One representative base per source group, chosen without labels.
            chosen = min(complete, key=lambda b: hashlib.sha256((SEED + b).encode()).hexdigest())
            eligible[family].append((group_id, bases[chosen]))
    selected = []
    for family in FAMILIES:
        candidates = sorted(eligible[family], key=lambda item:
                            hashlib.sha256((SEED + item[0]).encode()).hexdigest())
        assert len(candidates) >= groups_per_family
        for _, versions in candidates[:groups_per_family]:
            selected.extend(versions[tier] for tier in TIERS)
    ids = {r['id'] for r in selected}
    assert len(ids) == groups_per_family * 30
    assert set(collections.Counter(r['family'] + '|' + r['tier'] for r in selected).values()) == {groups_per_family}
    selected_groups = {r['training_group_id'] for r in selected}
    for split in ['train', 'val']:
        assert sha(BASE / (split + '.jsonl')) == configs['control']['split_hashes'][split]
        assert not selected_groups.intersection(r['training_group_id'] for r in rows(BASE / (split + '.jsonl')))
    with (out / 'selected_samples_PRIVATE.jsonl').open('w') as fp:
        for r in selected:
            fp.write(json.dumps(r, ensure_ascii=False) + '\n')
    provenance = [dict(id=r['id'], family=r['family'], tier=r['tier'],
                       training_group_id=r['training_group_id'],
                       image_path=r['image'], image_sha256=sha(r['image']),
                       input_sha256=r['input_sha256'], data_path=r['data_path'],
                       label_path=r['label_path'], data_sha256=r.get('data_sha256'),
                       label_sha256=r.get('label_sha256')) for r in selected]
    write(out / 'sample_provenance.json', provenance)
    protocol = dict(status='frozen_before_inference', seed=SEED, cases=len(selected),
        arms=[dict(name=name, training_guidance=variant, inference_guidance=guidance)
              for name, variant, guidance in ARMS],
        families=FAMILIES, tiers=TIERS, groups_per_family=groups_per_family,
        threshold=0.5, threshold_policy='Predeclared same 0.5 for all heads/arms; no pilot calibration',
        primary_comparison=['control_off', 'domain_on'],
        secondary_comparisons=[['control_off', 'control_on'], ['domain_off', 'domain_on'],
                               ['control_off', 'domain_off'], ['control_on', 'domain_on']],
        primary_scope='P1-P4 known G/C/D units; P0 sample alert rate reported separately',
        sampling='Outcome-independent hash selection of complete five-tier groups from historical test',
        test_split_sha256=sha(source), selected_rows_sha256=sha(out / 'selected_samples_PRIVATE.jsonl'),
        training=training, max_pixels=configs['control']['max_pixels'], initial_batch=2,
        target_inputs='Only current prompt and current observation/annotation image; no target/tier/P0 difference supplied',
        model='Qwen3-VL-8B-Instruct, NF4, frozen epoch-one QLoRA adapters',
        interpretation='Added explicit domain guidance during continuation training and/or inference; not erasure of learned domain knowledge',
        limitations=['Small intermediate pilot, not final two-epoch or full-test results',
                     'Historical public test, not fresh blind data',
                     'P0-derived G/C/D proxy, not eight-class localization or official critical recall',
                     'Complete-data sampling excludes absent images and unsupported targets',
                     'Both adapters inherited the same already domain-adapted OPC initializer'])
    write(out / 'protocol.json', protocol)
    progress(out, status='ready', cases=len(selected), total_forward_samples=4 * len(selected))


def metric(reference, predictions):
    counts = collections.Counter()
    for r, p in zip(reference, predictions):
        for target, known, probability in zip(r['target'], r['truth_valid'], p):
            if known:
                pred = probability >= .5
                counts['TP' if target and pred else 'FN' if target else 'FP' if pred else 'TN'] += 1
    tp, fp, fn, tn = (counts[key] for key in ['TP', 'FP', 'FN', 'TN'])
    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                FPR=fp/(fp+tn) if fp+tn else None)


def evaluate(out):
    protocol = read(out / 'protocol.json')
    records = rows(out / 'selected_samples_PRIVATE.jsonl')
    output = {}
    for name, _, _ in ARMS:
        predictions = rows(out / (name + '.jsonl'))
        assert [r['id'] for r in predictions] == [r['id'] for r in records]
        probabilities = [r['probabilities'] for r in predictions]
        def subset(indices):
            return metric([records[i] for i in indices], [probabilities[i] for i in indices])
        nonclean = [i for i, r in enumerate(records) if r['tier'] != 'P0']
        clean = [i for i, r in enumerate(records) if r['tier'] == 'P0']
        flagged = sum(any(p >= .5 for p in probabilities[i]) for i in clean)
        packages = []
        for family in FAMILIES:
            for tier in TIERS:
                indices = [i for i, r in enumerate(records) if r['family'] == family and r['tier'] == tier]
                packages.append(dict(family=family, tier=tier, records=len(indices), **subset(indices)))
        output[name] = dict(all_P1_P4=subset(nonclean), all_tiers=subset(range(len(records))),
                            P0_flagged=flagged, P0_samples=len(clean),
                            P0_sample_alert_rate=flagged/len(clean), packages=packages)
    write(out / 'metrics.json', dict(status='complete', protocol_sha256=sha(out/'protocol.json'),
        cases=len(records), arms=output,
        primary_delta_F1_pp=100*(output['domain_on']['all_P1_P4']['F1'] - output['control_off']['all_P1_P4']['F1']),
        interpretation=protocol['interpretation'], limitations=protocol['limitations']))


def run(out):
    protocol = read(out / 'protocol.json')
    protocol_hash = sha(out / 'protocol.json')
    assert protocol['status'] == 'frozen_before_inference'
    assert sha(out / 'selected_samples_PRIVATE.jsonl') == protocol['selected_rows_sha256']
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
    lock = (out.parent / 'icarus0_gpu1_domain_pilot.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    usage = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used',
                            '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    used = {int(line.split(',')[0]): int(line.split(',')[1]) for line in usage.stdout.splitlines()}
    assert used[1] < 512, ('GPU1 no longer idle', used)
    started = time.monotonic()
    progress(out, status='loading', cases=protocol['cases'])
    import numpy as np
    import torch
    from transformers import AutoProcessor
    sys.path.insert(0, str(ROOT / 'opc-agent/experiments/all_family_training_20261031'))
    sys.path.insert(0, str(TRAINING / 'code'))
    import qwen
    from grounding import grounded_prompt
    torch.set_num_threads(4)
    torch.manual_seed(20261105)
    np.random.seed(20261105)
    bundle = qwen.load(out / 'snapshots/control', False)
    model = bundle[1]
    model.load_adapter(str(out / 'snapshots/domain'), adapter_name='domain')
    processor = AutoProcessor.from_pretrained(ROOT / 'models/Qwen3-VL-8B-Instruct',
        min_pixels=28*28*64, max_pixels=protocol['max_pixels'])
    processor.tokenizer.padding_side = 'left'
    bundle = (processor, model, bundle[2])
    records = rows(out / 'selected_samples_PRIVATE.jsonl')
    total = len(records) * len(ARMS)
    timings = []
    completed = 0
    for name, variant, guidance in ARMS:
        model.set_adapter('default' if variant == 'control' else 'domain')
        model.eval()
        inputs = [dict(image=r['image'], prompt=grounded_prompt(r) if guidance else r['prompt']) for r in records]
        assert all(set(r) == {'image', 'prompt'} for r in inputs)
        arm_started = time.monotonic()
        with (out / (name + '.jsonl')).open('w') as fp:
            for offset in range(0, len(records), 2):
                batch = inputs[offset:offset+2]
                pp = qwen.predict(bundle, batch, batch=2)
                assert pp.shape == (len(batch), 3) and np.isfinite(pp).all()
                for r, p in zip(records[offset:offset+2], pp):
                    fp.write(json.dumps(dict(id=r['id'], input_sha256=r['input_sha256'],
                        probabilities=p.tolist(), prediction=(p>=.5).tolist(),
                        inference_prompt_sha256=hashlib.sha256((grounded_prompt(r) if guidance else r['prompt']).encode()).hexdigest())) + '\n')
                fp.flush()
                completed += len(batch)
                elapsed = time.monotonic()-started
                progress(out, status='running', arm=name, arm_completed=offset+len(batch),
                    arm_total=len(records), completed_forward_samples=completed,
                    total_forward_samples=total, elapsed_seconds=elapsed,
                    estimated_remaining_seconds=(total-completed)*elapsed/completed,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated())
        timings.append(dict(arm=name, seconds=time.monotonic()-arm_started))
    assert sha(out / 'protocol.json') == protocol_hash
    for variant in ['control', 'domain']:
        assert sha(out / 'snapshots' / variant / 'adapter_model.safetensors') == protocol['training'][variant]['snapshot_adapter_sha256']
    evaluate(out)
    write(out / 'execution_receipt.json', dict(status='complete', timings=timings,
        total_seconds=time.monotonic()-started, protocol_sha256=protocol_hash,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        gpu_name=torch.cuda.get_device_name(), inference_input_whitelist=['image','prompt']))
    progress(out, status='complete', completed_forward_samples=completed,
             total_forward_samples=total, elapsed_seconds=time.monotonic()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--groups-per-family', type=int, default=8)
    args = parser.parse_args()
    try:
        if args.mode == 'prepare':
            prepare(args.output, args.groups_per_family)
        else:
            run(args.output)
    except BaseException as error:
        if args.output.exists():
            progress(args.output, status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
