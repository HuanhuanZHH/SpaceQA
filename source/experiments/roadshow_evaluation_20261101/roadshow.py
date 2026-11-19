"""Read-only roadshow holdout replay; wait for validated inputs, never fit on B."""
import argparse
import collections
import copy
import csv
import hashlib
import importlib.util
import json
import os
import shlex
import socket
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from dispatch import BASE, OUT, REPO, FAMILIES, freeze, identify, lines, sha, write

SOURCE = Path('/staff/tangjiarui/OPC/OPC_路演测试数据/OPC_Roadshow_B_20261105')
PYTHON = REPO.parent / '.venv-image-models/bin/python'
SCRIPT = Path(__file__).resolve()


def status(value, **fields):
    write(OUT / 'state.json', dict(status=value, pid=os.getpid(), time_unix=time.time(), **fields))
    print(json.dumps(dict(status=value, **fields), ensure_ascii=False), flush=True)


def expected_archives():
    result = []
    for line in (SOURCE / 'SHA256SUMS.txt').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
        name = name.lstrip('*')
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts
        result.append((digest, SOURCE / path))
    assert len(result) == 12
    return result


def check_archives():
    cache_path = OUT / 'archive_integrity.json'
    previous = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    cached = {r['archive']: r for r in previous.get('packages', [])}
    checked = []
    for expected, path in expected_archives():
        result = dict(archive=path.name, package=path.stem, family=identify(path.stem),
                      expected_sha256=expected, exists=path.exists(), actual_sha256=None,
                      bytes=0, status='missing')
        if path.exists():
            stat = path.stat(); fingerprint = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
            old = cached.get(path.name, {})
            if old.get('fingerprint') == fingerprint and old.get('expected_sha256') == expected:
                result = old
            else:
                actual = sha(path); after = path.stat()
                changed = [after.st_size, after.st_mtime_ns, after.st_ctime_ns] != fingerprint
                result.update(bytes=stat.st_size, actual_sha256=actual, fingerprint=fingerprint,
                              status='changing' if changed else 'verified' if actual == expected else 'checksum_mismatch')
        checked.append(result)
    report = dict(expected_packages=12, verified=sum(r['status'] == 'verified' for r in checked),
                  missing=sum(r['status'] == 'missing' for r in checked),
                  checksum_mismatch=sum(r['status'] == 'checksum_mismatch' for r in checked),
                  packages=checked, checked_unix=time.time(), raw_data_downloaded_to_windows=False)
    write(cache_path, report)
    return report


def safe_extract(archive, destination, expected_package):
    destination = Path(destination).resolve(); destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:') as stream:
        members = stream.getmembers()
        for member in members:
            parts = Path(member.name).parts
            target = (destination / member.name).resolve()
            if (not parts or Path(member.name).is_absolute() or '..' in parts
                    or not target.is_relative_to(destination)
                    or not (member.isfile() or member.isdir())):
                raise ValueError('Unsafe archive member: ' + member.name)
        stream.extractall(destination, members=members)
    root = destination / expected_package
    assert root.is_dir(), 'Archive root differs from its package name'
    return root


def load_preprocessors():
    """Reuse frozen transforms, redirect their cache globals to this new run."""
    code = REPO / 'experiments/all_family_training_20261031'
    sys.path.insert(0, str(code))
    import common as base_common
    import prepare as allprep
    reference = allprep.reference
    original = sys.modules['common']
    try:
        sys.modules['common'] = reference
        spec = importlib.util.spec_from_file_location('roadshow_original_view_transform', REPO / 'experiments/full_matrix_20261031/prepare.py')
        fullprep = importlib.util.module_from_spec(spec); spec.loader.exec_module(fullprep)
    finally:
        sys.modules['common'] = original
    fullprep.OUT = OUT / 'current_views'
    for name in ['raw_views', 'views']:
        (fullprep.OUT / name).mkdir(parents=True, exist_ok=True)
    return base_common, allprep, fullprep


def pair_labels(root, family):
    root = Path(root)
    labels = sorted(root.rglob('*.json'))
    data = {p.resolve(): p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ['.png', '.csv']}
    answer = []; used = set()
    for label in labels:
        if family == 'AIA_AR_sample':
            expected = root / 'data' / label.name.removesuffix('.json')
        elif family == 'ULF_sample':
            expected = label.with_name(label.name.removesuffix('.json'))
        else:
            expected = label.with_suffix('.png')
        answer.append((label, expected)); used.add(expected.resolve())
    orphans = [p for key, p in data.items() if key not in used]
    return answer, orphans


def prepare_package(item):
    package = item['package']; folder = OUT / 'packages' / package
    marker = folder / 'preparation.json'
    if marker.exists():
        assert json.loads(marker.read_text())['archive_sha256'] == item['expected_sha256']
        return
    started = time.monotonic(); folder.mkdir(parents=True, exist_ok=True)
    archive = SOURCE / '数据' / item['archive']
    assert sha(archive) == item['expected_sha256'], 'Source changed before extraction'
    extraction = OUT / 'extracted' / item['expected_sha256']
    root = safe_extract(archive, extraction, package)
    common, allprep, fullprep = load_preprocessors()
    pairs, orphan_data = pair_labels(root, item['family'])
    rules = fullprep.rules
    # New release uses concept for AIA categories; do not silently ignore it.
    old_category = rules.event_category
    rules.event_category = lambda event: event.get('concept') or old_category(event)
    rules.RULE_VERSION = 'roadshow-adapter-2026.11.01-1'
    dataset = rules.inventory_dataset(root, None)
    checks = set(rules.ALL_CHECKS) - {'C10'}
    rules.base_analysis(dataset, 'B-ROADSHOW-' + item['expected_sha256'][:12], checks)
    rule_rows = []
    for index, finding in enumerate(dataset.findings, 1):
        finding['findingId'] = f"{finding['checkId']}-{item['family']}-{index:06d}"
        finding['evidenceRef'] = 'B-EVD-' + hashlib.sha256((package + finding['_locator'] + str(index)).encode()).hexdigest()[:16]
        # These are engineering observations, not the new release's official scores.
        finding['roadshowScoreEligible'] = 'not_evaluated'
        if finding['checkId'] == 'C06':
            finding['roadshowTrack'] = 'diagnostic_non_scoring_or_requires_criterion'
        else:
            finding['roadshowTrack'] = 'requires_release_scoring_review'
        rule_rows.append(finding)
    with (folder / 'rule_observations.jsonl').open('w') as stream:
        for finding in rule_rows:
            stream.write(json.dumps(finding, ensure_ascii=False) + '\n')
    # Exact current-file rule bits for model comparisons; never counted as B truth.
    bits = collections.defaultdict(lambda: [0, 0, 0])
    for finding in rule_rows:
        if finding['status'] != rules.STATUS_FAIL:
            continue
        code = finding['defectType']
        if code.startswith('GEOMETRY_') or code in ['L2-G', 'L3-MIS']:
            bit = 0
        elif code in ['L3-06', 'L3-07']:
            bit = 1
        elif code in ['L2-P', 'L2-F', 'L2-M', 'L2-T', 'ANNOTATION_STRUCTURE_CONFLICT', 'TIME_INVALID']:
            bit = 2
        else:
            continue
        # Finding locators are stored in the evidence payload on this implementation.
        locator = finding.get('_locator') or finding.get('locator') or finding.get('evidence', {}).get('locator')
        if locator:
            bits[str(locator).split('#')[0]][bit] = 1
    raw_locks = collections.defaultdict(threading.Lock)
    def current(pair):
        label_path, data_path = pair
        identity = package + ':' + label_path.name
        result = dict(id=identity, package=package, family=item['family'], data_path=str(data_path),
                      label_path=str(label_path), rule_bits=bits[str(label_path)],
                      image=None, training_image=None, features=[0.] * 256, input_sha256=None,
                      input_missing=not data_path.exists(), ground_truth_available=False)
        try:
            label = json.loads(label_path.read_text(encoding='utf-8-sig'))
            identify(package, label)
            if not data_path.exists():
                result['status'] = 'data_missing'; return result
            result['data_sha256'] = sha(data_path)
            header = rules.png_header(data_path) if data_path.suffix.lower() == '.png' else {}
            row = dict(id=identity, family=item['family'], dataset_family=item['family'],
                       data_exists=True, data_path=str(data_path), data_sha256=result['data_sha256'],
                       base_sample_id=identity, tier='B_unknown', image_width=header.get('width'), image_height=header.get('height'))
            display_label = copy.deepcopy(label)
            if item['family'] == 'ULF_sample':
                # Roadshow rule: CSV t is offset from basic_annotation.startTime.
                origin = label.get('basic_annotation', {}).get('startTime')
                if origin:
                    display_label.setdefault('science_annotation', {})['dateObs'] = origin
            with raw_locks[result['data_sha256']]:
                view = fullprep.feature_view(display_label, row)
            stats = json.loads((fullprep.OUT / 'raw_views' / (result['data_sha256'] + '.json')).read_text())
            row['features'] = view['features']
            features, prompt = allprep.current_input(row, label, stats)
            small = folder / 'student_views' / (hashlib.sha256(identity.encode()).hexdigest() + '.png')
            small.parent.mkdir(exist_ok=True)
            with Image.open(view['image']) as image:
                image.resize((256, 128), Image.Resampling.BILINEAR).save(small)
            result.update(image=view['image'], training_image=str(small), features=features, prompt=prompt,
                          input_sha256=common.digest([sha(view['image']), prompt, features]), status='prepared')
        except Exception as error:
            result.update(input_missing=True, status='preparation_failed', error=repr(error))
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        prepared = list(pool.map(current, pairs))
    for data in orphan_data:
        prepared.append(dict(id=package + ':orphan:' + data.name, package=package, family=item['family'],
                             data_path=str(data), label_path=None, image=None, training_image=None,
                             features=[0.] * 256, rule_bits=[0, 0, 0], input_sha256=None,
                             input_missing=True, status='label_missing', ground_truth_available=False))
    with (folder / 'inventory.jsonl').open('w') as stream:
        for record in prepared:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    previous = collections.defaultdict(set)
    for record in lines(BASE / 'inventory.jsonl'):
        if record.get('data_sha256'):
            previous[record['data_sha256']].add(record['split'])
    overlap = collections.Counter()
    for record in prepared:
        for split in previous.get(record.get('data_sha256'), []):
            overlap[split] += 1
    write(folder / 'raw_observation_overlap.json', dict(exact_raw_observation_overlap=dict(overlap),
          interpretation='Raw observation overlap does not imply identical annotation input; retain for independence audit'))
    write(marker, dict(status='complete', package=package, family=item['family'],
                       archive_sha256=item['expected_sha256'], records=len(prepared), labels=len(pairs),
                       orphan_data=len(orphan_data), missing_inputs=sum(r['input_missing'] for r in prepared),
                       preparation_failures=sum(r['status'] == 'preparation_failed' for r in prepared),
                       seconds=time.monotonic()-started, ground_truth_available=False,
                       exact_raw_observation_overlap=dict(overlap),
                       forbidden_cross_directory_clean_pairing=True, official_score_computed=False))


def model_replay(package, frozen):
    import torch
    sys.path.insert(0, str(REPO / 'experiments/all_family_training_20261031'))
    from common import ARCHS, SIZE, SPATIAL_SIZE
    from models import build
    from evaluate import CurrentInputs
    from torch.utils.data import DataLoader
    folder = OUT / 'packages' / package
    records = list(lines(folder / 'inventory.jsonl')); present = [r for r in records if not r['input_missing']]
    for route in frozen['experts']:
        marker = folder / 'models' / (route + '.complete.json')
        if marker.exists():
            continue
        started = time.monotonic(); marker.parent.mkdir(exist_ok=True)
        probabilities = {}
        if route == 'qlora':
            from qwen import load, predict
            bundle = load(BASE / 'checkpoints/qlora', False)
            for off in range(0, len(present), 32):
                batch = present[off:off+32]
                pp = predict(bundle, batch, 32)
                probabilities.update({r['id']: p.tolist() for r, p in zip(batch, pp)})
            del bundle
        else:
            checkpoint = BASE / 'checkpoints' / (route + '.pt')
            assert sha(checkpoint) == frozen['checkpoint_hashes'][str(checkpoint.relative_to(BASE))]
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model = build(route, pretrained=False, widths=saved.get('widths'), hidden=saved.get('hidden')).cuda().eval()
            model.load_state_dict(saved['state']); mean=saved['mean'].cuda(); std=saved['std'].cuda()
            loader = DataLoader(CurrentInputs(present, SPATIAL_SIZE if route in ARCHS else SIZE),
                                batch_size=16 if route in ARCHS else 64, num_workers=2, pin_memory=True, shuffle=False)
            with torch.inference_mode():
                for image, features, indices in loader:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits, _ = model(image.cuda(), (features.cuda()-mean)/std)
                    pp = logits.float().sigmoid().cpu().numpy()
                    assert np.isfinite(pp).all()
                    probabilities.update({present[i]['id']: p.tolist() for i, p in zip(indices.tolist(), pp)})
            del model, mean, std, saved
        torch.cuda.synchronize(); elapsed=time.monotonic()-started; torch.cuda.empty_cache()
        thresholds = frozen['model_details'][route]['thresholds'][FAMILIES.index(records[0]['family'])]
        output = folder / 'models' / (route + '.jsonl'); temporary=output.with_suffix('.jsonl.tmp')
        candidates=collections.Counter()
        with temporary.open('w') as stream:
            for r in records:
                p = probabilities.get(r['id'])
                candidate = [int(a >= b) for a, b in zip(p, thresholds)] if p is not None else None
                if candidate is not None:
                    for name, bit in zip(['G', 'C', 'D'], candidate):
                        candidates[name] += bit
                stream.write(json.dumps(dict(id=r['id'], probabilities=p, candidate_bits=candidate,
                    rule_bits=r['rule_bits'], status='success' if p is not None else r['status'],
                    verdict='undetermined', reason='Model candidate without independent roadshow truth; not automatically scored')) + '\n')
        temporary.replace(output)
        write(marker, dict(status='complete', package=package, route=route, records=len(records),
            inferred=len(probabilities), candidate_counts=dict(candidates), model_load_and_inference_seconds=elapsed,
            hardware=socket.gethostname()+' A100; not official A10 acceptance',
            F1=None, precision=None, recall=None, ground_truth_available=False,
            prediction_sha256=sha(output), base_frozen_stage=frozen['stage']))


def worker(package, gpu, dispatch_path):
    sys.path.insert(0, str(REPO / 'experiments/all_family_training_20261031'))
    from scheduling import lock, unlock, available_gpu
    held = lock('roadshow_' + package)
    if held is None:
        return
    gpu_lock = lock('gpu_' + gpu); slot = None
    try:
        if gpu_lock is None:
            return
        for i in range(2):
            slot = lock('helper_slot_' + str(i))
            if slot is not None:
                break
        if slot is None:
            return
        for _ in range(3):
            if not available_gpu(gpu):
                return
            time.sleep(2)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        os.environ['OMP_NUM_THREADS'] = '4'
        frozen = json.loads(Path(dispatch_path).read_text())
        write(OUT / 'packages' / package / 'worker.json',
              dict(status='running', host=socket.gethostname(), gpu=gpu, pid=os.getpid(), stage=frozen['stage']))
        model_replay(package, frozen)
        write(OUT / 'packages' / package / 'worker.json', dict(status='complete', stage=frozen['stage']))
    finally:
        unlock(slot); unlock(gpu_lock); unlock(held)


def free_devices():
    result=[]
    for host in ['icarus1', 'icarus2', 'icarus3']:
        try:
            text=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,
                'nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits'],text=True,timeout=15)
            for line in text.splitlines():
                uuid,memory=[s.strip() for s in line.split(',')]
                if float(memory)<512:
                    result.append((host,uuid))
        except (OSError,subprocess.SubprocessError):
            continue
    return result


def summarize(frozen):
    summaries=[]
    for marker in sorted((OUT/'packages').glob('*/models/*.complete.json')):
        item=json.loads(marker.read_text())
        family=identify(item['package'])
        summaries.append(dict(dataset=family, package=item['package'], model=item['route'],
            selected_by_original_A_val=int(frozen['selected'][family]['route']==item['route']),
            sample_count=item['records'], inferred=item['inferred'],
            G_candidates=item['candidate_counts'].get('G',0), C_candidates=item['candidate_counts'].get('C',0),
            D_candidates=item['candidate_counts'].get('D',0), F1=None, precision=None, recall=None,
            reason='No independent B defect truth', model_load_and_inference_seconds=item['model_load_and_inference_seconds']))
    if summaries:
        target=OUT/'roadshow_model_summary.csv';temporary=target.with_suffix('.csv.tmp')
        with temporary.open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
        temporary.replace(target)


def watch():
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    single=(OUT/'watcher.lock').open('a+');fcntl.flock(single,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(OUT/'registered.json',dict(pid=os.getpid(),code_sha256=sha(SCRIPT),
        mode='known-family static dispatch; B replay only; no route training',
        use_B_for_training_or_selection=False, deadline_hours=72))
    deadline=time.monotonic()+72*3600; jobs={}; frozen=None
    while time.monotonic()<deadline:
        progress=json.loads((BASE/'progress/status.json').read_text())
        current=OUT/'dispatch_current.json'
        if frozen is None or len(progress['evaluated_routes'])!=len(frozen['experts']):
            frozen=json.loads(current.read_text()) if current.exists() else None
            if frozen is None or len(progress['evaluated_routes'])!=len(frozen['experts']):
                frozen=freeze()
        integrity=check_archives()
        verified=[r for r in integrity['packages'] if r['status']=='verified']
        for item in verified:
            try:
                prepare_package(item)
            except Exception as error:
                write(OUT/'packages'/item['package']/'failure.json',dict(status='failed',stage='preparation',error=repr(error)))
        for name,job in list(jobs.items()):
            code=job['process'].poll()
            if code is not None:
                job['log'].close();del jobs[name]
        available=None
        for item in verified:
            package=item['package']; folder=OUT/'packages'/package
            if not (folder/'preparation.json').exists() or package in jobs:
                continue
            if all((folder/'models'/(r+'.complete.json')).exists() for r in frozen['experts']):
                continue
            if len(jobs)>=2:
                break
            if available is None:
                available=free_devices()
            used={job['device'] for job in jobs.values()}
            candidate=next((device for device in available if device not in used),None)
            if candidate is None:
                break
            host,gpu=candidate; log=(folder/'worker.log').open('a')
            dispatch_path=OUT/'dispatch'/f'dispatch_{frozen["stage"]}.json'
            command=shlex.join([str(PYTHON),'-u',str(SCRIPT),'worker','--package',package,'--gpu',gpu,'--dispatch',str(dispatch_path)])
            process=subprocess.Popen(['ssh','-o','BatchMode=yes',host,command],stdout=log,stderr=subprocess.STDOUT)
            jobs[package]=dict(process=process,device=candidate,log=log)
        completed=sum(all((OUT/'packages'/r['package']/'models'/(m+'.complete.json')).exists()
                          for m in frozen['experts']) for r in verified)
        summarize(frozen)
        status('waiting_for_valid_roadshow_archives' if integrity['verified']<12 else 'evaluating_roadshow',
               verified_packages=integrity['verified'], missing_packages=integrity['missing'],
               checksum_mismatch=integrity['checksum_mismatch'], evaluated_packages=completed,
               compared_models=len(frozen['experts']), active_workers=len(jobs),
               ground_truth_available=False, F1=None, router_training=False)
        if completed==12 and len(frozen['experts'])==20:
            status('replay_complete_without_ground_truth', evaluated_packages=12, compared_models=20,
                   F1=None, reason='Independent verified scientific truth required for F1; all model candidates remain unscored')
            return
        time.sleep(60)
    status('wait_budget_exceeded', reason='72-hour data/model readiness budget ended')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['inspect','watch','worker'])
    parser.add_argument('--package');parser.add_argument('--gpu');parser.add_argument('--dispatch')
    args=parser.parse_args()
    if args.mode=='inspect':
        report=check_archives();print(json.dumps({k:v for k,v in report.items() if k!='packages'}))
    elif args.mode=='worker':
        worker(args.package,args.gpu,args.dispatch)
    else:
        try:
            watch()
        except BaseException as error:
            status('failed',error=repr(error));raise
