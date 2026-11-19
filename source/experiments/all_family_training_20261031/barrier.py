"""Independently enforce all-route/all-family coverage before opening test inputs."""
import argparse
import json
from common import *


def build_barrier():
    code_manifest=json.loads((OUT/'code_manifest.json').read_text())
    assert all(sha(CODE/name)==value for name,value in code_manifest.items()),'Training source changed after queue launch'
    prep=json.loads((OUT/'preparation.json').read_text());assert prep['status']=='complete'
    train=[r for r in lines(OUT/'train.jsonl') if not r['input_missing'] and any(r['truth_valid'])]
    expected={r['id'] for r in train};coverage=assert_coverage(train)
    pretraining=json.loads((OUT/'pretraining/complete.json').read_text())
    pretrain_scope=lines(OUT/'pretraining/train_scope.jsonl')
    assert pretraining['status']=='complete' and pretraining['test_read'] is False
    assert pretraining['train_ids_sha256']==digest(sorted(expected))
    assert {r['id'] for r in pretrain_scope}==expected
    assert pretraining['checkpoint_sha256']==sha(OUT/'pretraining/raw_encoder.pt')
    assert pretraining['train_scope_sha256']==sha(OUT/'pretraining/train_scope.jsonl')
    frozen={};summaries={}
    for route in ROUTES:
        report=json.loads((OUT/'training'/(route+'.json')).read_text())
        assert report['status']=='complete' and report['test_read'] is False,route
        if route.startswith(('supervised','distilled')):assert report['encoder_initialization_sha256']==pretraining['checkpoint_sha256']
        assert report['train_split_sha256']==prep['split_hashes']['train'],route
        assert report['val_split_sha256']==prep['split_hashes']['val'],route
        assert report['training_records']==len(expected) and report['unique_records_updated']==len(expected),route
        assert report['training_ids_sha256']==digest(sorted(expected)),route
        actual=report['actual_train_coverage']
        assert all(actual[k]>=n for k,n in coverage.items()) and len(actual)==30,route
        ids={r['id'] for r in lines(OUT/'training'/(route+'.updated_ids.jsonl'))}
        assert ids==expected,route
        assert report['epochs']>=1 and report['selected_epoch']>=1,route
        for entry in report['trace']:
            assert entry['unique_train_records']==len(expected),route
            assert all(entry['actual_train_coverage'][k]==n for k,n in coverage.items()),route
        hashes={str(p.relative_to(OUT)):sha(p) for p in checkpoint_files(route)}
        assert hashes==report['checkpoint_hashes'],route
        frozen.update(hashes);summaries[route]=dict(report_sha256=sha(OUT/'training'/(route+'.json')),families=6,packages=30,records=len(expected))
    teacher=json.loads((OUT/'teacher_train.json').read_text())
    assert teacher['records']==len(expected) and teacher['split']=='train'
    assert teacher['soft_targets_sha256']==sha(OUT/'teacher_train.jsonl')
    assert {r['id'] for r in lines(OUT/'teacher_train.jsonl')}==expected
    # Validate the split partition before any actual test data are exposed to models.
    sets={s:{r['training_group_id'] for r in lines(OUT/(s+'.jsonl'))} for s in ['train','val','test']}
    assert not(sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test'])
    result=dict(status='pass',all_routes_complete=True,routes=summaries,checkpoint_hashes=frozen,
        split_hashes=prep['split_hashes'],protocol_sha256=sha(CODE/'protocol.md'),
        code_manifest=code_manifest,
        created_at_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        train_records=len(expected),test_predictions_started=bool(list((OUT/'predictions').glob('*.complete.json'))),
        evaluation_policy='each_model_after_its_all_family_training')
    write(OUT/'training_barrier.json',result);emit('training_barrier',status='pass',routes=len(ROUTES),families=6,packages=30)
    return result


def build_route_barrier(route):
    assert route in ROUTES,route
    from scheduling import verify_code
    verify_code();code_manifest=json.loads((OUT/'code_manifest.json').read_text())
    prep=json.loads((OUT/'preparation.json').read_text());assert prep['status']=='complete'
    assert json.loads((OUT/'prepared_validation.json').read_text())['status']=='pass'
    report_path=OUT/'training'/(route+'.json')
    if not report_path.exists():raise RuntimeError('Model evaluation is locked: training is incomplete for '+route)
    report=json.loads(report_path.read_text())
    assert report['status']=='complete' and report['test_read'] is False,route
    train=[r for r in lines(OUT/'train.jsonl') if not r['input_missing'] and any(r['truth_valid'])]
    expected={r['id'] for r in train};coverage=assert_coverage(train)
    assert report['training_records']==report['unique_records_updated']==len(expected)
    assert report['training_ids_sha256']==digest(sorted(expected))
    assert {r['id'] for r in lines(OUT/'training'/(route+'.updated_ids.jsonl'))}==expected
    assert report['train_split_sha256']==prep['split_hashes']['train']
    assert report['val_split_sha256']==prep['split_hashes']['val']
    assert report['epochs']==len(report['trace']) and 1<=report['selected_epoch']<=report['epochs']
    for entry in report['trace']:
        assert entry['unique_train_records']==len(expected)
        assert entry['actual_train_coverage']==coverage
    assert report['actual_train_coverage']=={k:n*report['epochs'] for k,n in coverage.items()}
    dependencies={}
    if route.startswith(('supervised','distilled')):
        p=json.loads((OUT/'pretraining/complete.json').read_text())
        assert p['status']=='complete' and p['test_read'] is False
        assert p['train_ids_sha256']==digest(sorted(expected))
        assert report['encoder_initialization_sha256']==p['checkpoint_sha256']==sha(OUT/'pretraining/raw_encoder.pt')
        dependencies['pretraining/complete.json']=sha(OUT/'pretraining/complete.json')
    if route.startswith('distilled'):
        t=json.loads((OUT/'teacher_train.json').read_text())
        assert t['status']=='complete' and t['split']=='train' and t['records']==len(expected)
        assert t['split_hash']==prep['split_hashes']['train']
        assert t['soft_targets_sha256']==sha(OUT/'teacher_train.jsonl')
        assert {r['id'] for r in lines(OUT/'teacher_train.jsonl')}==expected
        dependencies['teacher_train.json']=sha(OUT/'teacher_train.json')
    hashes={str(p.relative_to(OUT)):sha(p) for p in checkpoint_files(route)}
    assert hashes==report['checkpoint_hashes'],route
    for split,expected_hash in prep['split_hashes'].items():assert sha(OUT/(split+'.jsonl'))==expected_hash
    result=dict(status='pass',route=route,route_complete=True,all_routes_complete=False,
        routes={route:dict(report_sha256=sha(report_path),families=6,packages=30,records=len(expected))},
        checkpoint_hashes=hashes,split_hashes=prep['split_hashes'],protocol_sha256=sha(CODE/'protocol.md'),
        code_manifest=code_manifest,dependency_hashes=dependencies,train_records=len(expected),
        created_at_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    write(OUT/'route_barriers'/(route+'.json'),result)
    emit('route_training_barrier',route=route,status='pass',families=6,packages=30)
    return result


def verify_barrier(route=None):
    path=OUT/'route_barriers'/(route+'.json') if route else OUT/'training_barrier.json'
    if not path.exists():raise RuntimeError('Testing is locked: missing training barrier for '+str(route or 'all routes'))
    result=json.loads(path.read_text());assert result['status']=='pass'
    if route:
        assert result['route_complete'] and result['route']==route and set(result['routes'])=={route}
    else:
        assert result['all_routes_complete'] and set(result['routes'])==set(ROUTES)
    assert result['protocol_sha256']==sha(CODE/'protocol.md')
    assert all(sha(CODE/name)==value for name,value in result['code_manifest'].items())
    for name,expected in result['checkpoint_hashes'].items():assert sha(OUT/name)==expected,name
    for split,expected in result['split_hashes'].items():assert sha(OUT/(split+'.jsonl'))==expected,split
    for route,report in result['routes'].items():assert sha(OUT/'training'/(route+'.json'))==report['report_sha256'],route
    for name,expected in result.get('dependency_hashes',{}).items():assert sha(OUT/name)==expected,name
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--route');a=p.parse_args()
    if a.route:verify_barrier(a.route) if a.verify else build_route_barrier(a.route)
    else:verify_barrier() if a.verify else build_barrier()
