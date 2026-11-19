"""Independent bookkeeping, metric arithmetic, split and artifact audit."""
import hashlib,json,math
from pathlib import Path
from collections import Counter
import torch
from learn_model_comparison import OUT,ROOT,load,write


def main():
    train,val,test=load();rows=[json.loads(s) for s in (OUT/'dataset.jsonl').open()]
    extra=[json.loads(s) for s in (OUT/'extended_test.jsonl').open()]
    groups={s:{r['split_group'] for r in rows if r['split']==s} for s in ['train','val','test']}
    checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not value:raise AssertionError(name)
    check('group_disjoint',not(groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test']))
    check('training_balanced_400',Counter(r['target'] for r in train)=={i:100 for i in range(4)})
    check('primary_test_only_real_P1',all(r['source']=='public_P0_assisted_real_P1' for r in test))
    check('extended_only_heldout_groups',all(r['split_group'] in groups['test'] for r in extra))
    expected={r['id'] for r in train}
    for path in OUT.glob('student_*seed*.json'):
        data=json.loads(path.read_text())
        check(path.stem+'_train_ids',set(data['training_ids'])==expected and data['train_records']==400)
        for name,value in data.items():
            if not isinstance(value,dict) or 'predictions' not in value:continue
            pred=value['predictions'];actual=value['metrics']
            check(path.stem+'_'+name+'_coverage',all(r['id'] in pred for r in test))
            tp=sum(r['target']>0 and pred[r['id']]==r['target'] for r in test)
            fp=sum(pred[r['id']]>0 and pred[r['id']]!=r['target'] for r in test)
            fn=sum(r['target']>0 and pred[r['id']]!=r['target'] for r in test)
            check(path.stem+'_'+name+'_f1',abs(actual['defect_f1']-2*tp/max(2*tp+fp+fn,1))<1e-12)
    for path in OUT.glob('student_*_seed*.pt'):
        data=torch.load(path,map_location='cpu',weights_only=False)
        count=sum(v.numel() for v in data['state'].values())
        check(path.stem+'_actual_dimensions',count in (619332,155556))
        check(path.stem+'_actual_channels',data['widths'] in ([32,64,128],[16,32,64]))
    training=json.loads((OUT/'qlora_training.json').read_text())
    check('qlora_finished_80_updates',training['steps']==80 and len(training['trace'])==80 and training['trace'][-1]['step']==80)
    check('qlora_finite_loss',all(math.isfinite(r['loss']) for r in training['trace']))
    after=json.loads((OUT/'qlora_after.json').read_text());before=json.loads((OUT/'qlora_before.json').read_text())
    check('qlora_before_and_after_same_primary_support',before['metrics']['n']==after['metrics']['n']==len(test))
    check('qlora_before_and_after_same_stress_support',before['extended_metrics']['n']==after['extended_metrics']['n']==len(extra))
    check('qlora_complete_primary_and_stress',all(r['id'] in before['predictions'] and r['id'] in after['predictions'] for r in test+extra))
    for name,n in [('teacher_predictions.jsonl',1065),('teacher_finetuned_predictions.jsonl',1508),('teacher_extended_predictions.jsonl',443)]:
        data=[json.loads(s) for s in (OUT/name).open()]
        check(name+'_coverage',len(data)==n and len({r['id'] for r in data})==n and all(r['success'] for r in data))
    proof=[json.loads(s) for s in (OUT/'evidence_review_predictions.jsonl').open()]
    check('evidence_review_complete',len(proof)==len(val)+len(test)+len(extra) and len({r['id'] for r in proof})==len(proof))
    runtime=OUT.parent/'model-comparison-20261031'
    for tag in ['seq2_review_unique','seq32_review_unique']:
        data=[json.loads(s) for s in (runtime/f'batch_{tag}.jsonl').open()]
        check(tag+'_all_levels_twice',len(data)==12 and all(r['success']==r['requests']==32 for r in data))
    control=[json.loads(s) for s in (runtime/'batch_seq32_mem35_review_control.jsonl').open()]
    reference=[json.loads(s) for s in (runtime/'batch_seq32_review_unique.jsonl').open() if json.loads(s)['concurrency']==32]
    check('memory_control_all_64_success',len(control)==2 and all(r['success']==r['requests']==32 for r in control))
    check('memory_control_same_case_cohorts',[[r['id'] for r in x['results']] for x in control]==[[r['id'] for r in x['results']] for x in reference])
    cases={r['id']:r for r in map(json.loads,(runtime/'dataset.jsonl').open())}
    data=[json.loads(s) for s in (runtime/'batch_seq32_review_unique.jsonl').open()]
    ids=[r['id'] for group in data for r in group['results']]
    hashes={hashlib.sha256(Path(cases[i]['image']).read_bytes()).hexdigest() for i in ids}
    runtime_uniqueness=dict(case_ids=len(ids),unique_case_ids=len(set(ids)),unique_image_hashes=len(hashes),
                            note='Default server caches enabled; this is not a strict cold-cache test.')
    artifacts={}
    for name in ['dataset.jsonl','extended_test.jsonl','qlora_adapter/adapter_model.safetensors','qlora_adapter/adapter_config.json','comparison_summary.json']:
        path=OUT/name;artifacts[name]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write(OUT/'experiment_validation.json',dict(status='pass',check_count=len(checks),checks=checks,
          groups={k:len(v) for k,v in groups.items()},runtime_uniqueness=runtime_uniqueness,artifacts=artifacts,
          limitations=['No official hidden instance truth','Calendar-day event proxy','No official location/severity/ranking evaluation']))
    print('validated',len(checks),'checks',runtime_uniqueness)

if __name__=='__main__':main()
