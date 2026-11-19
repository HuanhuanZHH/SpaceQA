"""Assemble technical localization proposals; never claim human verification."""
import collections
import csv
import json
from pathlib import Path

ROOT=Path('/staff/tangjiarui/OPC/opc-agent')
OUT=ROOT/'runs/constraint-optimization-20261031'
AUDITS=ROOT.parent/'phase2_visual_audit_xzj'
SOURCES={'AIA_AR_sample':'aia_p0_p1_audit_20261026','HMI_AR_sample':'hmi_p0_p1_audit_20261029',
         'EPB_seg_sample':'epb_p0_p1_audit_20261027','DMSP_SSUSI':'dmsp_p0_p1_audit_20261029'}


def lines(p):
    return [json.loads(s) for s in p.open() if s.strip()]


def main():
    manifest=ROOT.parent/'phase1_manifest_audit_xzj_20261025/outputs/run_20261025T152348Z/manifest.jsonl'
    index={}; groups=collections.defaultdict(dict)
    for r in lines(manifest):
        if r['dataset_family'] in SOURCES and r['tier'] in ('P0','P1'):
            index[r['dataset_family'],r['source_key'],r['tier']]=r
            groups[r['base_sample_id']][r['tier']]=r
    split={r['base_sample_id']:(r['split'],r['split_group']) for r in lines(OUT/'expanded_dataset.jsonl')}
    queue=[]
    for family,sub in SOURCES.items():
        candidates=[r for r in lines(AUDITS/sub/'findings.jsonl') if r['code'] in ('L2-G','L3-06','L3-07')]
        # Small reference-estimated displacements are the priority; these are
        # suggestions to adjudicate, not new ground-truth annotations.
        def magnitude(r):
            m=r.get('metrics') or {}
            return m.get('max_vertex_shift_px',m.get('mean_vertex_shift_px',1e9))
        candidates.sort(key=lambda r:(magnitude(r),r['filename'],r.get('pointer','')))
        for f in candidates:
            actual=index[family,f['filename'],'P1'];clean=index[family,f['filename'],'P0']
            assigned=split.get(actual['base_sample_id'])
            queue.append(dict(family=family,kind='positive_localization_proposal',
                proposed_code=f['code'],base_sample_id=actual['base_sample_id'],
                split_group_id=actual['split_group_id'],frozen_pilot_split=assigned[0] if assigned else None,
                observation_group=assigned[1] if assigned else None,
                recommended_use='review_only_keep_frozen_test' if assigned and assigned[0]=='test' else 'expert_review_before_training',
                raw_path=actual['data_path'],label_path=actual['label_path'],reference_raw_path=clean['data_path'],
                reference_label_path=clean['label_path'],data_sha256=actual['data_sha256'],label_sha256=actual['label_sha256'],
                json_pointer=f.get('pointer'),reference_estimated_displacement_px=None if magnitude(f)==1e9 else magnitude(f),
                evidence_source=str(AUDITS/sub/'findings.jsonl'),technical_evidence=f,
                review_status='pending_expert',expert_code=None,expert_region=None,expert_comment=None,
                truth_status='public_reference_proposal_not_independent_scientific_ground_truth'))
        # Explicitly verified synchronized transforms are important hard
        # negatives. Do not use unregistered image changes as negative truth.
        p=AUDITS/sub/'transform_compensation.jsonl'
        if p.exists(): negative=lines(p)
        else:
            p=AUDITS/sub/'transform_compensation.json'
            negative=json.loads(p.read_text()) if p.exists() else []
        used=set()
        for n in negative:
            name=n.get('filename')
            if not name or name in used or any(f['filename']==name for f in candidates):continue
            used.add(name);actual=index[family,name,'P1'];clean=index[family,name,'P0']
            queue.append(dict(family=family,kind='transform_hard_negative_proposal',proposed_code='NONE_AFTER_ALIGNMENT',
                base_sample_id=actual['base_sample_id'],split_group_id=actual['split_group_id'],
                recommended_use='expert_review_before_training',raw_path=actual['data_path'],label_path=actual['label_path'],
                reference_raw_path=clean['data_path'],reference_label_path=clean['label_path'],
                json_pointer='/hek_event_annotation/matched_events',technical_evidence=n,
                review_status='pending_expert',expert_code=None,expert_region=None,expert_comment=None,
                truth_status='technical_transform_evidence_needs_independent_review'))
            if len(used)>=20:break
    with (OUT/'localization_review_queue.jsonl').open('w') as f:
        for r in queue:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    columns=['family','kind','proposed_code','base_sample_id','split_group_id','frozen_pilot_split','recommended_use',
             'raw_path','label_path','reference_raw_path','reference_label_path','json_pointer','reference_estimated_displacement_px',
             'review_status','expert_code','expert_region','expert_comment']
    with (OUT/'localization_review_queue.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,columns,extrasaction='ignore');w.writeheader();w.writerows(queue)
    summary=dict(records=len(queue),cells=dict(collections.Counter(f"{r['family']}:{r['kind']}:{r['proposed_code']}" for r in queue)),
                 expert_verified_records=0,training_disabled_until_review=True,
                 limitations=['Localization proposals are copied from earlier technical audits, not newly independent labels.',
                              'AIA/HMI still need scientific event grouping before train/validation/test allocation.',
                              'Never add frozen pilot test cases to training.'])
    (OUT/'localization_review_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
