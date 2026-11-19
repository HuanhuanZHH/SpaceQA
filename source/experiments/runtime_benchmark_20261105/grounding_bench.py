"""Fresh independent localization on every readable roadshow RGB image.

Only model candidates are attached. No detector absence is certified as spurious
annotation, and no model regions deduct confirmed technical scores.
"""
from __future__ import annotations
import bench_metrics as bm
bm.script_enter('grounding')
import argparse,collections,concurrent.futures,hashlib,io,itertools,json,math,time
from pathlib import Path
import sys
sys.path.insert(0, '/staff/tangjiarui/OPC/eight_class_20261104/code')
import numpy as np
import localizer as l
from spaceqa_agent import quality_ledger as q


def predict(model,path,family_index,size,pixel,presence,legacy,use_presence=True):
    """Observation and family only. No annotation object enters this function."""
    import torch
    from PIL import Image
    from torch.nn import functional as F
    with Image.open(path) as im:
        width,height=im.size;x=np.array(im.convert('RGB').resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)/255
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        output=model(torch.from_numpy(x.transpose(2,0,1).copy())[None].cuda())
        if legacy:z=output;present=1.
        else:
            z,pr=output;present=float(pr[0,family_index].sigmoid()) if use_presence else 1.
        probability=z[0,family_index].sigmoid().float()
        probability=F.interpolate(probability[None,None],size=(384,384),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
    return probability,(probability>=pixel)&(present>=presence),present,width,height


def predict_batch(model,items,family_index,pixel,presence,legacy,use_presence=True):
    """Only RGB arrays enter the model, never an annotation or a target mask."""
    import torch
    from torch.nn import functional as F
    x=np.stack([item['rgb'] for item in items])
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        output=model(torch.from_numpy(x).cuda())
        if legacy:z=output;present=np.ones(len(items))
        else:
            z,pr=output;present=pr[:,family_index].sigmoid().float().cpu().numpy() if use_presence else np.ones(len(items))
        probability=z[:,family_index].sigmoid().float()
        probability=F.interpolate(probability[:,None],size=(384,384),mode='bilinear',align_corners=False)[:,0].cpu().numpy()
    return probability,(probability>=pixel)&(present[:,None,None]>=presence),present


def candidates(row,ledger,signals,components,geometry,errors,h,checkpoint):
    from PIL import Image
    codes=set(signals['model_candidates']);size=384;width,height=ledger['imageSize']
    matched=h.match_instances(components,geometry,width,height,size=size)
    by_index={g['index']:g for g in geometry};result=[]
    def add(code,box,pointer,kind,measurements,severity,confidence):
        fine_probability=signals['probabilities'][signals['classes'].index(code)]
        trigger=signals.get('investigation_triggers',{}).get(code)
        group_probability=trigger['group_probability'] if trigger else None
        risk=max(fine_probability,group_probability) if trigger else fine_probability
        locator=dict(type='imageRegion',path=ledger['dataPath'],data_path=ledger['dataPath'],bbox=box,
            coordinate_system='image_pixels',origin='top_left',axis_order='xy',image_size=[width,height],
            frame_id=row['data_sha256'],annotation_pointer=pointer)
        identity=q.digest([row['id'],code,pointer,box])[:20]
        result.append(dict(findingId='NEURAL-'+identity,checkId=q.CHECKS[code],ruleVersion='eight-localizer-20261104',
            dirId=row['package'],sampleId=row['id'],defectType=code,sourceKind='independent_RGB_model',
            detail=kind+'；原始图像预测与当前标注对比，尚未获独立科学确认',locator=locator,
            severity=severity,measurements=measurements,confidence=confidence,
            fusion_class_probability=fine_probability,
            fusion_group_probability=group_probability,
            candidate_trigger=trigger['source'] if trigger else 'fine_classifier',
            risk_weight_source='GCD-C fallback' if trigger and group_probability>fine_probability else 'fine_classifier',
            weighted_burden=risk,
            checkpoint_sha256=checkpoint,scoreEligible=False,findingStage='candidate',verdict='undetermined',
            evidence=dict(label_path=row['label_path'],data_path=row['data_path'],data_sha256=row['data_sha256'],label_sha256=ledger['labelSha256'])))
    for match in matched['matches']:
        g=by_index[match['annotation_index']];component=components[match['component']]
        a=h.region_mask([g],width,height,size).astype(bool);r=component['mask']
        inter=int((a&r).sum());aa=int(a.sum());ra=int(r.sum())
        if not aa or not ra:continue
        cov_r=inter/ra;cov_a=inter/aa;ratio=aa/ra
        ay,ax=np.where(a);ry,rx=np.where(r)
        distance=math.hypot((ax.mean()-rx.mean())/size*width,(ay.mean()-ry.mean())/size*height)/math.hypot(width,height)
        box=[min(p[0] for p in g['points']),min(p[1] for p in g['points']),max(p[0] for p in g['points']),max(p[1] for p in g['points'])]
        measurements=dict(mask_iou=match['iou'],coverage_reference=cov_r,coverage_annotation=cov_a,area_ratio=ratio,normalized_distance=distance)
        if 'L3-MIS' in codes and ((cov_r<.9 and cov_a>=.9 and ratio<.9) or (cov_a<.9 and cov_r>=.9 and ratio>1.1)):
            under=ratio<1;value=1-cov_r if under else 1-cov_a
            add('L3-MIS',box,g['pointer'],'undercoverage' if under else 'overcoverage',measurements,
                dict(metric='uncovered_reference_fraction' if under else 'excess_annotation_fraction',value=value,unit='ratio'),component['confidence'])
        if 'L2-G' in codes and (match['iou']<.75 or distance>.05):
            add('L2-G',box,g['pointer'],'boundary_position_deviation',measurements,
                dict(metric='one_minus_mask_iou',value=1-match['iou'],unit='ratio'),component['confidence'])
    if 'L3-06' in codes and not errors:
        for k in matched['possible_omission_components']:
            c=components[k]
            if c['confidence']<.8:continue
            box=[c['bbox'][0]/size*width,c['bbox'][1]/size*height,c['bbox'][2]/size*width,c['bbox'][3]/size*height]
            add('L3-06',box,None,'unmatched_independent_target',dict(predicted_component_pixels=c['area_pixels']),
                dict(metric='unmatched_predicted_target',value=1.,unit='object'),c['confidence'])
    return result,matched


def main(args):
    import torch
    from PIL import Image
    h=l.helper();ann,_=h.annotation_module(l.ROOT);out=args.output;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(6);policy=json.loads((args.comparison/'deployment_policy.json').read_text());models={};cards=[];summaries=[];risk_rows=[];severity_rows=[]
    reviews={}
    for path in []: # Fresh inference does not replay old API pilot answers
        review=json.loads(path.read_text());reviews[review['source']['sampleId']]=review
    bm.mark('shared_imports_torch_and_localizer_initialization')
    for folder in sorted((args.roadshow/'packages').iterdir()):
        bm.package_begin(folder.name)
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        predictions={r['id']:r for r in map(json.loads,(folder/'predictions.jsonl').open())}
        ledgers=list(map(json.loads,(folder/'samples.jsonl').open()));family=ledgers[0]['family']
        inventories=l.ROOT/'opc-agent/runs/roadshow-evaluation-20261101/packages'/folder.name/'inventory.jsonl'
        records={r['id']:r for r in map(json.loads,inventories.open())};executed=0;located=collections.Counter();started=time.monotonic()
        coarse_path=l.ROOT/'ppt_results_20261104/hgb7_roadshow/packages'/folder.name/'predictions.jsonl'
        coarse={r['id']:r for r in map(json.loads,coarse_path.open())}
        route=policy['selected'].get(family,'not_pixel_task')
        bm.mark('read_predictions_ledgers_inventory_and_route')
        if route not in {'not_pixel_task','abstain'} and route not in models:
            bm.shared_begin('load_pixel_model')
            legacy=route=='legacy_recalibrated'
            path=(l.ROOT/'research_optimization_20261103/localizer/pilot_v1/weighted_bce_dice.pt') if legacy else args.localizer/(route+'.pt')
            saved=torch.load(path,map_location='cpu',weights_only=False)
            model=h.torch_model() if legacy else l.network(saved['width'])
            model.load_state_dict(saved['state_dict']);model=model.cuda().eval()
            models[route]=(model,saved['size'],h.sha(path),legacy)
            torch.cuda.synchronize()
            bm.shared_end('load_pixel_model',route=route,checkpoint=str(path))
        target=out/'packages'/folder.name;target.mkdir(parents=True)
        pending=[]
        for ledger in ledgers:
            row=records[ledger['sampleId']];s=predictions[row['id']];fine=set(s['model_candidates'])
            broad=coarse[row['id']]
            # This is an investigation trigger, not an eight-head probability.
            if (broad.get('model_candidate_bits') or [0,0,0])[1]:
                s=dict(s);s['model_candidates']=list(set(s['model_candidates'])|{'L3-06'})
                s['investigation_triggers']={'L3-06':dict(source='coarse_count_investigation',group_probability=broad['probabilities'][1])}
                fine=set(s['model_candidates'])
            if route in {'not_pixel_task','abstain'} or not ledger['dataPresent'] :continue
            pending.append((row,ledger,s))
        def load(item):
            row,ledger,signal=item;model,size,sha,legacy=models[route]
            raw_label=Path(row['label_path']).read_bytes()
            assert hashlib.sha256(raw_label).hexdigest()==ledger['labelSha256']
            before=h.fingerprint(row['data_path']);raw=Path(row['data_path']).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==row['data_sha256']
            with Image.open(io.BytesIO(raw)) as image:
                width,height=image.size
                rgb=np.array(image.convert('RGB').resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)/255
            assert before==h.fingerprint(row['data_path']) and [width,height]==ledger['imageSize']
            return dict(row=row,ledger=ledger,signal=signal,rgb=rgb.transpose(2,0,1).copy(),width=width,height=height,label=raw_label)
        bm.mark('prepare_candidates_and_output')
        with (target/'regions.jsonl').open('w') as detail,concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            prepared=iter(pool.map(load,pending))
            while True:
                items=list(itertools.islice(prepared,args.batch))
                if not items:break
                model,size,sha,legacy=models[route]
                th=policy['legacy_thresholds'][family] if legacy else json.loads(Path(policy['localizer_threshold_files'][route]).read_text())['thresholds'][family]
                bm.forward_begin(torch)
                probabilities,masks,presences=predict_batch(model,items,h.FAMILIES.index(family),th['pixel'],th['presence'],legacy,use_presence=route=='presence_gated')
                bm.forward_end(torch)
                for item,probability,mask,present in zip(items,probabilities,masks,presences):
                    row,ledger,s=item['row'],item['ledger'],item['signal'];width,height=item['width'],item['height']
                    components=h.instances(mask,probability,min_pixels=18,max_components=64)
                    # Current annotations enter only AFTER the independent forward pass.
                    label=json.loads(item['label'].decode('utf-8-sig'));extracted=ann.extract_annotation(label,family)
                    geometry,errors=h.extract_geometry(extracted['objects'],width,height)
                    if extracted['schema_status']!='ok':errors.append(dict(reason='unsupported_annotation_schema'))
                    proposed,matched=candidates(row,ledger,s,components,geometry,errors,h,sha)
                    q.attach_candidates(ledger,proposed);located.update(c['defectType'] for c in proposed);executed+=1
                    detail.write(json.dumps(dict(sampleId=row['id'],route=route,presence_probability=float(present),checkpoint_sha256=sha,
                        predicted_regions=[{k:v for k,v in c.items() if k!='mask'} for c in components],
                        coordinate_frame='384px internal masks; exported finding boxes use original image pixels',
                        geometry_errors=errors,candidates=proposed,comparison=matched,
                        spurious_negative_evidence_verified=False))+'\n')
                print(json.dumps(dict(stage='grounding',package=folder.name,executed=executed,screened=len(pending),batch=args.batch)),flush=True)
        bm.mark('fresh_decode_forward_component_match_and_regions_write')
        for ledger in ledgers:
            review=reviews.get(ledger['sampleId'])
            if review:
                ledger['visualReview']=dict(status=review['status'],caseId=review['case_id'],model='qwen3.8-chat',
                    official_accuracy_established=False,scope='24-case API protocol pilot',limitations=review.get('limitations',[]))
                for candidate in review.get('combined_defects',[]):
                    if candidate.get('status')=='confirmed':continue
                    loc=dict(candidate['locator']);loc['path']=ledger['labelPath'] if loc.get('path')=='current/label.json' else ledger['dataPath']
                    proposed=dict(candidate,sampleId=ledger['sampleId'],sample_id=ledger['sampleId'],
                        findingId=candidate['id'],defectType=candidate['defect_class'],checkId=candidate['check_id'],
                        ruleVersion=candidate['rule_version'],sourceKind='vision_LLM_pilot',locator=loc,
                        detail=candidate.get('explanation'),originalCaseId=review['case_id'],
                        evidence=dict(label_path=ledger['labelPath'],data_path=ledger['dataPath'],label_sha256=ledger['labelSha256'],
                            review_case_sha256=review.get('case_sha256')))
                    q.attach_candidates(ledger,[proposed])
        for code in q.CLASSES:
            issues=[(ledger,issue) for ledger in ledgers for issue in ledger['findings']
                if issue['defectType']==code and issue.get('findingStage')=='candidate']
            sample_risk={};weights=[]
            for ledger,issue in issues:
                weight=issue.get('weighted_burden')
                if q.finite(weight):
                    weights.append(weight);sid=ledger['sampleId'];sample_risk[sid]=max(sample_risk.get(sid,0.),weight)
                sev=issue.get('severity') or {}
                if q.finite(sev.get('value')):
                    severity_rows.append(dict(package=folder.name,sampleId=ledger['sampleId'],defectType=code,
                        findingId=issue['findingId'],metric=sev['metric'],unit=sev['unit'],value=sev['value'],
                        source=issue['sourceKind'],state='candidate',scoreEligible=False))
            risk_rows.append(dict(package=folder.name,defectType=code,samples=len(ledgers),candidateInstances=len(issues),
                weightedInstances=len(weights),candidateSamples=len({ledger['sampleId'] for ledger,_ in issues}),
                observedWeightedBurdenSum=math.fsum(weights),observedWeightedBurdenPerAllSamples=math.fsum(weights)/len(ledgers),
                meanSampleRisk=math.fsum(sample_risk.values())/len(ledgers),executedPixelSamples=executed,
                interpretation='Heuristic candidate risk, not calibrated pollution proportion or confirmed physical severity; instance burden can exceed 1'))
        with (target/'samples.jsonl').open('w') as fp:
            for ledger in ledgers:fp.write(json.dumps(ledger,ensure_ascii=False)+'\n')
        card=q.aggregate(ledgers);q.dump(target/'scorecard.json',card);cards.append(card)
        summaries.append(dict(package=folder.name,family=family,route=route,records=len(ledgers),executed=executed,
            located_candidate_instances=dict(located),seconds=time.monotonic()-started,confirmed_problem_samples=card['confirmedProblemSamples']))
        print(json.dumps(summaries[-1]),flush=True)
        bm.mark('candidate_risk_severity_aggregate_and_write')
        bm.package_end(samples=len(ledgers),family=family,images=executed,route=route,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0)
    bm.mark('shared_loop_finalization')
    q.dump(out/'scorecards.json',q.rank(cards));q.dump(out/'complete.json',dict(status='complete',packages=summaries,
        official_localized_metrics=None,scope='all readable roadshow RGB images freshly forwarded; frozen classifiers only gate candidate emission; model regions remain candidates',
        missing_or_spurious_count_not_inferred_from_empty_annotations=True,spurious_from_absence_of_detection=False))
    severity_groups=collections.defaultdict(list)
    for row in severity_rows:
        severity_groups[(row['package'],row['defectType'],row['metric'],row['unit'])].append(row)
    for group in severity_groups.values():
        previous=None;rank=None
        for index,row in enumerate(sorted(group,key=lambda r:-r['value']),1):
            if row['value']!=previous:rank=index;previous=row['value']
            row['severityRankWithinPackageAndMetric']=rank
    q.csv_write(out/'candidate_risk.csv',risk_rows);q.csv_write(out/'candidate_severity.csv',severity_rows)
    bm.mark('shared_global_scorecards_and_severity_ranking')
    bm.done()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--roadshow',type=Path,required=True);ap.add_argument('--localizer',type=Path,required=True)
    ap.add_argument('--comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--batch',type=int,default=16);main(ap.parse_args())
