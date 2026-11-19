"""Frozen independent image localization of eight-head roadshow candidates.

Only model candidates are attached. No detector absence is certified as spurious
annotation, and no model regions deduct confirmed technical scores.
"""
from __future__ import annotations
import argparse,collections,concurrent.futures,hashlib,io,itertools,json,math,time
from pathlib import Path
import numpy as np
import localizer as l
from spaceqa_agent import quality_ledger as q


def _component_polygon(component, width, height, size):
    """Return a compact original-pixel polygon for a predicted component.

    The segmentation network produces a binary component mask.  A convex hull
    is used only as a display/locator approximation; the full mask remains in
    the private region record.  This is especially important for EPB, whose
    source annotation is a polygon rather than a rectangle.
    """
    mask = component.get('mask')
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        return None
    yy, xx = np.where(mask)
    if len(xx) < 3:
        return None
    points = sorted({(float(x) / size * width, float(y) / size * height) for x, y in zip(xx, yy)})
    if len(points) < 3:
        return None
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return [[round(x, 3), round(y, 3)] for x, y in hull] if len(hull) >= 3 else None


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
    def add(code,box,pointer,kind,measurements,severity,confidence,polygon=None):
        fine_probability=signals['probabilities'][signals['classes'].index(code)]
        trigger=signals.get('investigation_triggers',{}).get(code)
        group_probability=trigger['group_probability'] if trigger else None
        risk=max(fine_probability,group_probability) if trigger else fine_probability
        locator=dict(type='imageRegion',path=ledger['dataPath'],data_path=ledger['dataPath'],bbox=box,
            coordinate_system='image_pixels',origin='top_left',axis_order='xy',image_size=[width,height],
            frame_id=row['data_sha256'],annotation_pointer=pointer)
        if polygon and len(polygon) >= 3:
            # Keep the native polygon for display and downstream review.  The
            # bbox stays as an index shortcut, never as a replacement geometry.
            locator['type'] = 'polygon'
            locator['polygon'] = polygon
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
        component_polygon=_component_polygon(component,width,height,size)
        measurements=dict(mask_iou=match['iou'],coverage_reference=cov_r,coverage_annotation=cov_a,area_ratio=ratio,normalized_distance=distance)
        if 'L3-MIS' in codes and ((cov_r<.9 and cov_a>=.9 and ratio<.9) or (cov_a<.9 and cov_r>=.9 and ratio>1.1)):
            under=ratio<1;value=1-cov_r if under else 1-cov_a
            add('L3-MIS',box,g['pointer'],'undercoverage' if under else 'overcoverage',measurements,
                dict(metric='uncovered_reference_fraction' if under else 'excess_annotation_fraction',value=value,unit='ratio'),component['confidence'],component_polygon)
        elif 'L2-G' in codes and (match['iou']<.75 or distance>.05):
            add('L2-G',box,g['pointer'],'boundary_position_deviation',measurements,
                dict(metric='one_minus_mask_iou',value=1-match['iou'],unit='ratio'),component['confidence'],component_polygon)
    if 'L3-06' in codes and not errors:
        for k in matched['possible_omission_components']:
            c=components[k]
            if c['confidence']<.8:continue
            box=[c['bbox'][0]/size*width,c['bbox'][1]/size*height,c['bbox'][2]/size*width,c['bbox'][3]/size*height]
            add('L3-06',box,None,'unmatched_independent_target',dict(predicted_component_pixels=c['area_pixels']),
                dict(metric='unmatched_predicted_target',value=1.,unit='object'),c['confidence'],_component_polygon(c,width,height,size))
    return result,matched


def main(args):
    import torch
    from PIL import Image
    h=l.helper();ann,_=h.annotation_module(l.ROOT);out=args.output;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(6);policy=json.loads((args.comparison/'deployment_policy.json').read_text());models={};cards=[];summaries=[];risk_rows=[];severity_rows=[]
    reviews={}
    for path in (args.roadshow.parent/'visual_review_v4/responses').glob('*.json'):
        review=json.loads(path.read_text());reviews[review['source']['sampleId']]=review
    for folder in sorted((args.roadshow/'packages').iterdir()):
        predictions={r['id']:r for r in map(json.loads,(folder/'predictions.jsonl').open())}
        ledgers=list(map(json.loads,(folder/'samples.jsonl').open()));family=ledgers[0]['family']
        inventories=l.ROOT/'opc-agent/runs/roadshow-evaluation-20261101/packages'/folder.name/'inventory.jsonl'
        records={r['id']:r for r in map(json.loads,inventories.open())};executed=0;located=collections.Counter();started=time.monotonic()
        coarse_path=l.ROOT/'ppt_results_20261104/hgb7_roadshow/packages'/folder.name/'predictions.jsonl'
        coarse={r['id']:r for r in map(json.loads,coarse_path.open())}
        route=policy['selected'].get(family,'not_pixel_task')
        if route not in {'not_pixel_task','abstain'} and route not in models:
            legacy=route=='legacy_recalibrated'
            path=(l.ROOT/'research_optimization_20261103/localizer/pilot_v1/weighted_bce_dice.pt') if legacy else args.localizer/(route+'.pt')
            saved=torch.load(path,map_location='cpu',weights_only=False)
            model=h.torch_model() if legacy else l.network(saved['width'])
            model.load_state_dict(saved['state_dict']);model=model.cuda().eval()
            models[route]=(model,saved['size'],h.sha(path),legacy)
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
            if route in {'not_pixel_task','abstain'} or not ledger['dataPresent'] or not fine.intersection({'L2-G','L3-06','L3-07','L3-MIS'}):continue
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
        with (target/'regions.jsonl').open('w') as detail,concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            prepared=iter(pool.map(load,pending))
            while True:
                items=list(itertools.islice(prepared,args.batch))
                if not items:break
                model,size,sha,legacy=models[route]
                th=policy['legacy_thresholds'][family] if legacy else json.loads(Path(policy['localizer_threshold_files'][route]).read_text())['thresholds'][family]
                probabilities,masks,presences=predict_batch(model,items,h.FAMILIES.index(family),th['pixel'],th['presence'],legacy,use_presence=route=='presence_gated')
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
    q.dump(out/'scorecards.json',q.rank(cards));q.dump(out/'complete.json',dict(status='complete',packages=summaries,
        official_localized_metrics=None,scope='frozen classifier-screened raw-RGB localization; model regions remain candidates',
        missing_or_spurious_count_not_inferred_from_empty_annotations=True,spurious_from_absence_of_detection=False))
    for row in severity_rows:
        row['severityRankWithinPackageAndMetric']=1+sum(other['package']==row['package'] and other['defectType']==row['defectType']
            and other['metric']==row['metric'] and other['unit']==row['unit'] and other['value']>row['value'] for other in severity_rows)
    q.csv_write(out/'candidate_risk.csv',risk_rows);q.csv_write(out/'candidate_severity.csv',severity_rows)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--roadshow',type=Path,required=True);ap.add_argument('--localizer',type=Path,required=True)
    ap.add_argument('--comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--batch',type=int,default=16);main(ap.parse_args())
