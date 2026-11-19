"""Evidence-backed six-dimension presentation export; never invent official q."""
from pathlib import Path
import collections,csv,hashlib,json,re,time,zipfile
import ensemble as e

ROOT=e.ROOT;BASE=ROOT/'ppt_results_20261104';OUT=BASE/'roadshow_report'
SOURCE=ROOT/'complete_delivery_20261103/package_acceptance_051/run_2'
SCORED={'L0-01','L0-03','L2-G','L2-P','L2-T','L2-F','L3-06','L3-07','L3-MIS'}
PAIR={'L0-01','L0-03'}
POLICY='spaceqa-ppt-density-2026.11.04-v1'


def code(f):return f.get('defectType') or f.get('legacyFinding',{}).get('defectType') or f.get('checkId')
def csvwrite(path,rows,fields=None):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    details=e.read(ROOT/'roadshow_demo_20261104/package_result_details.json')
    hgb=e.read(BASE/'hgb7_roadshow/complete.json');assert hgb['status']=='complete'
    hrows=list(csv.DictReader((BASE/'hgb7_roadshow/package_summary.csv').open(encoding='utf-8-sig')))
    hmap={r['package']:r for r in hrows};rows=[];evidence=[];receipts={};verified={};original_checks=set()
    for item in sorted(details,key=lambda x:x['packageName']):
        name=item['packageName'];taskid=item['sourceTaskIds'][0];p=SOURCE/taskid
        result=e.read(p/'result.json');package=next(x for x in result['packages'] if x['packageId']==item['packageId'])
        cases=e.read(p/'cases.json');tools=package['toolResults'];bytool={t['tool']:t for t in tools}
        inventory=bytool['inventory_package']['samples'];samplemap={x['sampleId']:Path(x['labelPath']).stem for x in inventory}
        assert len(inventory)==item['sampleCount']
        confirmed=set();bycode=collections.defaultdict(set);pending=set();diagnostic=0;case_rows=[]
        for case in cases:
            f=case['finding'];c=code(f);sid=case.get('sampleId') or f.get('sampleId')
            eligible=f.get('scoreEligible') is True and f.get('verdict')=='fail' and f.get('findingStage')=='confirmed' and c in SCORED and f.get('sourceKind') in {'rule','human_review','independent_reference'}
            if eligible:
                assert sid in samplemap,(name,c,sid)
                confirmed.add(sid);bycode[c].add(sid)
            if f.get('verdict')=='undetermined' and sid:pending.add(sid)
            if f.get('verdict')=='fail' and not eligible and f.get('sourceKind')!='model':diagnostic+=1
            # Historical model candidates are retained only in the immutable source.
            if f.get('sourceKind')=='model':continue
            original_checks.add(f.get('checkId',''))
            locator=f.get('locator',{});value=locator.get('value','');value=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False)
            refs=[]
            for ev in case.get('evidence',[]):
                if ev.get('path'):
                    key=(ev['path'],ev.get('sha256'))
                    if key not in verified:
                        path=Path(ev['path']);assert path.is_file(),str(path)
                        if ev.get('sha256'):assert e.sha(path)==ev['sha256'],str(path)
                        verified[key]=True
                    refs.append(ev['path'])
            evidence.append(dict(findingId=f['findingId'],checkId=f.get('checkId',''),ruleVersion=f.get('ruleVersion',''),dirId=name,objectId=samplemap.get(sid,sid or ''),fieldPath=value,locatorType=locator.get('locatorType','file'),verdict=f.get('verdict','undetermined'),detail=json.dumps(dict(defectType=c,basis=f.get('basis'),scoreEligible=eligible,evidencePaths=refs,sourceTaskId=taskid),ensure_ascii=False,separators=(',',':'))))
        n=item['sampleCount'];missing=len(set().union(*(bycode[c] for c in PAIR)))
        annotated=set().union(*(s for c,s in bycode.items() if c not in PAIR)) if bycode else set()
        assert len(confirmed)==item['eligibleObservedSamples']
        dist=bytool['check_distribution'].get('distribution',{})
        usage=bytool['assess_usage'].get('usageConditions',{})
        dims=[
          dict(id='completeness',name='完整性',score=100*(1-missing/n),verdict='fail' if missing else 'pass',scope='原始配对、PNG头和JSON解析；不替代完整科学可读性审核',affected=missing,denominator=n),
          dict(id='annotationQuality',name='标注质量',score=100*(1-len(annotated)/max(1,n-missing)),verdict='fail' if annotated else 'undetermined',scope='已确认类别/尺寸等结构错误的技术分；模型语义和定位候选另列',affected=len(annotated),denominator=n-missing),
          dict(id='sampleCorrespondence',name='样本对应关系',score=100*(1-missing/n),verdict='fail' if missing else 'pass',scope='标注到实盘数据的引用存在性，不代表全部物理语义对应',affected=missing,denominator=n),
          dict(id='coverageDistribution',name='覆盖与分布',score=None,verdict='undetermined',scope='已统计空标、密度、类别及时间覆盖；缺预期类别比例/采样选取基准',statistics={k:dist.get(k) for k in ['emptyAnnotations','emptyAnnotationFraction','targetCount','targetDensityPerParsedLabel','classCounts']}),
          dict(id='traceability',name='文档与来源追溯',score=None,verdict='undetermined',scope='来源/版本字段已收集；缺独立且适用于本次子集的来源声明，不以完整B榜数量处罚子集'),
          dict(id='usageConditions',name='使用条件',score=None,verdict='undetermined',scope='已分别给出验收、入库、选数的条件与整改范围；缺用途专属验收基准',purposes={k:{q:v.get(q) for q in ['verdict','conditions']} for k,v in usage.items()}),
        ]
        for j,d in enumerate(dims,41):
            evidence.append(dict(findingId=f'PPT-{name}-C{j}',checkId=f'C{j}',ruleVersion=POLICY,dirId=name,objectId='',fieldPath=str(p/'result.json'),locatorType='file',verdict=d['verdict'],detail=json.dumps(d,ensure_ascii=False,separators=(',',':'))))
        if name.startswith('ULF_sample_'):
            evidence.append(dict(findingId=f'PPT-{name}-C48',checkId='C48',ruleVersion=POLICY,dirId=name,objectId='',fieldPath=str(p/'result.json'),locatorType='file',verdict='not_applicable',detail='原生数据为CSV时间序列，PNG像素尺寸检查不适用；绘制出的曲线预览不作为原生图像。'))
        hp=BASE/'hgb7_roadshow/packages'/name/'predictions.jsonl'
        for pred in e.rows(hp):
            for bit,flag in enumerate(pred['model_candidate_bits']):
                if not flag:continue
                fid=hashlib.sha256((pred['id']+str(bit)).encode()).hexdigest()[:20]
                evidence.append(dict(findingId='HGB7-'+fid,checkId='C'+str(31+bit),ruleVersion='HGB7-'+hgb['stack_sha256'][:12],dirId=name,objectId=Path(pred['label_path']).stem,fieldPath=pred['label_path'],locatorType='sampleId',verdict='undetermined',detail=json.dumps(dict(group='GCD'[bit],probability=pred['probabilities'][bit],threshold=pred['thresholds'][bit],scoreEligible=False,localizationAvailable=False,reason='HGB7样本级候选，需定位与独立复核；不自动扣分'),ensure_ascii=False,separators=(',',':'))))
        tech=100*(1-len(confirmed)/n)
        rows.append(dict(package=name,family='_'.join(name.split('_')[:-1]),sampleCount=n,dataCount=item['dataCount'],confirmedSamples=len(confirmed),confirmedRatio=len(confirmed)/n,technicalScore=tech,officialScore=None,legacyScore=item['baselineScore'],missingDataReferences=missing,annotationAffectedSamples=len(annotated),confirmedByCode={c:len(s) for c,s in bycode.items()},modelCandidateSamples=int(hmap[name]['model_candidate_samples']),G_model_candidates=int(hmap[name]['G_model_candidates']),C_model_candidates=int(hmap[name]['C_model_candidates']),D_model_candidates=int(hmap[name]['D_model_candidates']),diagnosticFindingCount=diagnostic,pendingSampleCount=len(pending),sourceElapsedSeconds=item['sourceElapsedSeconds'],historicalAll21ExpertSeconds=float(hmap[name]['historical_all21_expert_seconds']),auditPassed=item['auditPassed'],sourceTaskId=taskid,dimensions=dims,distribution=dims[3].get('statistics'),model='HGB7-21expert-fusion'))
        receipts[name]=dict(resultPath=str(p/'result.json'),resultSha256=e.sha(p/'result.json'),casesSha256=e.sha(p/'cases.json'),hgbPredictionSha256=e.sha(hp))
    rows.sort(key=lambda x:(x['confirmedRatio'],x['package']))
    ranks=collections.Counter()
    for i,row in enumerate(rows,1):
        ranks[row['family']]+=1;row['rankGlobal']=i;row['rankInDataset']=ranks[row['family']]
    assert len(rows)==12 and sum(r['sampleCount'] for r in rows)==24782
    assert sum(r['confirmedSamples'] for r in rows)==1690
    assert sum(r['missingDataReferences'] for r in rows)==272
    assert all(rows[i]['technicalScore']>=rows[i+1]['technicalScore'] for i in range(11))
    # Keep original check IDs in detail and provide an explicit local export-spec mapping.
    custom=sorted(c for c in original_checks if not re.fullmatch(r'C\d{2}',c));mapping={c:f'C{i+51}' for i,c in enumerate(custom)}
    for rec in evidence:
        if rec['checkId'] in mapping:
            rec['detail']='sourceCheckId='+rec['checkId']+'; '+rec['detail'];rec['checkId']=mapping[rec['checkId']]
    assert len({r['findingId'] for r in evidence})==len(evidence)
    assert {r['verdict'] for r in evidence}=={'pass','fail','undetermined','not_applicable'}
    csvwrite(OUT/'证据表_SpaceQA_技术展示版.csv',evidence)
    csvwrite(OUT/'路演结果_SpaceQA_技术展示版.csv',[dict(datasetId=r['family'],datasetVersion='v34',dirId=r['package'],rankInDataset=r['rankInDataset'],rankGlobal=r['rankGlobal'],totalScore=round(r['technicalScore'],1),mainDefectTypes=';'.join(k for k,v in sorted(r['confirmedByCode'].items()) if v),evidence=f"已确认问题样本{r['confirmedSamples']}/{r['sampleCount']}，缺数据{r['missingDataReferences']}，HGB7候选{r['modelCandidateSamples']}；自定义技术分，非官方四单元q") for r in rows])
    flat=[]
    for r in rows:
        x={k:v for k,v in r.items() if not isinstance(v,(dict,list))}
        for d in r['dimensions']:x[d['name']+'分']=d['score'];x[d['name']+'状态']=d['verdict']
        flat.append(x)
    csvwrite(OUT/'12包六维评分与排序.csv',flat)
    report=dict(status='complete',model='HGB7-21expert-fusion',policy=POLICY,policyFormula='100 * (1 - distinct confirmed scored affected samples / all registered label samples)',officialFormulaAvailable=False,officialFourUnitQ=None,rankingBasis='Confirmed affected-sample density across all 12 packages. Exact score ties retain same quality score and use directory ID for unique display order, not scientific superiority.',sixDimensions='Three scoped technical numeric scores and three explicit undetermined conclusions. Unknown dimensions are never replaced by 0 or 100.',rows=rows,evidenceRows=len(evidence),verdictCounts=dict(collections.Counter(r['verdict'] for r in evidence)),uniqueEvidenceFilesVerified=len(verified),exportCheckMapping=mapping,sourceReceipts=receipts,hgbReceipt=hgb,ruleRecheck=e.read(BASE/'rule_recheck/state.json'),limitations=['Technical display export, not approved official four-unit scoring','HGB7 sample candidates are undetermined, not confirmed localized defects','Tied clean-score directory ordering is administrative','Historical per-package Agent time uses the old selected experts; all21 historical model time is separately listed; fusion replay time is not full inference time'])
    e.write(OUT/'roadshow_report.json',report)
    e.write(OUT/'scoring_policy.json',dict(version=POLICY,totalScoreFormula=report['policyFormula'],dimensions=report['sixDimensions'],rule='confirmed+scoreEligible+official-scoring-type only; sample-union deduplication; diagnostics and model-only candidates excluded',official=False,exportCheckMapping=mapping))
    for filename in ['证据表_SpaceQA_技术展示版.csv','路演结果_SpaceQA_技术展示版.csv','12包六维评分与排序.csv','roadshow_report.json','scoring_policy.json']:
        with zipfile.ZipFile(OUT/(filename+'.zip'),'w',zipfile.ZIP_DEFLATED) as z:z.write(OUT/filename,arcname=filename)
    print(json.dumps(dict(status='complete',packages=12,confirmedSamples=1690,missingDataReferences=272,evidenceRows=len(evidence),verifiedEvidenceFiles=len(verified),output=str(OUT))),flush=True)


if __name__=='__main__':main()
