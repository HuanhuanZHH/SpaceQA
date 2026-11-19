"""Read-only v2.3 delivery checks. No dataset inference or model calls."""
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse, csv, hashlib, json, math, subprocess, tarfile, time

EXCLUDED = {'历史备查_20261110', '交付归档'}
FINDING_FIELDS = 'findingId checkId ruleVersion dirId datasetId datasetVersion defectType status evidenceRef severity affectedScope criticality basis notApplicableReason confirmedOrPending'.split()
RANK_FIELDS = 'rank dirId datasetId datasetVersion totalScore dimensionScores rankInDataset rankGlobal groupId comparisonBasis retainSubRanking'.split()
EVIDENCE_FIELDS = 'evidenceId findingId locatorType locator contentHash capturedBy'.split()
LOG_FIELDS = 'ts runId datasetId dirId event checkId status elapsedMs resourcePeak tokenUsage retryCount'.split()
ROAD_FIELDS = 'datasetId datasetVersion dirId rankInDataset rankGlobal totalScore mainDefectTypes evidence'.split()
DISPLAY_FIELDS = 'findingId checkId ruleVersion dirId objectId fieldPath locatorType verdict detail'.split()

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def rows(p):
    with p.open(encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def payload_files(root):
    for p in root.rglob('*'):
        rel = p.relative_to(root)
        if rel.parts[0] in EXCLUDED: continue
        if p.is_file(): yield p

def verify(root, hash_evidence=True):
    root = Path(root).resolve(); start = time.monotonic(); errors = []; checks = []
    def check(name, ok, detail=None):
        checks.append({'check': name, 'passed': bool(ok), 'detail': detail})
        if not ok: errors.append({'check': name, 'detail': detail})
    cards = {p.stem: read(p) for p in sorted((root/'04_评分卡').glob('*.json'))}
    fixed = ['00_MANIFEST.yaml','SHA256SUMS.txt','01_方案规划.md','02_部署与使用说明.md','03_评价规范.md','03_评价规范.yaml','09_未完成事项说明.md','10_演示视频.mp4']
    fixed += ['04_评分卡/'+d+s for d in cards for s in ('.json','.html')]
    fixed += ['05_排序表/ranking_global.csv','05_排序表/ranking_explanation.md','06_问题清单/findings.jsonl','06_问题清单/findings_report.md','07_证据索引/evidence_index.csv','08_运行日志/run.log.jsonl','08_运行日志/system_env.txt','08_运行日志/seeds.yaml']
    deck = list((root/'11_展示材料').glob('*.pptx'))
    fixed += [str(deck[0].relative_to(root))] if len(deck)==1 else []
    fixed += ['11_展示材料/路演结果_LTH团队.csv','11_展示材料/证据表_LTH团队.csv','11_展示材料/运行记录.txt']
    fixed += ['agent/'+n+'.yaml' for n in ['manifest','tools','skills','rules','memory','plugins']]
    fixed += ['image/LTH_image.tar','image/Dockerfile','image/docker-compose.yml']
    check('fixed_53_files',len(fixed)==53 and all((root/n).is_file() and (root/n).stat().st_size>0 for n in fixed),{'count':len(fixed),'missing':[n for n in fixed if not (root/n).is_file()]})
    check('three_datasets_twelve_directories', len(cards)==12 and len({c['datasetId'] for c in cards.values()})==3, sorted(cards))
    check('complete_source_and_dependencies',(root/'source/src/spaceqa_agent').is_dir() and (root/'source/pyproject.toml').is_file())
    manifest = read(root/'00_MANIFEST.yaml')
    check('manifest_scope_matches_cards',set(manifest['evaluationScope']['dirIds'])==set(cards) and manifest['evaluationScope']['dirCount']==12)
    check('manifest_DEL1_through_DEL12', {d['id'] for d in manifest['deliverables']}=={'DEL-'+str(i) for i in range(1,13)})
    findings = []
    with (root/'06_问题清单/findings.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): findings.append(json.loads(line))
    by_id = {f['findingId']: f for f in findings}
    check('unique_finding_ids',len(by_id)==len(findings),len(findings))
    check('finding_fields',all(set(FINDING_FIELDS)<=set(f) and {'metric','value','unit','tolerance'}<=set(f.get('severity') or {}) for f in findings))
    check('finding_directory_identity',all(f['dirId'] in cards and f['datasetId']==cards[f['dirId']]['datasetId'] and f['datasetVersion']==cards[f['dirId']]['datasetVersion'] for f in findings))
    index = rows(root/'07_证据索引/evidence_index.csv')
    check('evidence_schema',bool(index) and set(EVIDENCE_FIELDS)<=set(index[0]))
    eids = {r['evidenceId']: r for r in index}
    check('unique_evidence_ids',len(eids)==len(index))
    check('evidence_finding_closure',{r['findingId'] for r in index}==set(by_id) and all(f['evidenceRef'] in eids and eids[f['evidenceRef']]['findingId']==f['findingId'] for f in findings))
    def verify_evidence(r):
        p=(root/r['path']).resolve()
        if not p.is_relative_to(root) or not p.is_file(): return r['evidenceId']
        if hash_evidence and digest(p)!=r['contentHash']: return r['evidenceId']
    with ThreadPoolExecutor(max_workers=6) as ex: bad=[x for x in ex.map(verify_evidence,index) if x]
    check('evidence_actual_bytes_and_sha256',not bad,{'checked':len(index),'hashes_checked':hash_evidence,'bad':bad[:20]})
    grouped=defaultdict(list)
    for f in findings: grouped[f['dirId']].append(f)
    scores={}; package=[]
    for d,c in cards.items():
        fs=grouped[d]; confirmed={f['sampleId'] for f in fs if f.get('findingStage')=='confirmed' and f.get('scoreEligible') is True}
        candidates={f['sampleId'] for f in fs if f.get('findingStage')=='candidate'}
        include_pending = c.get('totalScoreType') == 'confirmedPlusPendingScenarioScore'
        affected = confirmed | candidates if include_pending else confirmed
        score = 100 * (1 - len(affected) / c['sampleCount'])
        scores[d] = score
        if include_pending:
            check('pending_union_recompute:'+d, c.get('scenarioProblemSamples') == len(affected)
                  and abs(c.get('scenarioErrorRate', -1) - len(affected)/c['sampleCount']) < 1e-12)
            values = [dim['score'] for dim in c['dimensions']]
            check('six_observed_recompute:'+d, len(values)==6 and all(0 <= v <= 100 for v in values)
                  and abs(sum(values)/6 - c.get('sixDimensionCoverageAdjustedScore', c['sixDimensionObservedScore'])) < 1e-9)
        check('score_recompute:'+d,abs(score-c['totalScore'])<1e-9,{'confirmedSamples':len(confirmed),'denominator':c['sampleCount'],'score':score})
        check('card_finding_ids:'+d,{f['findingId'] for f in c['findings']}=={f['findingId'] for f in fs})
        check('card_six_dimensions:'+d,len(c['dimensions'])==6 and all({'dimId','score','range','grade','meaning'}<=set(x) for x in c['dimensions']))
        check('card_four_state_counts:'+d,{'pass','fail','undetermined','not_applicable'}<=set(c['statusCounts']))
        check('card_evidence_ids:'+d,set(c['evidenceIds'])<={f['evidenceRef'] for f in fs})
        package.append({'dirId':d,'datasetId':c['datasetId'],'samples':c['sampleCount'],'confirmedSamples':len(confirmed),'candidateSamples':len(candidates),'technicalScore':score,'formalScore':c.get('fullQualityScore'),'nativeRunId':c.get('runId'),'exportRunId':c.get('exportRunId')})
    ranking=rows(root/'05_排序表/ranking_global.csv')
    check('ranking_schema_and_scope',len(ranking)==12 and set(RANK_FIELDS)<=set(ranking[0]) and {r['dirId'] for r in ranking}==set(cards))
    for r in ranking:
        d=r['dirId']; rank=1+sum(s>scores[d] for s in scores.values()); family=cards[d]['datasetId']
        sub=1+sum(s>scores[d] for k,s in scores.items() if cards[k]['datasetId']==family)
        check('ranking_recompute:'+d,abs(float(r['totalScore'])-scores[d])<1e-9 and int(r['rankGlobal'])==rank and int(r['rank'])==rank and int(r['rankInDataset'])==sub)
    road=rows(root/'11_展示材料/路演结果_LTH团队.csv'); display=rows(root/'11_展示材料/证据表_LTH团队.csv')
    check('roadshow_schema',len(road)==12 and set(ROAD_FIELDS)<=set(road[0]))
    check('roadshow_score_rounding',all(abs(float(r['totalScore'])-scores[r['dirId']])<=.050000001 for r in road))
    check('display_evidence_schema',bool(display) and set(DISPLAY_FIELDS)<=set(display[0]))
    check('display_ids_and_provenance',len(display)==len(findings) and {r['findingId'] for r in display}==set(by_id) and all(r['checkId']==by_id[r['findingId']]['checkId'] and r['dirId']==by_id[r['findingId']]['dirId'] and r['ruleVersion']==by_id[r['findingId']]['ruleVersion'] for r in display))
    log_count=0;missing=0;bad_ts=0;states=Counter();native_ids=set()
    with (root/'08_运行日志/run.log.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line);log_count+=1;missing+=not set(LOG_FIELDS)<=set(r);states[r.get('status')]+=1;native_ids.add(r.get('runId'))
            try:
                dt=datetime.fromisoformat(r['ts'].replace('Z','+00:00'));bad_ts+=dt.utcoffset().total_seconds()!=0
            except Exception:bad_ts+=1
    check('log_required_fields_and_UTC',missing==0 and bad_ts==0,{'rows':log_count,'missingFieldRows':missing,'badUtcRows':bad_ts})
    check('real_failure_records_retained',states['failed']>0,dict(states))
    im=root/'image/LTH_image.tar';check('image_under_20_decimal_GB',im.stat().st_size<=20_000_000_000,im.stat().st_size)
    try:
        with tarfile.open(im,'r:') as t:
            m=json.loads(t.extractfile('manifest.json').read());tags=[v for x in m for v in x.get('RepoTags',[])]
        check('docker_save_archive_manifest',bool(tags),tags)
    except Exception as exc:check('docker_save_archive_manifest',False,str(exc))
    try:
        r=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name','-of','json',str(root/'10_演示视频.mp4')],capture_output=True,text=True,check=True)
        video=json.loads(r.stdout);check('video_H264_under_five_minutes',float(video['format']['duration'])<=300 and any(x.get('codec_name')=='h264' for x in video['streams']),video)
    except Exception as exc:check('video_H264_under_five_minutes',False,str(exc))
    policy=root/'13_六维评分修复_20261110/approved_weight_policy.json'
    check('approved_weight_policy_preserved',read(policy).get('approvalStatus')=='approved' and set(read(policy)['weights'].values())=={1},digest(policy))
    organize=read(root/'08_运行日志/交付整理/整理记录.json')
    check('PPT_bytes_unchanged',len(deck)==1 and digest(deck[0])==organize['pptSha256'],organize['pptSha256'])
    limitations=[
      'Formal six-dimension composite scores remain unavailable for all 12 packages; independent reference, calibration and target/authority evidence are incomplete.',
      'Critical-defect recall >=0.80 is not verified against an independent official reference.',
      'Historical package-native runId is missing; exportRunId is explicitly an export identity, not a fabricated inference identity.',
      'The retained two-repeat evidence covers existing small subsets, not two fresh full 12-package runs.',
      'PPT is retained byte-for-byte at user request; its scientific claims are outside this packaging audit.',
      'The retained video is the existing results walkthrough; this task did not record or launch a new model run.'
    ]
    return {'status':'structural_pass_scientific_partial' if not errors else 'structural_fail','completedUtc':datetime.now(timezone.utc).isoformat(),'elapsedSeconds':time.monotonic()-start,'auditScope':'Existing artifacts, schema, identity, evidence bytes and arithmetic only; no evaluation/inference/retraining','newPackageRuns':0,'newModelCalls':0,'fixedFileCount':len(fixed),'checksPassed':sum(c['passed'] for c in checks),'checksTotal':len(checks),'errors':errors,'packages':package,'findingRecords':len(findings),'confirmedRecords':sum(f.get('findingStage')=='confirmed' and f.get('scoreEligible') is True for f in findings),'candidateRecords':sum(f.get('findingStage')=='candidate' for f in findings),'evidenceRecords':len(index),'logRows':log_count,'formalFullScoresAvailable':sum(p['formalScore'] is not None for p in package),'limitations':limitations,'checks':checks,'fixedFiles':fixed}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);ap.add_argument('--output',type=Path);ap.add_argument('--skip-evidence-hashes',action='store_true');a=ap.parse_args()
    result=verify(a.root,not a.skip_evidence_hashes)
    if a.output:a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['packages','checks','fixedFiles']},ensure_ascii=False,indent=2))
    raise SystemExit(bool(result['errors']))
