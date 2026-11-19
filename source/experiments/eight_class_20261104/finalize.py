"""Independent output checks and a small reproducible incremental delivery."""
from __future__ import annotations
import collections,csv,datetime,hashlib,json,math,platform,subprocess,zipfile
from pathlib import Path
from spaceqa_agent import quality_ledger as q

ROOT=Path('/staff/tangjiarui/OPC/eight_class_20261104')


def main():
    root=ROOT;issues=[];counts=collections.Counter();cards=[];risks={};sources=set()
    comparison=json.loads((root/'fusion_v2/comparison.json').read_text())
    assert comparison['status']=='complete' and comparison['packages']==30 and comparison['test_records']==18450
    for name in ['current_HGB15_eight','fusion_HGB7_eight','fusion_HGB15_eight']:
        coverage=json.loads((root/'fusion_v2'/(name+'_training_coverage.json')).read_text())
        assert len(coverage)==8 and all(c['learned'] for c in coverage.values())
    for folder in sorted((root/'grounded_roadshow_v1/packages').iterdir()):
        rows=[json.loads(line) for line in (folder/'samples.jsonl').open()]
        saved=json.loads((folder/'scorecard.json').read_text());actual=q.aggregate(rows)
        assert actual==saved,folder.name+' scorecard recomputation mismatch'
        cards.append(actual);counts['samples']+=len(rows);counts['confirmedProblemSamples']+=actual['confirmedProblemSamples'];counts['packages']+=1
        for row in rows:
            seen=set()
            for finding in row['findings']:
                assert finding['findingId'] not in seen;seen.add(finding['findingId'])
                assert finding.get('checkId') and finding.get('ruleVersion')
                stage=finding['findingStage'];counts[stage+'Findings']+=1;sources.add(finding.get('sourceKind','unknown'))
                assert not(stage=='candidate' and finding.get('scoreEligible'))
                loc=finding['locator'];box=loc.get('bbox')
                if box:
                    assert len(box)==4 and all(q.finite(v) for v in box)
                    assert box[0]<=box[2] and box[1]<=box[3]
                    size=loc.get('image_size') or row.get('imageSize')
                    if finding.get('sourceKind')=='independent_RGB_model':
                        assert size and -1e-5<=box[0]<=box[2]<=size[0]+1e-5 and -1e-5<=box[1]<=box[3]<=size[1]+1e-5
                        assert loc['frame_id']==finding['evidence']['data_sha256']
                        assert finding['checkpoint_sha256']
                        counts['locatedImageCandidates']+=1
                    counts['bboxFindings']+=1
                if stage=='candidate' and q.finite(finding.get('weighted_burden')):
                    key=(folder.name,finding['defectType']);risks[key]=risks.get(key,0.)+finding['weighted_burden']
                pointer=loc.get('annotation_pointer')
                if pointer is not None:
                    path=Path(row['labelPath']);cache_key=str(path)
                    raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==row['labelSha256']
                    obj=json.loads(raw.decode('utf-8-sig'))
                    for token in pointer.split('/')[1:]:
                        token=token.replace('~1','/').replace('~0','~');obj=obj[int(token)] if isinstance(obj,list) else obj[token]
                    counts['resolvedPointers']+=1
    assert counts['packages']==12 and counts['samples']==24782 and counts['confirmedProblemSamples']==1690
    ranked=q.rank(cards);saved=json.loads((root/'grounded_roadshow_v1/scorecards.json').read_text());assert ranked==saved
    for r in csv.DictReader((root/'grounded_roadshow_v1/candidate_risk.csv').open(encoding='utf-8-sig')):
        actual=risks.get((r['package'],r['defectType']),0.)
        assert math.isclose(actual,float(r['observedWeightedBurdenSum']),abs_tol=1e-8)
        assert math.isclose(actual/int(r['samples']),float(r['observedWeightedBurdenPerAllSamples']),abs_tol=1e-10)
    for p in [root/'localizer_v2/comparison.json',root/'candidate_comparison_v1/comparison.json',root/'spatial_comparison_v1/comparison.json']:
        assert json.loads(p.read_text())['status']=='complete'
    weights={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=q.file_sha(p)) for p in [
        root/'fusion_v2/models.pkl',root/'localizer_v2/presence_gated.pt',root/'localizer_v2/no_presence_loss.pt']}
    original_policy=json.loads((root.parent/'jwvl_optimization_20261104/ensemble_v1/frozen_policy.json').read_text())
    manifest=dict(status='pass',created=datetime.datetime.now().astimezone().isoformat(),seed=20261104,
        counts=dict(counts),sources=sorted(sources),weights=weights,python=platform.python_version(),
        original_experts=original_policy['experts'],original_expert_provenance=original_policy['provenance'],
        classifier_policy_sha256=q.file_sha(root/'fusion_v2/frozen_policy.json'),
        location_policy_sha256=q.file_sha(root/'candidate_comparison_v1/deployment_policy.json'),
        source_code_sha256={p.name:q.file_sha(p) for p in (root/'code').glob('*.py')},
        scientific_truth_available=False,official_localized_metrics=None,old_services_replaced=False,
        checks=['24 trained binary heads','all-30 classification test','all-12 scorecard exact recomputation',
            'confirmed vs candidate segregation','native-region bounds and source frames','annotation pointers resolve',
            'risk sum and denominators','technical rankings exact recomputation','new weight hashes'],
        limitations=['P0-derived technical labels are not independent scientific truth','Missing-label recall remains weak',
            'API protocol pilot is not an accuracy benchmark','Expert prediction caches reused; no new full-expert latency acceptance',
            'Origin/license/distribution targets still require external evidence'])
    q.dump(root/'report/delivery_audit.json',manifest)
    for name in ['classification_audit.json','evaluation_resume.json']:
        src=root/name
        if src.exists(): (root/'report'/name).write_bytes(src.read_bytes())
    decisions=[]
    for name in ['fusion_v2','fusion_v3_missing4']:
        result=json.loads((root/name/'comparison.json').read_text())
        decisions.append(dict(configuration=name,selected=name=='fusion_v2',
            validation=next(v['all_P1_P4'] for v in result['validation'] if v['variant']=='selected_eight'),
            strict_test=next(v['all_P1_P4'] for v in result['strict_mask_test'] if v['variant']=='selected_eight')))
    q.dump(root/'report/classifier_selection.json',dict(decisions=decisions,basis='Original configuration has higher validation-selection F1; tests separately reported; historical test reused'))
    primary=next(x['all_P1_P4'] for x in comparison['strict_mask_test'] if x['variant']=='selected_eight')
    grounding=json.loads((root/'grounded_roadshow_v1/complete.json').read_text())
    executed=sum(p['executed'] for p in grounding['packages']);seconds=sum(p['seconds'] for p in grounding['packages'])
    delivery=f'''# 本轮交付与自查

本轮已完成训练、全30包分类测试、12包路演分类和候选定位、结果复算及增量交付。正式旧模型和服务没有被覆盖。新报告页为 http://127.0.0.1:18770/classification.html 。

当前保留fusion_v2。原分组测试集18,450条、六类30包；P1～P4严格配对技术代理的精确率{primary['precision']:.2%}、召回率{primary['recall']:.2%}、F1 {primary['F1']:.2%}。这不是官方缺陷定位成绩。去掉缺文件、类别词表、尺寸三类后，五类子集F1为89.53%；漏标召回19.67%、F1 26.67%，仍是明显短板。

## 目前做到哪里

|事项|实际状态|
|---|---|
|八类样本级多标签分类|完成三组各八头训练，规则与按类型/缺陷冻结组合，全30包分开报告|
|六类覆盖|六类都进入分类；图像像素分支只适用AIA/HMI/EPB/DMSP，CME/ULF使用参数/时序分支|
|新定位训练|两组各12轮、batch16、384×384，独立RGB输入，全部训练后再评估|
|新旧定位比较|同一943张验证图像、958张测试图像；严格误报约束与候选像素Dice两种阈值协议分开|
|路演推理|全12包24,782条完成分类；筛选后实际对{executed}张图像执行定位，生成{counts['locatedImageCandidates']}处图像候选；另有1个API复核候选|
|八类都准确定位|尚未完成；当前区域候选集中于边界、覆盖、漏标；不能把样本级虚标概率当成已定位虚标|
|时间缺陷|ULF监督为观测窗口声明，CME为次级事件身份代理；独立事件起止真值仍缺|
|程度与比例|确认样本比例、未知范围、同指标程度排序可复算；候选风险单独导出，不当真实污染率|
|六维评分和排序|输出六维状态、覆盖率、有限检查范围技术分和排序；来源、许可、预期分布缺外部依据的项仍待判|
|视觉大模型|12包共24例调用试验，19例通过协议校验；没有独立准确率提升证据|
|官方关键缺陷召回|尚不能报告；缺独立、含实例位置的官方参考答案|
|3小时与20GB验收|本轮未重跑完整21专家打包验收；新增融合与定位耗时不代替完整验收|
|证据审计|12包评分卡逐值复算一致；{counts['resolvedPointers']}个JSON位置可解析；模型区域在原图边界内；候选未扣确认分|

## 结果与程序

- [详细指标与定位比较](分类结果与当前状态.md)
- [完整模型结构、输入输出维度、训练参数与复现指令](模型结构与训练说明.md)
- [全部30包彩色排名与路演位置预览](classification.html)
- [独立交付审计与权重哈希](delivery_audit.json)
- [候选风险表](candidate_risk.csv)：Σ候选实例概率÷全包样本数；实例多时可超过1，不能当已确认污染率。
- [候选程度及同类排序](candidate_severity.csv)：按同包、同类、同指标与单位比较。

本轮路演仍有1,690个不同样本存在规则确认问题，共{counts['confirmedFindings']}条确认记录；同一样本可有多条记录。模型和LLM产生的{counts['candidateFindings']}条候选不混入这一数字。定位阶段实测约{seconds:.1f}秒，复用了已校验的21专家输出，不能当作整套系统推理时间。

新权重位于服务器`/staff/tangjiarui/OPC/eight_class_20261104/fusion_v2/models.pkl`及`localizer_v2/`。本增量包只含代码与报告，数据、权重、密钥留服务器；原21专家按现有来源引用。新静态结果页已可用，旧前端默认模型没有自动替换。
'''
    (root/'report/本轮交付与自查.md').write_text(delivery,encoding='utf-8')
    bundle=root/'SpaceQA_八类分类与定位_增量交付_20261105.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for directory in ['code','report']:
            for path in (root/directory).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix in {'.py','.md','.json','.csv','.html','.jpg','.png'}:
                    z.write(path,str(path.relative_to(root)))
    q.dump(root/'incremental_delivery.json',dict(path=str(bundle),bytes=bundle.stat().st_size,sha256=q.file_sha(bundle),
        includes='New experiment source and reports only; incremental delivery',excludes=['raw datasets','weights','credentials'],audit='report/delivery_audit.json'))
    print(json.dumps(dict(status='pass',counts=dict(counts),bundle=str(bundle),bytes=bundle.stat().st_size)))


if __name__=='__main__':main()
