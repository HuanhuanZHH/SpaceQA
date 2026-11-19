"""Evidence-scoped report presentation. Unknown verdicts never become passes."""
from __future__ import annotations
import copy
import html
import math

DIMENSIONS = {
    'completeness':'完整性', 'annotation_quality':'标注质量',
    'sample_correspondence':'样本对应', 'coverage_distribution':'覆盖与分布',
    'documentation_provenance':'文档与来源追溯', 'usage_conditions':'使用条件',
}
ALIASES = dict(annotationQuality='annotation_quality',sampleCorrespondence='sample_correspondence',
               coverageDistribution='coverage_distribution',traceability='documentation_provenance',usageConditions='usage_conditions')
REASONS = {
    'coverage_distribution':('需覆盖基准','已有数量、类别及时间分布统计；缺少预期覆盖范围、目标分布和允许偏差，不能据此判定分布是否合格。'),
    'documentation_provenance':('需来源材料','需要可核验的数据说明、字段字典、来源链及版本信息；数据匿名化不作为错误扣分。'),
    'usage_conditions':('需用途标准','需要使用许可、目标用途及适用条件，才能判断是否满足接入、验收与选数要求。'),
}

def normalize_dimensions(value):
    source = value.values() if isinstance(value, dict) else value or []
    result = {}
    for raw in source:
        if not isinstance(raw,dict):continue
        row=copy.deepcopy(raw)
        key=ALIASES.get(row.get('id'),row.get('id'))
        if key not in DIMENSIONS: continue
        row.update(id=key,name=DIMENSIONS[key])
        if row.get('scoreKind')=='scoped_technical_observation':
            row['statusLabel']='技术观测分 · '+row.get('scope','已定义检查范围')
        elif key in REASONS:
            row['statusLabel'],row['reason']=REASONS[key]
        else:
            score=row.get('score')
            row['statusLabel']='已核验范围得分' if isinstance(score,(int,float)) else '需独立判定依据'
            row['reason']='得分仅覆盖已完成且可直接核验的检查项。分类已执行、候选已定位与科学结论已确认分别统计；缺少独立目标或事件参考的检查保留未决。'
        result[key]=row
    return result

def enrich_insights(value):
    value['sixDimensions']=normalize_dimensions(value.get('sixDimensions',{}))
    gaps=[]
    for axis in value.get('axes',[]):
        n=(axis.get('coverage') or {}).get('undeterminedSamples',0)
        if not n: continue
        code=axis['code']
        if code=='L2-T':
            reason='缺少独立事件起止真值；ULF数值时间轴尚无可信UTC原点。当前定位指向被质疑的输入事件，不能替代真实端点。'
            action='补充时间基准、事件定义和少量人工端点参考，再复核候选。'
        elif code in {'L3-06','L3-07'}:
            reason='模型候选已生成，但目标存在或不存在尚缺独立证据；空标注、模型未检出均不能直接确认漏标或虚标。'
            action='查看模型区域与输入对象匹配证据，优先人工复核高置信候选。'
        elif code in {'L2-G','L3-MIS'}:
            reason='几何候选需要可信目标边界和容差；有预测区域不等于已确认科学边界。缺数据的记录也无法做图像核验。'
            action='查看定位及IoU/覆盖测量，补充专业边界参考与容差。'
        elif code=='L2-P':
            reason='当前可核验字段契约冲突；图像或事件的真实语义类别还需独立类别定义及参考。'
            action='区分字段冲突与语义错误，核对类别字典和原始记录。'
        else:
            reason='缺少本项所需字段、原始文件或可独立核验的依据。'
            action='查看逐样本检查原因后补齐相应材料。'
        gaps.append(dict(code=code,name=axis['name'],unresolvedVerdictSamples=n,
                         reason=reason,nextAction=action,rawReasons=axis.get('reasons',[])))
    value['coverageGaps']=gaps
    value['coverageMeaning']=dict(
        execution='分类执行率：已生成有效模型输出的可推理样本 / 可推理样本。缺文件样本由规则处理。',
        localization='定位覆盖率：已指认文件、字段、区域或时段的检出样本 / 本类检出样本；不表示定位真值正确率。',
        verdict='可核验检查覆盖：已有足够证据给出通过或不通过的检查数 / 适用检查数。未决不等于未执行。',
        states='通过、不通过、证据未决、不适用四种状态分别展示；证据未决按缺失的具体依据解释。')
    return value

def render_report(session, insights, cases):
    esc=lambda x: html.escape(str(x if x is not None else '—').replace('待评估','需补充依据'),quote=True)
    number=lambda x: f'{x:.2f}' if isinstance(x,(int,float)) else '—'
    facts=session.get('analysis',{}).get('result') or {}
    summary=insights['summary'];axes=insights['axes']
    n=summary.get('totalSamples') or 0
    title=esc(insights['datasetName'])
    cards=''.join(f'<div class="metric"><span>{label}</span><b>{esc(summary.get(key))}</b></div>' for label,key in
                  [('登记样本','totalSamples'),('规则确认样本','confirmedSamples'),('仅候选样本','candidateOnlySamples'),('去重检出样本','foundSamples')])
    rows=''.join('<tr>'+''.join(f'<td>{v}</td>' for v in [esc(a['name']),esc(a['code']),esc(a['confirmedSamples']),esc(a['candidateOnlySamples']),
                 number(a.get('displayScore')),esc(a.get('stateLabel')),f"{a.get('localizationCoverage',{}).get('locatedSamples','—')} / {a.get('foundSamples','—')}"] )+'</tr>' for a in axes)
    dimensions=''.join(f'<article><h3>{esc(r["name"])}</h3><b>{number(r.get("displayScore",r.get("score")))+" 分 · 技术观测" if r.get("displayScore",r.get("score")) is not None else esc(r["statusLabel"])}</b><p>限定指标完成：{number(r.get("scopeCheckCoverage",r.get("checkCoverage"))*100)+"%" if isinstance(r.get("scopeCheckCoverage",r.get("checkCoverage")),(int,float)) else "无可判分母"}</p><p>{esc(r["reason"])}</p></article>' for r in insights['sixDimensions'].values())
    points=[];labels=[]
    for i,a in enumerate(axes):
        angle=2*math.pi*i/8-math.pi/2
        score=a.get('displayScore')
        radius=(score or 0)*1.2
        points.append(f'{220+radius*math.cos(angle):.2f},{160+radius*math.sin(angle):.2f}')
        x,y=220+155*math.cos(angle),160+145*math.sin(angle)
        short=['文件缺失','边界偏移','类别错误','时间抖动','尺寸矛盾','漏标','虚标','覆盖偏差'][i]
        labels.append(f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="middle" font-size="12">{short}</text>')
    rings=''.join(f'<circle cx="220" cy="160" r="{r}" fill="none" stroke="#dce4ee"/>' for r in [30,60,90,120])
    radar=f'<svg viewBox="0 0 440 330" role="img" aria-label="八类检出画像">{rings}<polygon points="{" ".join(points)}" fill="#2563eb22" stroke="#2563eb" stroke-width="2"/>{"".join(labels)}</svg>'
    evidence=[]
    representatives=[]
    for a in axes:
        matching=[c for c in cases if c.get('defectType')==a['code']]
        if matching:representatives.append(matching[0])
    for c in representatives:
        locator=c.get('locator') or {}
        import json
        concise={k:v for k,v in locator.items() if k in {'annotation_pointer','bbox','timeInterval','time_interval','coordinate_system'}}
        for key in ['path','expected_data_path','data_path']:
            if locator.get(key):concise[key]=str(locator[key]).replace('\\','/').rsplit('/',1)[-1]
        location='；'.join(f'{k}: {json.dumps(v,ensure_ascii=False) if not isinstance(v,str) else v}' for k,v in concise.items())
        href=f'/chat.html?session={session["id"]}'
        evidence.append(f'<tr><td>{esc(c.get("sampleName",c.get("sampleId")))}</td><td>{esc(c.get("defectType"))}</td><td>{"已确认" if c.get("stage")=="confirmed" else "候选"}</td><td>{esc(str(c.get("basis", ""))[:120])}</td><td><code>{esc(location)}</code></td><td>{esc(c.get("checkId"))}<br>{esc(c.get("ruleVersion"))}</td></tr>')
    gaps=''.join(f'<li><b>{esc(g["name"])}</b>：{esc(g["reason"])} {esc(g["nextAction"])}</li>' for g in insights.get('coverageGaps',[]))
    run=session.get('liveRun') or {};audit=run.get('audit') or session.get('_liveReport',{}).get('liveAudit') or {}
    execution=insights.get('classificationCoverage') or {};rgb=insights.get('inferenceCoverage') or {}
    runtext=f'八类融合记录：{execution.get("executedSamples","—")}；本轮图像前向：{rgb.get("actualFreshImageForwards","—")}；耗时：{number(audit.get("elapsedSeconds",facts.get("elapsedSeconds")))} 秒。21个上游专家复用经过哈希校验的预测缓存。'
    pollution=insights.get('pollution') or {}
    axis_names={a.get('code'):a.get('name',a.get('code')) for a in axes}
    pollution_rows=''.join('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in [axis_names.get(code,code), code, number(item.get('threshold')), item.get('overThresholdSamples','—'), number(item.get('probabilitySum')), number(item.get('pollutionIndexContribution'))])+'</tr>' for code,item in (pollution.get('defects') or {}).items())
    pollution_note=pollution.get('meaning','未能从当前报告复算概率污染指数；不虚构数值。')
    mode='本次真实整包重评' if session.get('analysis',{}).get('source')=='live_recompute' else '已保存的真实评价报告'
    print_css='''@media print{body{font-size:11px;line-height:1.5}h1{font-size:20px;margin:8px 0}h2{font-size:16px;margin-top:20px}h3{font-size:12px}header{padding-bottom:12px}.metrics{grid-template-columns:repeat(4,1fr);gap:8px;margin:16px 0}.metric{padding:12px}.metric b{font-size:23px}.portrait{display:grid!important;grid-template-columns:240px 1fr;gap:15px}.portrait svg{max-height:190px;width:100%}.dimensions{grid-template-columns:repeat(3,1fr)}article{padding:12px}article p{font-size:10px}.note{padding:12px 15px}table{font-size:10px}td,th{padding:8px}code{font-size:9px}.evidence th:nth-child(1){width:16%}.evidence th:nth-child(2){width:7%}.evidence th:nth-child(3){width:6%}.evidence th:nth-child(4){width:25%}.evidence th:nth-child(5){width:31%}.evidence th:nth-child(6){width:15%}}'''
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SpaceQA 质量报告 · {title}</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#edf2f8;color:#17243b;font:14px/1.7 "Microsoft YaHei",system-ui,sans-serif}}main{{max-width:1180px;margin:32px auto;background:white;padding:48px;border-radius:18px;box-shadow:0 15px 55px #19385a10}}header{{border-bottom:2px solid #2563eb;padding-bottom:24px}}.brand{{letter-spacing:3px;color:#2563eb;font-size:12px}}h1{{font-size:26px;overflow-wrap:anywhere;margin:12px 0}}h2{{font-size:20px;margin-top:36px}}h3{{font-size:14px;margin:0 0 8px}}.muted,p{{color:#5c6b80}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}}.metric,article{{padding:18px;border:1px solid #e0e7f0;border-radius:12px;background:#f8fafd}}.metric span,.metric b{{display:block}}.metric b{{font-size:28px}}.portrait{{display:grid;grid-template-columns:380px 1fr;align-items:center;gap:24px}}table{{border-collapse:collapse;width:100%;font-size:12px}}th{{background:#edf3fc;text-align:left}}td,th{{padding:11px;border-bottom:1px solid #e1e8f2;vertical-align:top}}.dimensions{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}article b{{color:#2456a0}}article p{{font-size:12px}}.note{{padding:16px 20px;background:#eff5ff;border-left:3px solid #2563eb;border-radius:5px}}code{{font-size:10px;white-space:pre-wrap;overflow-wrap:anywhere}}.evidence{{table-layout:fixed}}.evidence td{{overflow-wrap:anywhere}}a{{color:#2563eb}}.toolbar{{display:flex;gap:20px;justify-content:flex-end;margin-bottom:16px}}@media(max-width:800px){{main{{margin:0;padding:22px}}.portrait{{display:block}}.metrics,.dimensions{{grid-template-columns:repeat(2,1fr)}}}}@media print{{body{{background:white}}main{{margin:0;padding:8mm;max-width:none;box-shadow:none}}.toolbar{{display:none}}article,.metric,header{{break-inside:avoid}}h2{{break-after:avoid}}thead{{display:table-header-group}}tr{{break-inside:avoid}}}}
{print_css}</style><main><div class="toolbar"><span>浏览器 Ctrl+P 可保存 PDF</span><a href="/api/chat/sessions/{esc(session['id'])}/report">下载结构化 JSON</a></div><header><div class="brand">SPACEQA / EVIDENCE-BASED DATA QUALITY</div><h1>{title}</h1><p>{mode} · {esc(insights.get('evaluatedAt'))}<br>报告版本 {esc(insights.get('reportVersion'))} · 任务 {esc(facts.get('taskId'))}</p></header>
<div class="metrics">{cards}</div><div class="note"><b>结论摘要</b><br>本包共 {n} 条记录，规则确认 {esc(summary.get('confirmedSamples'))} 个问题样本，另有 {esc(summary.get('candidateOnlySamples'))} 个仅候选样本。不同类别可能重叠，逐类计数不直接相加。<br>{esc(runtext)}</div>
<h2>01 / 八类缺陷画像与检出</h2><div class="portrait">{radar}<p>画像分 = 100 ×（1 − 本类去重检出样本数 / 登记样本数）。包含已确认和模型候选，越高表示当前检出比例越低。未覆盖、不适用项不补分。此处是数据包画像，不能作为软件准确率或官方六维总分。</p></div><table><thead><tr><th>缺陷</th><th>编号</th><th>确认样本</th><th>仅候选</th><th>画像分</th><th>执行</th><th>检出中定位</th></tr></thead><tbody>{rows}</tbody></table>
<h2>02 / 六维技术观测与依据</h2><div class="dimensions">{dimensions}</div><p class="note">六维技术观测分 = 100 × 通过检查数 / 已完成且可核验检查数。限定指标完成比例与原有科学检查判定覆盖分别记录，证据未决保留在适用分母及得分区间中。覆盖分布仅核对声明一致性，追溯仅核验记录链路，使用条件仅核验技术接入；它们不代替总体代表性、真实来源认证或使用许可。<a href="{esc(insights.get('sixDimensionEvidenceUrl'))}">下载六维逐样本检查证据</a>。</p>
<h2>03 / 污染指数与排序依据</h2><p class="note">综合污染指数：100 × 各类超阈值概率之和 ÷ 登记样本数；指数越低越好，八类允许重叠。当前综合指数：{number(pollution.get('pollutionIndex'))}。{esc(pollution_note)}</p><table><thead><tr><th>缺陷</th><th>编号</th><th>阈值</th><th>超阈值样本</th><th>概率和</th><th>指数贡献</th></tr></thead><tbody>{pollution_rows}</tbody></table>
<h2>04 / 需补充的判断依据</h2><ul>{gaps}</ul>
<h2>05 / 问题与定位证据台账</h2><p>共 {len(cases)} 条问题记录，下表按类别与状态选取 {len(representatives)} 条代表案例。<a href="/api/chat/sessions/{esc(session['id'])}/evidence.jsonl">下载全部问题与定位台账 JSONL</a>。候选位置指向需复核的对象或时段，不自动确认为真实缺陷。图像可在交互案例库中开关原始标注与问题区域。</p><table class="evidence"><thead><tr><th>样本</th><th>缺陷</th><th>状态</th><th>依据</th><th>文件／字段／区域／时段</th><th>检查与版本</th></tr></thead><tbody>{''.join(evidence)}</tbody></table>
<h2>06 / 复核与计分说明</h2><p>确认技术分 = 100 ×（1 − 已确认且可计分的去重问题样本 / 全部登记样本）。模型候选保留概率与证据，独立列出，不直接扣确认技术分。定位命中只有在独立参考和容差均可用时确认；没有独立路演真值，不报告路演整体精确率、召回率、F1。当前报告不执行数据修复或仪器质量定标。</p><p><a id="roadshow-ranking-link" href="/roadshow-ranking.html">查看路演12包污染指数排序 →</a>（指数越低越靠前，同分保留并列。）</p><footer class="muted">SpaceQA · 版本、输入哈希、逐样本检查、定位和运行日志可在原始证据文件中复算。</footer></main></html>'''
