"""Portable browser report: actual cards, locations and bounded visual reviews."""
from pathlib import Path
import argparse
import base64
import json


PAGE=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SpaceQA · 八类缺陷整改台</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:#203047;font:15px/1.65 "Microsoft YaHei",sans-serif}header{background:#102a46;color:white;padding:28px 5%}header h1{margin:0;font-size:29px}header p{margin:6px 0 0;color:#c8d7e7}main{max-width:1500px;margin:auto;padding:24px 3%}.cards{display:flex;gap:15px;flex-wrap:wrap}.metric{background:white;border:1px solid #dde5ee;border-radius:12px;padding:15px 24px;flex:1;min-width:170px}.metric b{display:block;font-size:28px;color:#196fa0}.notice{padding:14px 18px;border-left:4px solid #cc8b28;background:#fff5e5;margin:20px 0}section{background:white;border:1px solid #dde5ee;border-radius:12px;padding:20px;margin:20px 0}h2{font-size:20px;margin:0 0 14px}h3{font-size:17px}select,input,button{font:inherit;border:1px solid #becbd9;border-radius:6px;padding:7px;background:white;margin:3px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px 10px;border-bottom:1px solid #e3e9f1;text-align:left;vertical-align:top}th{background:#edf3f9;color:#3c5773;white-space:nowrap}tr.selectable{cursor:pointer}tr.selectable:hover{background:#f0f7ff}.scroll{overflow:auto}.bar{background:#e5eff8;border-radius:3px;min-width:110px}.bar i{display:block;background:#428bb5;height:6px;border-radius:3px}.muted{color:#718096;font-size:12px}.pill{display:inline-block;border-radius:4px;padding:1px 6px;background:#fff1d6;color:#8a5a0b;font-size:12px}.ok{background:#e6f5ec;color:#286340}.bad{background:#ffecec;color:#a43232}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5fa;padding:16px;border-radius:8px;font:12px/1.7 Consolas,monospace;max-height:500px;overflow:auto}.split{display:grid;grid-template-columns:1fr 1fr;gap:18px}.split img{width:100%;object-fit:contain;max-height:430px;background:#101820}.empty{padding:18px;color:#718096}@media(max-width:850px){.split{grid-template-columns:1fr}}a{color:#1f6899}footer{color:#718096;font-size:12px;padding:12px}
</style><header><h1>SpaceQA · 八类缺陷整改台</h1><p>原始证据 → 八类账本 → 比例与程度 → 六维检查 → 技术排序</p></header><main>
<div class="cards" id="metrics"></div><div class="notice">这是服务器重新读取全部12包后生成的结果。技术分仅统计已确认的文件、类别契约和尺寸问题；完整八类科学质量分及官方F1仍待独立验证。100分表示本检查范围内未确认问题。相同技术分保留真实并列，显示名次不能代替科学优劣。</div>
<section><h2>12包总览与技术排序</h2><div class="scroll"><table id="packages"></table></div><p class="muted">点击一行，查看该包的八类状态、六维覆盖、分布和问题证据。</p></section>
<section><h2 id="selected"></h2><div class="scroll"><table id="classes"></table></div><p class="muted">比例范围表示尚未完成的判断可能覆盖哪些样本，不是置信区间。“未完成”不改写成零缺陷。</p><h3>六维评分与检查覆盖</h3><div class="scroll"><table id="dimensions"></table></div><details><summary>展开分布、时间覆盖、来源和用途结论</summary><pre id="context"></pre></details></section>
<section><h2>细分问题与程度证据</h2><select id="code"><option value="">全部八类</option></select><input id="query" placeholder="搜索样本、字段或现象"><span id="count" class="muted"></span><div class="scroll"><table id="findings"></table></div><h3>选择问题查看原始文件、字段、坐标与程度</h3><pre id="evidence">请选择上表中的一条问题。</pre></section>
<section><h2>视觉复核试验 · 12包各2例</h2><p>原图/波形、输入标注与专家信号进入多模态模型。下方是实际冻结回答与位置，格式通过不表示科学判断正确。候选不扣确认问题分；本试验不能代表整包定位F1。</p><select id="review"></select><div id="reviewStatus"></div><div class="split"><div><h3>当前观测</h3><img id="raw"></div><div><h3>输入标注视图</h3><img id="overlay"></div></div><pre id="reviewJson"></pre></section>
<footer>规则 eight-class-ledger-1.0 · 随机种子20261104 · 评分与名次全部由程序计算 · 原始数据和权重保留服务器</footer></main>
<script id="payload" type="application/json">__PAYLOAD__</script><script>
const D=JSON.parse(document.getElementById('payload').textContent);const $=x=>document.getElementById(x);let selected=D.cards[D.cards.length-1].dirId;
const text=(el,s)=>el.textContent=s;const pct=x=>x==null?'待判':(100*x).toFixed(2)+'%';const num=x=>x==null?'待判':x.toFixed(2);const json=x=>JSON.stringify(x,null,2);const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function table(id,heads,rows){$(id).innerHTML='<thead><tr>'+heads.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</tbody>'}
const total=D.cards.reduce((s,c)=>s+c.sampleCount,0),confirmed=D.cards.reduce((s,c)=>s+c.confirmedProblemSamples,0);
$('metrics').innerHTML=[['12','已全量读取数据包'],[total.toLocaleString(),'登记样本'],[confirmed.toLocaleString(),'确认问题样本'],[D.findings.filter(f=>f.findingStage==='confirmed').length.toLocaleString(),'可追溯细分证据']].map(([n,l])=>'<div class="metric"><b>'+n+'</b>'+l+'</div>').join('');
table('packages',['显示名次','数据包','确认样本/N','观测技术分','HGB大类疑点','科学检查完成度'],D.cards.map(c=>[c.rankGlobalDisplay,esc(c.dirId),c.confirmedProblemSamples+'/'+c.sampleCount,num(c.observedTechnicalScore)+'<div class="bar"><i style="width:'+c.observedTechnicalScore+'%"></i></div>',c.hgbCoarseCandidateSamples,pct(c.scientificCheckCoverage)]));
[...$('packages').tBodies[0].rows].forEach((tr,i)=>{tr.className='selectable';tr.onclick=()=>choose(D.cards[i].dirId)});
for(const [code,name]of Object.entries(D.classes)){const o=document.createElement('option');o.value=code;o.textContent=code+' '+name;$('code').append(o)}
function choose(id){selected=id;const c=D.cards.find(x=>x.dirId===id);text($('selected'),id+' · 八类缺陷');table('classes',['类别','确认样本','定位证据数','视觉候选样本','观测比例','完整比例估计','未完成范围','待判样本'],Object.entries(c.defects).map(([code,v])=>[esc(code+' '+v.name),v.confirmedSamples,v.locatedInstances,v.candidateSamples,pct(v.observedConfirmedProportion),pct(v.proportionEstimate),v.proportionBounds.map(pct).join(' ～ '),v.statusCounts.undetermined]));table('dimensions',['维度','已实现范围技术分','完整维度分','检查完成/适用','覆盖','结论'],c.dimensions.map(d=>[esc(d.name),num(d.observedTechnicalScore),num(d.score),d.completedChecks+'/'+d.applicableCheckDenominator,pct(d.checkCoverage),'<span class="pill">'+esc(d.state)+'</span>']));text($('context'),json({distribution:c.distribution,provenance:c.provenance,usage:c.usage,ties:c.tiedDirectoryIds}));findings()}
function findings(){const code=$('code').value,query=$('query').value.toLowerCase();const rows=D.findings.filter(f=>f.dirId===selected&&(!code||f.defectType===code)&&(!query||json(f).toLowerCase().includes(query)));text($('count'),rows.length+'条，显示前100条；可搜索过滤');table('findings',['类别/状态','样本','位置','严重程度'],rows.slice(0,100).map(f=>[esc(f.defectType)+'<br><span class="pill '+(f.findingStage==='confirmed'?'ok':'')+'">'+esc(f.findingStage)+'</span>',esc(f.sampleId),esc(f.locator.annotation_pointer||f.locator.record_id||f.locator.path||json(f.locator)),f.severity?esc(f.severity.metric+': '+f.severity.value+' '+f.severity.unit):'未量化']));text($('evidence'),rows.length?json(rows[0]):'当前筛选未返回证据；这不等于整个数据包无缺陷。');[...$('findings').tBodies[0].rows].forEach((tr,i)=>{tr.className='selectable';tr.onclick=()=>text($('evidence'),json(rows[i]))})}
$('code').onchange=findings;$('query').oninput=findings;
for(let i=0;i<D.reviews.length;i++){const r=D.reviews[i],o=document.createElement('option');o.value=i;o.textContent=r.source.sampleId+' · '+r.status;$('review').append(o)}
function review(){const r=D.reviews[Number($('review').value)];if(!r)return;text($('reviewStatus'),r.model+' · '+r.status+' · '+r.seconds.toFixed(1)+'秒 · '+(r.combined_defect_count||0)+'个程序事实或模型候选');$('raw').src=D.media[r.case_id+'-observation.jpg']||'';$('overlay').src=D.media[r.case_id+'-annotation.jpg']||D.media[r.case_id+'-observation.jpg']||'';text($('reviewJson'),json({checks:r.final_checks,candidates:r.combined_defects,blocked:r.blocked_mechanical_claims,attempts:r.attempts,limitations:r.limitations}))}
$('review').value=D.reviews.reduce((best,r,i)=>r.combined_defect_count>(D.reviews[best].combined_defect_count||0)?i:best,0);$('review').onchange=review;choose(selected);review();
</script></html>'''


def render(results, reviews, output):
    cards=json.loads((results/'scorecards.json').read_text(encoding='utf-8'))
    cards=[{k:v for k,v in c.items() if k!='severity'} for c in cards]
    findings=[json.loads(x) for x in (results/'findings.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    rr=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((reviews/'responses').glob('*.json'))]
    media={p.name:'data:image/jpeg;base64,'+base64.b64encode(p.read_bytes()).decode() for p in (reviews/'media').glob('*.jpg')}
    data=json.dumps(dict(cards=cards,findings=findings,classes={k:v['name'] for k,v in cards[0]['defects'].items()},reviews=rr,media=media),ensure_ascii=False,separators=(',',':'),allow_nan=False)
    data=data.replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    output.write_text(PAGE.replace('__PAYLOAD__',data),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True)
    p.add_argument('--reviews',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(a.results,a.reviews,a.output)
