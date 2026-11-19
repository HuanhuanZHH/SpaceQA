"""Readable metrics, all-30 comparison, actual routing, and architecture links."""
from __future__ import annotations
import argparse,collections,csv,html,json,shutil
from pathlib import Path

NAMES={'rules':'确定性规则','current_HGB15_eight':'当前特征八头HGB15','fusion_HGB7_eight':'21专家＋八头HGB7','fusion_HGB15_eight':'21专家＋八头HGB15','selected_eight':'按类型和缺陷选择的组合'}
CN={'L0-01':'缺数据文件','L2-G':'边界/位置','L2-P':'类别','L2-T':'时间声明','L2-F':'尺寸','L3-06':'漏标','L3-07':'虚标','L3-MIS':'覆盖'}
COLORS=['#ffd5d5','#fff1a6','#e7d1fa','#d3e6ff','#d9f0d8']


def pct(v):return '—' if v is None else f'{100*v:.2f}%'
def metric(tp,fp,fn):return [tp/(tp+fp) if tp+fp else None,tp/(tp+fn) if tp+fn else None,2*tp/(2*tp+fp+fn) if tp+fp+fn else None]
def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def main(args):
    root=args.root;out=args.output;out.mkdir(parents=True,exist_ok=True)
    fusion=root/'fusion_v2';summary=json.loads((fusion/'comparison.json').read_text());policy=json.loads((fusion/'frozen_policy.json').read_text())
    rows=list(csv.DictReader((fusion/'all30_strict_mask_test.csv').open(encoding='utf-8-sig')))
    classrows=list(csv.DictReader((fusion/'all30_classes_strict_mask_test.csv').open(encoding='utf-8-sig')))
    variants=list(NAMES);families=list(policy['selected']);sections=[]
    main_rows=[]
    for s in summary['strict_mask_test']:
        m=s['all_P1_P4'];main_rows.append([NAMES[s['variant']],pct(m['precision']),pct(m['recall']),pct(m['F1']),m['TP'],m['FP'],m['FN']])
    sections.append('<h2>同一严格配对掩码上的八类测试</h2>'+table(['方案','精确率','召回率','F1','TP','FP','FN'],main_rows))
    sections.append('<p>测试集18,450条记录，覆盖六类30包。表中排除CME速度三元组候选配对的时间正例；仍是P0相对技术监督，未计定位匹配。ULF时间项是观测窗口声明，不代表事件起止真值。</p>')
    scientific=[]
    for s in summary['strict_mask_test']:
        m=s['scientific_proxy_P1_P4'];scientific.append([NAMES[s['variant']],pct(m['precision']),pct(m['recall']),pct(m['F1']),m['TP'],m['FP'],m['FN']])
    sections.append('<h2>较难的五类子集：排除缺文件、类别词表与尺寸项</h2>'+table(['方案','精确率','召回率','F1','TP','FP','FN'],scientific))
    sections.append('<p>此表包含几何、时间声明、漏标、虚标与覆盖，仍为技术代理分类。它与八类综合指标的差距说明不能只看综合分。</p>')
    weak=next(s for s in summary['test'] if s['variant']=='selected_eight')['all_P1_P4']
    sections.append('<p>加入单独标明的CME候选身份时间标签，组合结果为：P '+pct(weak['precision'])+'，R '+pct(weak['recall'])+'，F1 '+pct(weak['F1'])+'。两种口径分别保留，不能与旧G/C/D三组标签的F1直接比较。</p>')
    # All 30 packages, all configurations, requested rank colours; ties share rank.
    for family in families:
        part=[]
        for tier in ['P0','P1','P2','P3','P4']:
            measure='FPR' if tier=='P0' else 'F1'
            values={r['variant']:float(r[measure]) if r[measure] else None for r in rows if r['family']==family and r['tier']==tier}
            levels=sorted({v for v in values.values() if v is not None},reverse=tier!='P0')
            cells=[]
            for name in variants:
                value=values[name]
                if value is None:cells.append('—');continue
                rank=levels.index(value);cells.append('<span class="rank" style="background:'+COLORS[min(rank,4)]+'">'+pct(value)+' · '+str(rank+1)+'</span>')
            n=next(r['records'] for r in rows if r['family']==family and r['tier']==tier)
            part.append([tier+(' 误报率↓' if tier=='P0' else ' F1↑'),n]+cells)
        sections.append('<h2>'+html.escape(family)+'</h2>'+table(['档位','记录数']+[NAMES[x] for x in variants],part))
    sections.append('<p>颜色依次为第一红、第二黄、第三紫、第四蓝、第五绿；同分同名次。—表示该口径无可计算F1，不能当作100%。P0没有阳性时尤其要查看误报量与分母。</p>')
    # Class totals expose difficult rare classes instead of hiding behind micro F1.
    totals=collections.defaultdict(collections.Counter)
    for r in classrows:
        if r['variant']!='selected_eight' or r['tier']=='P0':continue
        for k in ['TP','FP','FN','TN','known','unknown','positives']:totals[r['code']][k]+=int(r[k])
    details=[]
    for code in CN:
        c=totals[code];p,r,f=metric(c['TP'],c['FP'],c['FN'])
        details.append([code+' '+CN[code],c['positives'],c['known'],c['unknown'],pct(p),pct(r),pct(f)])
    sections.append('<h2>八个细分类分别表现</h2>'+table(['类别','阳性数','已知项','未知/不适用','精确率','召回率','F1'],details))
    routes=[]
    for family,chosen in policy['selected'].items():routes.append([family]+[NAMES[chosen[c]] for c in CN])
    sections.append('<h2>真实冻结的模型选择</h2>'+table(['数据类型']+[c+' '+CN[c] for c in CN],routes))
    sections.append('<p>某类选择规则，可能是规则已够用，也可能是选择集尚无足够证据证明模型有效。尤其不能据此宣称已经识别该类所有实际问题。</p>')
    ablation_path=root/'fusion_v3_missing4/comparison.json';ablation_rows=[]
    if ablation_path.exists():
        altered=json.loads(ablation_path.read_text())
        for label,result in [('原八头组合（保留）',summary),('漏标正例权重×4（对照）',altered)]:
            val=next(x for x in result['validation'] if x['variant']=='selected_eight')['all_P1_P4']
            test=next(x for x in result['strict_mask_test'] if x['variant']=='selected_eight')['all_P1_P4']
            ablation_rows.append([label,pct(val['F1']),pct(test['precision']),pct(test['recall']),pct(test['F1'])])
        sections.append('<h2>追加漏标加权实验：原配置保留</h2>'+table(['配置','验证选择集F1（含弱时间）','严格测试P','严格测试R','严格测试F1'],ablation_rows))
        sections.append('<p>加权提高漏标召回，但误报增加、验证选择集F1下降。主结果与路演继续使用fusion_v2。追加实验由历史测试短板启发，原测试已重复使用，不能称全新盲测。</p>')
    review_path=root/'visual_review_v4/receipt.json'
    if review_path.exists():
        review=json.loads(review_path.read_text())
        sections.append('<h2>视觉大模型复核实测</h2><p>12包各选2例，共'+str(review['requested'])+'例；保存'+str(review['received'])+'份结果，'+str(review['validated'])+'份通过复核协议校验，生成'+str(review['fine_candidates'])+'个细分类待复核候选。模型为API提供的qwen3.8-chat。其余答复记录为不可用，不视作检查通过。没有独立答案，未建立F1提升结论。</p>')
    spatial=root/'spatial_comparison_v1/comparison.json';candidate_spatial=root/'candidate_comparison_v1/comparison.json';candidate_sr=[]
    if spatial.exists():
        spatial_result=json.loads(spatial.read_text());sr=[]
        for name,values in spatial_result['metrics'].items():
            if not name.endswith('|test'):continue
            for family,m in values.items():sr.append([name.removesuffix('|test'),family,m['records'],pct(m['positive_mean_iou']),pct(m['positive_mean_dice']),pct(m['empty_prediction_rate'])])
        sections.append('<h2>严格空样本误报约束下的定位比较</h2>'+table(['方案','数据类型','图像数','阳性平均IoU','阳性平均Dice','空样本误报率'],sr))
        sections.append('<p>0 IoU/Dice可能是阈值抑制了全部目标，不能称作定位成功。EPB仅1个空标注验证样本，该类的约束容忍1次误报，证据量不足。</p>')
    else:
        state=root/'localizer_v2/state.json';status=json.loads(state.read_text()) if state.exists() else {}
        sections.append('<h2>定位训练尚未完成</h2><pre>'+html.escape(json.dumps(status,ensure_ascii=False,indent=2))+'</pre>')
    if candidate_spatial.exists():
        candidate_result=json.loads(candidate_spatial.read_text())
        for name,values in candidate_result['metrics'].items():
            if not name.endswith('|test'):continue
            for family,m in values.items():candidate_sr.append([name.removesuffix('|test'),family,m['records'],pct(m['positive_mean_iou']),pct(m['positive_mean_dice']),pct(m['empty_prediction_rate']),pct(m['foreground_pixel_dice'])])
        sections.append('<h2>供疑点调查的定位配置：验证集按像素Dice选阈值</h2>'+table(['方案','数据类型','图像数','阳性IoU','阳性Dice','空样本误报率','前景像素Dice'],candidate_sr))
        sections.append('<p>候选调查使用此策略。它保留更多区域，同时明确显示误报代价；模型输出不作为科学确认，不扣确认分。数据与第一张定位表完全相同，区别是阈值选择目标。</p>')
    road=root/'roadshow_v2/package_summary.csv'
    if road.exists():
        rr=list(csv.DictReader(road.open(encoding='utf-8-sig')))
        sections.append('<h2>路演12包：确认项与模型候选</h2>'+table(['包','样本数','已确认问题样本','模型候选样本','规则技术分'],[[r['package'],r['records'],r['confirmed_problem_samples'],r['model_candidate_samples'],f"{float(r['observed_technical_score']):.3f}"] for r in rr]))
        sections.append('<p>规则技术分评价数据包已确认问题比例，不是软件比赛得分。分类概率不等于问题个数或真实污染率；待复核模型候选不直接扣分。没有独立路演答案，P/R/F1为空。</p>')
    grounded=root/'grounded_roadshow_v1/complete.json'
    if grounded.exists():
        g=json.loads(grounded.read_text());gr=[]
        for r in g['packages']:gr.append([r['package'],r['route'],r['records'],r['executed'],html.escape(json.dumps(r['located_candidate_instances'],ensure_ascii=False))])
        sections.append('<h2>路演定位分支实际完成量</h2>'+table(['包','验证集选择的定位模型','总样本数','经分类筛选后执行定位','已定位模型候选实例'],gr))
        risk_path=root/'grounded_roadshow_v1/candidate_risk.csv'
        if risk_path.exists():
            risks=list(csv.DictReader(risk_path.open(encoding='utf-8-sig')))
            shown=[[r['package'],r['defectType'],r['candidateInstances'],r['candidateSamples'],f"{float(r['observedWeightedBurdenPerAllSamples']):.5f}"] for r in risks if int(r['candidateInstances'])]
            sections.append('<h2>候选实例数与加权风险</h2>'+table(['包','类别','候选实例','候选样本','Σ实例概率/全包样本数'],shown))
            sections.append('<p>此处落实“问题个数×概率÷总样本数”的候选风险口径。它不是已确认污染比例，实例风险可能超过1。严重程度另按区域误差、覆盖率及相同单位排序。</p>')
            for filename in ['candidate_risk.csv','candidate_severity.csv']:
                source=root/'grounded_roadshow_v1'/filename
                if source.exists():shutil.copyfile(source,out/filename)
        examples=[]
        for folder in sorted((root/'grounded_roadshow_v1/packages').iterdir()):
            selected=None
            with (folder/'samples.jsonl').open() as fp:
                for line in fp:
                    ledger=json.loads(line)
                    located=[x for x in ledger['findings'] if x.get('sourceKind')=='independent_RGB_model']
                    if located:selected=(ledger,located);break
            if selected:
                import base64
                from PIL import Image,ImageDraw,ImageOps
                ledger,located=selected;path=Path(ledger['dataPath'])
                with Image.open(path) as im:
                    native=im.size;preview=ImageOps.contain(im.convert('RGB'),(768,768))
                draw=ImageDraw.Draw(preview)
                for issue in located[:12]:
                    locator=issue['locator']; polygon=locator.get('polygon')
                    if isinstance(polygon,list) and len(polygon)>=3:
                        points=[(x/native[0]*preview.width,y/native[1]*preview.height) for x,y in polygon]
                        draw.polygon(points,outline='#ffbf00')
                        anchor=points[0]
                    else:
                        x0,y0,x1,y1=locator['bbox'];points=[(x0/native[0]*preview.width,y0/native[1]*preview.height),(x1/native[0]*preview.width,y0/native[1]*preview.height),(x1/native[0]*preview.width,y1/native[1]*preview.height),(x0/native[0]*preview.width,y1/native[1]*preview.height)];draw.line(points+[points[0]],fill='#ffbf00',width=3);anchor=points[0]
                    draw.text((anchor[0],max(0,anchor[1]-12)),issue['defectType'],fill='#ffbf00')
                media=out/'previews';media.mkdir(exist_ok=True);image=media/(folder.name+'.jpg');preview.save(image,quality=90)
                encoded=base64.b64encode(image.read_bytes()).decode()
                examples.append('<article><h3>'+html.escape(folder.name)+'</h3><img style="max-width:100%;max-height:450px" src="data:image/jpeg;base64,'+encoded+'"><pre>'+html.escape(json.dumps(located[0],ensure_ascii=False,indent=2))+'</pre></article>')
        if examples:sections.append('<h2>原图位置示例：黄色为待复核模型候选</h2>'+''.join(examples))
    sections.append('<p>来源追溯、许可与分布等维度缺少的外部证据，不会因八头训练自动补齐。官方关键缺陷召回与完整定位F1仍需独立参考。</p>')
    live='<details><summary>实时训练与接续状态（服务器页面每20秒更新）</summary><pre id="live">读取中；离线文件以文中快照为准。</pre></details><script>let completed=false;async function progress(){try{let d=await (await fetch("status.json?t="+Date.now())).json();document.getElementById("live").textContent=JSON.stringify(d,null,2);if(d.completion?.stage==="complete"&&!completed){completed=true;document.getElementById("live").textContent+="\\n完整结果已生成，刷新页面查看。";}}catch{}}progress();setInterval(progress,20000);</script>'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SpaceQA 八类模型比较</title><style>body{font:15px/1.7 "Microsoft YaHei",sans-serif;background:#f3f6fa;color:#203247;margin:0;padding:24px}main{max-width:1500px;margin:auto;background:white;padding:28px;border-radius:12px}h1{margin:0}h2{font-size:20px;margin-top:32px}p{color:#506277}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;border-bottom:1px solid #dbe3ec;text-align:left}th{background:#edf3fa}.scroll{overflow:auto}.rank{display:inline-block;padding:4px 8px;border-radius:5px;white-space:nowrap}pre{background:#edf3fa;padding:15px;white-space:pre-wrap}</style><main><h1>SpaceQA · 八类细分类与定位</h1><p>完整30包验证，保留各类分母、未知项与真实模型选择。</p>'+live+''.join(sections)+'</main></html>'
    (out/'classification.html').write_text(page,encoding='utf-8')
    md=['# 八类细分类：实测结果与完整结构','', '已完成三组八头模型训练、全30包测试、12包路演分类。旧HGB7与服务保留。'+('两组定位模型训练、新旧定位测试和路演候选定位也已完成。' if grounded.exists() else '定位训练和实测结果以独立记录为准。'),'',
        '|方案|精确率|召回率|F1|TP|FP|FN|','|---|---|---|---|---|---|---|']
    md+=['|'+'|'.join(map(str,r))+'|' for r in main_rows]
    md+=['','以上采用相同严格配对掩码，排除CME候选身份时间标签，仍是P0相对技术监督，未计定位正确性。加入CME弱时间标签的组合F1为'+pct(weak['F1'])+'。ULF时间项是观测窗口端点声明，不能当成事件起止真值。','',
        '## 八类分别表现','','|类别|阳性数|已知项|未知/不适用|精确率|召回率|F1|','|---|---|---|---|---|---|---|']
    md+=['|'+'|'.join(map(str,r))+'|' for r in details]
    md+=['','较难的五类子集组合P/R/F1为：'+ ' / '.join(scientific[-1][1:4])+'。漏标是当前最弱项，不能用综合F1替代该类结果。']
    if ablation_rows:
        md+=['','## 漏标加权对照','','|配置|验证选择集F1（含弱时间）|严格测试P|严格测试R|严格测试F1|','|---|---|---|---|---|']
        md+=['|'+'|'.join(map(str,r))+'|' for r in ablation_rows]
        md+=['','保留fusion_v2；加权实验召回增加，但误报增加、验证选择集表现下降。']
    if spatial.exists():
        md+=['','## 定位模型实测','','|配置|数据类型|图像数|阳性IoU|阳性Dice|空样本误报率|','|---|---|---|---|---|---|']
        md+=['|'+'|'.join(map(str,r))+'|' for r in sr]
        md+=['','按验证集冻结选择：`'+json.dumps(spatial_result['selected_on_validation'],ensure_ascii=False)+'`。这测量P0弱掩膜定位，尚不是八类缺陷实例匹配F1。']
    if candidate_sr:
        md+=['','## 候选调查的定位配置（按验证集像素Dice选择）','','|配置|类型|图像数|阳性IoU|阳性Dice|空样本误报率|像素Dice|','|---|---|---|---|---|---|---|']
        md+=['|'+'|'.join(map(str,r))+'|' for r in candidate_sr]
        md+=['','路演调查配置：`'+json.dumps(candidate_result['selected_on_validation'],ensure_ascii=False)+'`。这是候选定位，误报另列，未获科学确认。']
    if grounded.exists():
        md+=['','## 路演定位实际完成量','','|包|定位模型|总样本|执行定位|区域候选数|','|---|---|---|---|---|']
        md+=['|'+'|'.join(map(str,[r['package'],r['route'],r['records'],r['executed'],json.dumps(r['located_candidate_instances'],ensure_ascii=False)]))+'|' for r in g['packages']]
        md+=['','模型输出原图坐标、JSON位置、区域交并与覆盖程度；仍属待复核，不扣确认分。CME与ULF走参数/时间分支，不进入图像像素模型。']
    md+=['',
        '八类分别为缺文件、边界位置、类别、时间声明、尺寸、漏标、虚标、覆盖不足/过度。同一样本可以多类阳性。','',
        '完整结构、参数、数据划分与启动方式见 [模型结构与训练说明](模型结构与训练说明.md)。全30包和各细分类指标可在 [交互结果表](classification.html) 查看。','',
        '官方关键召回、路演定位F1仍没有独立参考答案。路演预测与科学确认分开，模型候选不直接计入确认扣分。']
    (out/'分类结果与当前状态.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    shutil.copyfile(Path(__file__).with_name('README.md'),out/'模型结构与训练说明.md')
    for filename in ['all30_test.csv','all30_classes_test.csv','all30_strict_mask_test.csv','all30_classes_strict_mask_test.csv','comparison.json','frozen_policy.json']:
        shutil.copyfile(fusion/filename,out/filename)
    for filename in ['package_summary.csv','ranking.csv']:
        if (root/'roadshow_v2'/filename).exists():shutil.copyfile(root/'roadshow_v2'/filename,out/('roadshow_'+filename))
    print(json.dumps(dict(output=str(out),status='rendered',classification='complete',localization='complete' if spatial.exists() else 'pending')))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/staff/tangjiarui/OPC/eight_class_20261104'))
    ap.add_argument('--output',type=Path,required=True);main(ap.parse_args())
