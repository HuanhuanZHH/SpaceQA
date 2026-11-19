"""Small decision artifacts only; no images, weights or raw arrays are exported."""
import csv
import json
from datetime import datetime, timezone

import numpy as np

from common import OUT, dump, lines
from evaluate import selection_key


def pct(v):
    return '—' if v is None else f'{100*v:.2f}%'


def main():
    results = json.loads((OUT/'evaluation_results.json').read_text())
    selection = json.loads((OUT/'selection.json').read_text())
    validation = json.loads((OUT/'validation.json').read_text()) if (OUT/'validation.json').exists() else {}
    states = {n: json.loads((OUT/n).read_text()) for n in ['state.json', 'followup_state.json'] if (OUT/n).exists()}
    rows = lines(OUT/'dataset.jsonl')
    table = []
    classes = []
    primary = sorted([n for n, r in results.items() if r['feature_set'] == 'primary'], key=lambda n: selection_key(results[n]), reverse=True)
    extended = sorted([n for n, r in results.items() if r['feature_set'] == 'instance_extended'], key=lambda n: selection_key(results[n]), reverse=True)
    for n, r in results.items():
        c = r['calibrations']['tier_balanced']
        row = dict(configuration=n, validation_mean_f1=c['validation_mean_tier_f1'],
                   validation_worst_f1=c['validation_min_tier_f1'], weights_bytes=r['weights_bytes'])
        for split in ['val', 'test']:
            for tier in ['P1', 'P2', 'P3', 'P4']:
                m = c['cohorts'][split+':ALL:'+tier]
                for key in ['micro_f1', 'precision', 'recall', 'clean_fpr', 'samples', 'tp', 'fp', 'fn']:
                    row[split+'_'+tier+'_'+key] = m[key]
                for family in ['ALL', 'EPB_seg_sample', 'DMSP_SSUSI']:
                    mm = c['cohorts'][split+':'+family+':'+tier]
                    for code, values in mm['per_class'].items():
                        classes.append(dict(configuration=n, split=split, family=family, tier=tier, code=code, **values))
        row['test_mean_tier_f1'] = float(np.mean([row['test_'+t+'_micro_f1'] for t in ['P1', 'P2', 'P3', 'P4']]))
        table.append(row)
    for name, data in [('decision_table.csv', table), ('class_comparison.csv', classes)]:
        with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    out = ['# OPC 图像模型与 P1～P4 逐档实测', '',
        '生成时间：'+datetime.now(timezone.utc).isoformat(), '',
        '**范围：EPB、DMSP；几何偏差、漏标、虚标三个空间类别。以下 F1 是多标签 micro F1，不是比赛总分或官方类别加定位得分。**', '',
        '训练/验证/测试按同源样本、观测日和重复内容隔离，共 1,623 个同源样本、7,799 条有效档位记录。316 个缺图记录保留在统计中，不用于证明语义漏标。测试集此前已被查看，本报告属于探索性回归，尚无独立专家科学真值。', '',
        '统一模型和阈值按验证集 P1～P4 F1 等权平均选择，再比较最弱档 F1、P1 无空间缺陷样本误报率和模型大小；误报约束为验证 P1 ≤1%。污染档位只用于评估分组，不作为模型输入，也不据此切换模型。', '',
        '## 完成情况', '',
        *['- '+n+'：'+s['status'] for n, s in states.items()],
        '- 产物核查：'+('通过' if validation.get('passed') else '尚未通过/待完成')+'；检查数 '+str(validation.get('checks', 0))+'。检查通过只证明可复算和数据隔离，不证明科学准确率。', '',
        '## 主比较：相同 35 个特征', '',
        '所有图像路线使用当前标注的 21 个结构量与图像预测残差的 14 个量，训练三个独立 ExtraTrees 分类器。annotation_only 使用 21 个标注结构量，它也是学习模型。正式规则加可信 clean 对照不等同于这个对照组。', '',
        '| 配置 | 验证四档均值 | 验证最弱档 | 测试 P1 | 测试 P2 | 测试 P3 | 测试 P4 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def add_table(names):
        for n in names:
            c = results[n]['calibrations']['tier_balanced']
            out.append('| '+n.replace('__primary', '').replace('__instance_extended', '')+' | '+
                ' | '.join([pct(c['validation_mean_tier_f1']), pct(c['validation_min_tier_f1'])]+
                    [pct(c['cohorts']['test:ALL:'+t]['micro_f1']) for t in ['P1', 'P2', 'P3', 'P4']])+' |')
    add_table(primary)
    winner = selection['primary_selected']
    out += ['', '验证协议选择的主配置：`'+winner+'`。这个选择不等于它在四档测试上均领先。', '',
        '## 加入实例匹配特征的次级比较', '',
        '在相同 35 个量之上加 17 个固定实例匹配量，包括目标数量、不匹配数量、IoU、置信度、中心偏差和面积比。不能把此表的增益全部归因于网络结构。', '',
        '| 配置 | 验证四档均值 | 验证最弱档 | 测试 P1 | 测试 P2 | 测试 P3 | 测试 P4 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    add_table(extended)
    out += ['', '## 各档样本与缺陷数量', '',
        '| 测试档位 | 记录数 | 几何 | 漏标 | 虚标 |', '|---|---:|---:|---:|---:|']
    for tier in ['P1', 'P2', 'P3', 'P4']:
        m = results[winner]['calibrations']['tier_balanced']['cohorts']['test:ALL:'+tier]
        out.append('| '+tier+' | '+' | '.join(map(str, [m['samples']]+[m['per_class'][c]['support'] for c in ['L2-G', 'L3-06', 'L3-07']]))+' |')
    out += ['', 'P1 测试仅有几何 5、漏标 1、虚标 1 个空间正例；P1 验证没有漏标正例。P1 的一次命中或误报会明显改变 F1，不能凭这些数字宣布轻微污染已经解决。', '',
        '## 结构与调用', '',
        '- YOLO11n 检测：原图→候选框；与当前标注框比较。框填充仅作为统一残差特征，不能解释为精确像素分割。',
        '- YOLO11n-seg、YOLOv8n-seg：原图→实例框、实例掩膜和置信度。原 COCO 权重仅提供类别无关候选，训练后才适配本任务目标。',
        '- Attention U-Net：在 U-Net 跳跃连接中加入注意力门；U-Net++：通过嵌套跳跃连接融合不同层级。',
        '- ResNet18 U-Net：预训练编码器加逐级解码器；DeepLabV3 MobileNet：轻量预训练骨干加多尺度空洞卷积分割。',
        '- 既有 DINO/I-JEPA：冻结视觉编码器加训练空间头；crossfit U-Net 使用组外预测训练残差分类器。',
        '- 推理输入为当前原图和当前标注；图像模型预测独立目标区域，再与标注匹配、分类候选并留存证据。评分、比例、排序仍由确定性程序负责。', '',
        '## 配置与稳定性', '',
        '首轮新架构均为 448 像素、24 轮；按验证集选择两个新架构，再比较边界/定位损失、48 轮和两个额外随机种子。detail：语义分割增加可微形态边界 BCE；YOLO 提高 box/dfl 权重，分割 mask_ratio 改为 2。每种配置仍以 P0 验证 Dice 或框架 P0 验证 mAP 选择权重。', '',
        '| 架构 | 默认及两种子完成数 | 测试四档均值：均值±标准差 |', '|---|---:|---:|']
    architectures = selection['tuning_architectures']
    for arch in architectures:
        names = [arch+'__primary', arch+'__seed20261101__primary', arch+'__seed20261102__primary']
        values = [np.mean([results[n]['calibrations']['tier_balanced']['cohorts']['test:ALL:'+t]['micro_f1'] for t in ['P1', 'P2', 'P3', 'P4']]) for n in names if n in results]
        out.append('| '+arch+' | '+str(len(values))+' | '+pct(float(np.mean(values)))+' ± '+pct(float(np.std(values)))+' |')
    out += ['', '## 时间、大小与正式接入', '',
        '本轮训练时间不属于官方单包 3 小时推理限制。权重体积与全部 1,623 张缓存图的网络推理时间如下；该耗时不含原图解码、标注匹配、规则、评分和报告。训练峰值显存也不能替代推理显存。', '',
        '| 模型/配置 | 权重 MB | 训练秒 | 缓存/缩放图推理秒 |', '|---|---:|---:|---:|']
    for path in sorted((OUT/'training').glob('*.json')):
        r = json.loads(path.read_text())
        secs = r.get('cached_raw_inference_seconds', r.get('resized_png_inference_seconds'))
        out.append(f"| {path.stem} | {r['weights_bytes']/1e6:.2f} | {r['training_seconds']:.2f} | {secs:.2f} |")
    out += ['', '测试硬件为 A100 40GB。原零样本 YOLO 与前两项 YOLO11 训练曾使用 GPU0，与已有服务共享；后续已修正到 GPU1，不能把前几项耗时当作隔离基准。原 8001 服务保留。', '',
        '目前未完成新候选在官方 A10 上的完整单包验收，也没有用本轮权重构建最终 ≤20GB 的离线 Docker 镜像。已有 U-Net448 分支在 A100 上的 EPB P1 1,350 条/45.39 秒、DMSP P1 286 条/5.96 秒仅为此前候选分支实测。', '',
        'AIA/HMI 存在图像与标注同步变换，不能直接套用本轮同图真值；CME、ULF 也未纳入此次空间网络训练。因此不能将此处排名称为全部六类数据的最终最优配置。', '',
        '## 查看和复算', '',
        '`'+str(OUT)+'`', '',
        '- `decision_table.csv`：各配置、各档 F1/精确率/召回率/误报率。',
        '- `class_comparison.csv`：按数据类型、档位、缺陷类别的 TP/FP/FN/F1。',
        '- `tier_comparison.csv`：三种门控协议的完整比较；本报告使用 tier_balanced。',
        '- `evaluation_results.json`、`selection.json`、`validation.json`：指标、验证选择与独立复算。',
        '- `state.json`、`followup_state.json`、`training/`：任务状态与训练轨迹。',
        '- 正式评分程序没有接入本轮实验模型。', '']
    (OUT/'decision_report.md').write_text('\n'.join(out), encoding='utf-8')
    dump('report_summary.json', dict(primary_selected=winner, extended_selected=selection['extended_selected'],
        initial_state=states.get('state.json', {}).get('status'), followup_state=states.get('followup_state.json', {}).get('status'),
        verified=validation.get('passed', False), configurations=len(results), primary_configurations=len(primary),
        selected_test={t: results[winner]['calibrations']['tier_balanced']['cohorts']['test:ALL:'+t] for t in ['P1', 'P2', 'P3', 'P4']}))
    print(json.dumps(dict(report=str(OUT/'decision_report.md'), configurations=len(results))))


if __name__ == '__main__':
    main()
