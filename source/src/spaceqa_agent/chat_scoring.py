"""Reproducible, explicitly scoped six-dimensional technical observations.

These checks measure the supplied files and metadata. They do not certify
scientific annotation truth, population representativeness, or legal permission.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct

VERSION='six-observation-20261106-v2'
DIMENSIONS={
 'completeness':('完整性','文件与标注可读取'),
 'annotation_quality':('标注质量','类别、几何及事件时段结构一致性'),
 'sample_correspondence':('样本对应','文件名、引用和尺寸对应'),
 'coverage_distribution':('覆盖与分布','已声明数量、类别和时间范围一致性'),
 'documentation_provenance':('文档与来源追溯','标注头、版本与源文件追溯'),
 'usage_conditions':('使用条件','离线检查工具的技术接入条件'),
}
RULES={
 'S6-C1':('completeness','引用数据文件存在'),
 'S6-C2':('completeness','标注JSON可解析'),
 'S6-A1':('annotation_quality','对象类别符合当前任务词表'),
 'S6-A2':('annotation_quality','给定几何顶点有限且面积非零'),
 'S6-A3':('annotation_quality','给定事件起止可解析且顺序有效'),
 'S6-P1':('sample_correspondence','标注声明文件名与数据文件对应'),
 'S6-P2':('sample_correspondence','登记数据引用可定位到当前包'),
 'S6-P3':('sample_correspondence','显式声明尺寸与实测尺寸一致'),
 'S6-D1':('coverage_distribution','声明对象数量等于实际记录数量'),
 'S6-D2':('coverage_distribution','声明类别集合与对象类别集合一致'),
 'S6-D3':('coverage_distribution','声明观测起止可解析且顺序有效'),
 'S6-T1':('documentation_provenance','标注头记录文件名'),
 'S6-T2':('documentation_provenance','标注版本可定位'),
 'S6-T3':('documentation_provenance','标注生成时间可解析'),
 'S6-T4':('documentation_provenance','标注生成方法已记录'),
 'S6-T5':('documentation_provenance','当前标注SHA256与登记值一致'),
 'S6-U1':('usage_conditions','当前数据格式符合图像/时序输入约定'),
 'S6-U2':('usage_conditions','数据头可读取并识别'),
 'S6-U3':('usage_conditions','模态声明与当前文件类型一致'),
}
VOCAB={'AIA_AR_sample':{'AR'},'HMI_AR_sample':{'AR'},'CME_C2_sample':{'CME'},'EPB_seg_sample':{'EPB'}}

def walk(x,p=''):
    if isinstance(x,dict):
        yield p,x
        for k,v in x.items():yield from walk(v,p+'/'+str(k).replace('~','~0').replace('/','~1'))
    elif isinstance(x,list):
        for i,v in enumerate(x):yield from walk(v,p+'/'+str(i))

def timestamp(x):
    if not isinstance(x,str) or not x.strip():return False
    try:datetime.fromisoformat(x.strip().replace('Z','+00:00'));return True
    except ValueError:return False

def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)

def sample_checks(row,root):
    root=Path(root).resolve();sid=row['sampleId']
    label=Path(row['labelPath']) if isinstance(row.get('labelPath'),str) and row['labelPath'] else None
    data=Path(row['dataPath']) if isinstance(row.get('dataPath'),str) and row['dataPath'] else None
    result=[]
    def add(code,status,detail,pointer=None):
        result.append(dict(sampleId=sid,dimension=RULES[code][0],checkId=code,ruleVersion=VERSION,
          status=status,detail=detail,locator=dict(label_path=str(label) if label else None,data_path=str(data) if data else None,annotation_pointer=pointer),
          labelSha256=row.get('labelSha256')))
    def boolean(code,value,detail,pointer=None):add(code,'pass' if value else 'fail',detail,pointer)
    if label is None or data is None:
        for code in RULES:add(code,'undetermined','登记台账未提供文件路径，未执行该项检查')
        return result
    try:
        label.resolve().relative_to(root);data.resolve().relative_to(root)
    except ValueError:
        for code in RULES:add(code,'undetermined','登记路径超出当前包，未读取')
        return result
    present=data.is_file();boolean('S6-C1',present,dict(exists=present))
    try:
        if label.stat().st_size>32*1024**2:raise ValueError('oversized annotation')
        raw=label.read_bytes();ann=json.loads(raw.decode('utf-8-sig'))
        if not isinstance(ann,dict):raise ValueError('annotation must be object')
    except (OSError,ValueError,UnicodeError):
        boolean('S6-C2',False,'JSON缺失或解析失败')
        for code in RULES:
            if code not in {'S6-C1','S6-C2'}:add(code,'undetermined','标注无法解析，未进行该项检查')
        return result
    boolean('S6-C2',True,'JSON对象解析成功')
    objects=[];collections=[]
    for pointer,obj in walk(ann):
        for key in ['matched_events','shapes']:
            if isinstance(obj.get(key),list):
                collections.append((pointer,obj,key))
                objects += [(pointer+'/'+key+'/'+str(i),v) for i,v in enumerate(obj[key]) if isinstance(v,dict)]
    family=row.get('family');allowed=VOCAB.get(family)
    categories=[];bad_geometry=[];geometry_count=0
    for pointer,obj in objects:
        category=next((obj[k] for k in ['label','event_type','kind'] if isinstance(obj.get(k),str)),None)
        if category is not None:categories.append(category)
        points=obj.get('bbox_polygon_pixel',obj.get('points'))
        if points is not None:
            geometry_count+=1
            valid=isinstance(points,list) and len(points)>=3 and all(isinstance(p,list) and len(p)>=2 and all(finite(v) for v in p[:2]) for p in points)
            if valid:valid=abs(sum(points[i][0]*points[(i+1)%len(points)][1]-points[(i+1)%len(points)][0]*points[i][1] for i in range(len(points))))>1e-8
            if not valid:bad_geometry.append(pointer)
    if allowed is not None and categories:boolean('S6-A1',all(c in allowed for c in categories),dict(actual=sorted(set(categories)),allowed=sorted(allowed)))
    else:add('S6-A1','not_applicable','当前无类别对象，或未配置可核验词表')
    if geometry_count:boolean('S6-A2',not bad_geometry,dict(objects=geometry_count,invalid=bad_geometry[:8]))
    else:add('S6-A2','not_applicable','当前对象未提供图像几何')
    timed=[(pointer,obj) for pointer,obj in objects if 'start_dt' in obj or 'end_dt' in obj or family=='ULF_sample']
    if timed:
        invalid=[];ambiguous=[]
        for pointer,obj in timed:
            start,end=obj.get('start_dt'),obj.get('end_dt')
            if not timestamp(start) or not timestamp(end):invalid.append(pointer);continue
            a=datetime.fromisoformat(start.strip().replace('Z','+00:00'));b=datetime.fromisoformat(end.strip().replace('Z','+00:00'))
            try:
                if a>b:invalid.append(pointer)
            except TypeError:ambiguous.append(pointer)
        detail=dict(objects=len(timed),invalid=invalid[:8],timezone_ambiguous=ambiguous[:8],scope='结构有效性，不验证科学真实起止或抖动')
        if invalid:boolean('S6-A3',False,detail)
        elif ambiguous:add('S6-A3','undetermined',detail)
        else:boolean('S6-A3',True,detail)
    else:add('S6-A3','not_applicable','当前未提供事件时段对象')
    basic=ann.get('basic_annotation') or {};header=ann.get('label_header') or {}
    declared=basic.get('fileName',header.get('fileName'))
    if declared:boolean('S6-P1',Path(str(declared).replace('\\','/')).name==data.name,dict(declared=declared,actual=data.name),'/basic_annotation/fileName')
    else:add('S6-P1','undetermined','标注中无可核对的文件名')
    boolean('S6-P2',present,dict(expected=str(data),exists=present))
    dimcheck=(row.get('checks') or {}).get('L2-F') or {}
    status=dimcheck.get('scopeStatus',dimcheck.get('status','undetermined'))
    add('S6-P3',status,dimcheck.get('reason','没有显式声明尺寸'))
    found_count=False;found_types=False
    for pointer,obj,key in collections:
        if 'event_count' in obj:
            found_count=True;boolean('S6-D1',obj['event_count']==len(obj[key]),dict(declared=obj['event_count'],actual=len(obj[key])),pointer+'/event_count')
        if isinstance(obj.get('event_types'),list):
            found_types=True;actual={next(v[k] for k in ['event_type','kind','label'] if isinstance(v.get(k),str)) for v in obj[key] if isinstance(v,dict) and any(isinstance(v.get(k),str) for k in ['event_type','kind','label'])}
            boolean('S6-D2',set(obj['event_types'])==actual,dict(declared=obj['event_types'],actual=sorted(actual)),pointer+'/event_types')
    # Polygon schemas have no separate summary count/type fields: structural
    # inventory statistics are still emitted, without inventing an expected mix.
    if not found_count:add('S6-D1','not_applicable','无独立声明数量；已统计当前对象数量，不假设期望数量')
    if not found_types:add('S6-D2','not_applicable','无独立声明类别集合；不假定均衡分布')
    interval=row.get('observationInterval')
    if isinstance(interval,list) and len(interval)==2:
        good=all(timestamp(t) for t in interval)
        if good:
            try:good=datetime.fromisoformat(interval[0].replace('Z','+00:00'))<=datetime.fromisoformat(interval[1].replace('Z','+00:00'))
            except TypeError:good=False
        boolean('S6-D3',good,dict(interval=interval))
    else:add('S6-D3','undetermined','缺少声明观测时段；不能判断覆盖范围')
    boolean('S6-T1',bool(header.get('fileName')),dict(value=header.get('fileName')),'/label_header/fileName')
    boolean('S6-T2',bool(header.get('labelVersion')),dict(value=header.get('labelVersion')),'/label_header/labelVersion')
    create=header.get('labelCreateTime',header.get('labelCreatTime'))
    boolean('S6-T3',timestamp(create),dict(value=create),'/label_header')
    boolean('S6-T4',bool(header.get('annotationMethod')),dict(value=header.get('annotationMethod')),'/label_header/annotationMethod')
    expected=row.get('labelSha256');actual=hashlib.sha256(raw).hexdigest()
    if expected:boolean('S6-T5',actual==expected,dict(expected=expected,actual=actual))
    else:add('S6-T5','undetermined','缺登记哈希；没有用现算哈希自证已有链路')
    ext=data.suffix.lower();image=ext in {'.png','.jpg','.jpeg','.fits','.fit'};series=ext=='.csv'
    boolean('S6-U1',image or series,dict(extension=ext,scope='离线质量检查工具接入'))
    readable=False
    if present:
        try:
            with data.open('rb') as stream:head=stream.read(64)
            readable=bool(head)
            if ext=='.png':readable=len(head)>=24 and head[:8]==b'\x89PNG\r\n\x1a\n' and all(struct.unpack('>II',head[16:24]))
            elif ext in {'.jpg','.jpeg'}:readable=head[:2]==b'\xff\xd8'
            elif ext in {'.fits','.fit'}:readable=head.startswith(b'SIMPLE')
            elif ext=='.csv':head.decode('utf-8-sig');readable=bool(head.strip())
        except (OSError,UnicodeError,struct.error):readable=False
    boolean('S6-U2',readable,dict(data_present=present,header_readable=readable,scope='文件头，不等于全量像素科学审查'))
    modality=str(basic.get('modality','')).lower()
    if modality:boolean('S6-U3',(image and modality in {'image','images'}) or (series and modality in {'timeseries','time_series','time series','table','tabular','csv','series'}),dict(declared=modality,actual='image' if image else 'csv'),'/basic_annotation/modality')
    else:add('S6-U3','undetermined','未声明模态，不默认判为通过')
    return result

def evaluate(records,root,evidence_path=None):
    stats={k:Counter() for k in DIMENSIONS};by_rule={k:Counter() for k in RULES};examples={k:[] for k in RULES}
    evidence=gzip.open(evidence_path,'wt',encoding='utf-8') if evidence_path else None
    n=0
    try:
        for record in records:
            n+=1
            for check in sample_checks(record,root):
                stats[check['dimension']][check['status']]+=1;by_rule[check['checkId']][check['status']]+=1
                if len(examples[check['checkId']])<2 or (check['status']=='fail' and not any(e['status']=='fail' for e in examples[check['checkId']])):examples[check['checkId']].append(check)
                if evidence:evidence.write(json.dumps(check,ensure_ascii=False)+'\n')
    finally:
        if evidence:evidence.close()
    dims={}
    for key,(name,scope) in DIMENSIONS.items():
        c=stats[key];app=c['pass']+c['fail']+c['undetermined'];done=c['pass']+c['fail']
        dims[key]=dict(id=key,name=name,displayScore=100*c['pass']/done if done else None,
          scoreKind='scoped_technical_observation',scoreLabel='技术观测分',scope=scope,
          scopeCheckCoverage=done/app if app else None,scopeChecksCompleted=done,scopeChecksApplicable=app,
          scopeCounts=dict(c),scoreBounds=[100*c['pass']/app,100*(c['pass']+c['undetermined'])/app] if app else None,
          criteria=[dict(checkId=k,name=label,counts=dict(by_rule[k]),examples=examples[k]) for k,(d,label) in RULES.items() if d==key],
          reason=scope+'；按实际可核验检查的通过率计分，证据未决单列，不计作通过或失败。',
          formalScore=None)
    formal_blockers = []
    if any(row.get('scopeCounts', {}).get('undetermined', 0) for row in dims.values()):
        formal_blockers.append('存在证据未决检查；未决项仍保留在适用分母')
    if dims.get('coverage_distribution', {}).get('scopeCheckCoverage', 0) < 1:
        formal_blockers.append('覆盖与分布缺少预期覆盖基准或允许偏差')
    if dims.get('documentation_provenance', {}).get('scopeCheckCoverage', 0) < 1:
        formal_blockers.append('文档与来源追溯材料不完整')
    if dims.get('usage_conditions', {}).get('scopeCheckCoverage', 0) < 1:
        formal_blockers.append('使用条件、权限或科学适用性材料不完整')
    if not any(record.get('normativeMaterialsProvided') is True for record in records):
        formal_blockers.append('未在输入清单中声明已提供规范、字段字典、用途和可比性材料')
    return dict(version=VERSION,samples=n,dimensions=dims,
      formula='100 * pass / (pass + fail); 证据未决保留在适用分母及区间中，不计为通过或失败；不适用单列',
      scope='六维技术观测画像；区别于完整科学质量分和官方成绩',
      formalEvaluation=dict(eligible=not formal_blockers, score=None if formal_blockers else sum(d['displayScore'] for d in dims.values() if isinstance(d.get('displayScore'),(int,float))) / max(1, sum(isinstance(d.get('displayScore'),(int,float)) for d in dims.values())), blockers=formal_blockers,
                            rule='六维正式分仅在六维适用检查完成、规范材料齐全且实现与规范一致时计算'),
      limitations=['覆盖分布只核对声明与记录的一致性，不证明总体代表性。','匿名化身份不扣分；追溯分仅核验标注头、版本及文件链路，不认证原始观测来源。','使用条件分仅针对离线质量检查工具的技术接入，不确认许可、再分发权或特定科学用途。'])
