"""Complete, concise current-only inputs; never inspect P0 or pollution tiers."""
import hashlib
import json
from pathlib import Path

from common import OUT, ann, digest, oldprep


def adapt(row):
    r=dict(row)
    if r['family'] not in ['EPB_seg_sample','DMSP_SSUSI']:
        label=json.loads(Path(r['label_path']).read_text(encoding='utf-8-sig'))
        extracted=ann.extract_annotation(label,r['family']);basic=label.get('basic_annotation',{});science=label.get('science_annotation',{})
        rawkey=r.get('data_sha256')
        path=OUT/'raw_views'/(str(rawkey)+'.json')
        stats=json.loads(path.read_text()) if path.exists() else {'input_missing':True}
        stats.pop('verified_source_sha256',None)
        summary=dict(family=r['family'],current_classes=[o['class_label'] or o['raw'].get('kind') for o in extracted['objects']],
            decoded_observation=stats,declared_dimensions=extracted['declared_dimensions']['sources'])
        if r['family']=='CME_C2_sample':
            summary['frame_time']=basic.get('startTime') or science.get('dateObs')
            summary['event_parameters']=[{k:v for k,v in o['raw'].items() if k in ['event_type','cme_time','central_pa','width','mpa']} for o in extracted['objects']]
        if r['family']=='ULF_sample':
            summary['declared_window']={k:basic.get(k) for k in ['startTime','endTime']}
            summary['declared_parameters']={k:science.get(k) for k in ['dateObs','sampling_rate_hz','duration_s','n_points','components','units']}
            summary['current_events']=[{k:v for k,v in o['raw'].items() if k in ['kind','start_dt','end_dt']} for o in extracted['objects']]
        r['prompt']=(oldprep.PROMPT+' For AIA/HMI, annotations mark Active Regions. For CME, annotations describe event times and angular parameters. '
            'For ULF, panels show CSV waveforms and current windows; temporal/numeric parameter errors belong to D. '
            'Current input only: '+json.dumps(summary,separators=(',',':')))
        r['input_sha256']=digest([hashlib.sha256(Path(r['image']).read_bytes()).hexdigest(),r['prompt']])
    r['input_profile']='full-family-compact-v1'
    return r
