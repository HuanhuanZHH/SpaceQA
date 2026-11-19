"""Pure, deterministic annotation extraction and conservative P0/variant comparison.

Changes are observations relative to P0, not scientific error verdicts. Geometry
is retained in the original label coordinate convention; this module never flips,
clips, rescales, repairs, or reads source files. JSON Pointers follow RFC 6901.
"""
import collections
import copy
import json
import math
import re

VERSION = 'annotation-diff-1.0.0'
IDENTITY_KEYS = ('kb_archivid', 'SOL_standard', 'solar_object_locator', 'object_id', 'event_id', 'identifier', 'id')
CLASS_KEYS = {'label', 'event_type', 'eventtype', 'concept', 'event_types', 'evenType'}
GEOMETRY_KEYS = {'points', 'bbox_polygon_pixel', 'shape_type'}
SCIENCE_PARAMETER_KEYS = {'central_pa', 'width', 'mpa', 'linear_speed', 'speed_2nd_initial', 'speed_2nd_final', 'speed_2nd_20R', 'acceleration', 'mass', 'kinetic_energy'}
PATHS = {
    'AIA_AR_sample': ('hek_event_annotation', 'matched_events'),
    'HMI_AR_sample': ('hek_event_annotation', 'matched_events'),
    'CME_C2_sample': ('hek_event_annotation', 'matched_events'),
    'EPB_seg_sample': ('science_annotation', 'mask', 'shapes'),
    'DMSP_SSUSI': ('science_annotation', 'specialShapes'),
}
ABSENT = object()


def _escape(x): return str(x).replace('~', '~0').replace('/', '~1')
def _pointer(parts): return ''.join('/'+_escape(x) for x in parts)
def _canon(x): return json.dumps(x, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=True)
def _numeric(x): return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
def _equal(a, b):
    if type(a) is not type(b): return False
    if isinstance(a, dict): return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, list): return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _get(x, parts, default=ABSENT):
    for k in parts:
        if not isinstance(x, dict) or k not in x: return default
        x = x[k]
    return x


def _category(key):
    low = str(key).lower()
    if key in CLASS_KEYS: return 'class_changed'
    if key in GEOMETRY_KEYS: return 'geometry_changed'
    if 'time' in low or low in {'dateobs', 'dateend', 'tobs', 'start_dt', 'end_dt'}: return 'time_changed'
    return 'metadata_changed'


def _change(before, after, before_path, after_path, category, **extra):
    r = {'pointer': _pointer(after_path if after is not ABSENT else before_path),
         'clean_pointer': _pointer(before_path) if before is not ABSENT else None,
         'variant_pointer': _pointer(after_path) if after is not ABSENT else None,
         'before_present': before is not ABSENT, 'after_present': after is not ABSENT,
         'before': None if before is ABSENT else copy.deepcopy(before),
         'after': None if after is ABSENT else copy.deepcopy(after), 'category': category}
    if before_path and before_path[-1] in SCIENCE_PARAMETER_KEYS:
        r['subcategory'] = 'science_parameters_changed'
    r.update(extra)
    return r


def _diff(a, b, pa=(), pb=(), skip=()):
    if pa == skip and skip: return []
    if a is not ABSENT and b is not ABSENT and _equal(a, b): return []
    key = pb[-1] if pb else (pa[-1] if pa else '')
    # Coordinate sequences are one structured evidence item, never millions of scalars.
    if key in CLASS_KEYS and isinstance(a, list) and isinstance(b, list):
        category = 'annotation_order_changed' if collections.Counter(map(_canon,a)) == collections.Counter(map(_canon,b)) else 'class_changed'
        return [_change(a, b, pa, pb, category)]
    if key in GEOMETRY_KEYS or a is ABSENT or b is ABSENT or type(a) is not type(b):
        return [_change(a, b, pa, pb, _category(key))]
    if isinstance(a, dict):
        out = []
        for k in sorted(a.keys() | b.keys()): out.extend(_diff(a.get(k, ABSENT), b.get(k, ABSENT), pa+(k,), pb+(k,), skip))
        return out
    if isinstance(a, list):
        out = []
        for i in range(max(len(a), len(b))):
            out.extend(_diff(a[i] if i < len(a) else ABSENT, b[i] if i < len(b) else ABSENT, pa+(i,), pb+(i,), skip))
        return out
    return [_change(a, b, pa, pb, _category(key))]


def _dimensions(label):
    sources = []
    for path in [('science_annotation', 'mask'), ('basic_annotation',), ()]:
        obj = _get(label, path, {}) if path else label
        if isinstance(obj, dict):
            for axis, field in [('width', 'imageWidth'), ('height', 'imageHeight')]:
                if field in obj: sources.append({'axis': axis, 'value': obj[field], 'pointer': _pointer(path+(field,))})
    size = _get(label, ('basic_annotation', 'imgSize'), None)
    if isinstance(size, str):
        m = re.search(r'(\d+)\s*[xX×]\s*(\d+)', size)
        if m:
            for axis, value in zip(('width', 'height'), m.groups()):
                sources.append({'axis': axis, 'value': int(value), 'pointer': '/basic_annotation/imgSize', 'parsed_from_text': True})
    ans = {'width': None, 'height': None, 'sources': sources, 'consistent': True}
    for axis in ['width', 'height']:
        vals = [x['value'] for x in sources if x['axis'] == axis]
        ans[axis] = vals[0] if vals else None
        if len({_canon(v) for v in vals}) > 1: ans['consistent'] = False
    return ans


def extract_annotation(label, family):
    path = PATHS.get(family)
    result = {'version': VERSION, 'family': family, 'schema_status': 'ok', 'warnings': [],
              'objects': [], 'object_count': 0, 'objects_pointer': _pointer(path) if path else None,
              'declared_dimensions': _dimensions(label) if isinstance(label, dict) else {'width': None, 'height': None, 'sources': [], 'consistent': True},
              'metadata': copy.deepcopy(label)}
    if path is None or not isinstance(label, dict):
        result.update(schema_status='unsupported_structure'); result['warnings'].append('unknown_family_or_nonobject_label'); return result
    objects = _get(label, path)
    if not isinstance(objects, list):
        result['schema_status'] = 'unsupported_structure'
        result['warnings'].append('expected_object_list_missing_null_or_wrong_type'); return result
    # Preserve complete metadata independently of the object collection.
    target = result['metadata']
    for k in path[:-1]: target = target[k]
    target.pop(path[-1])
    for index, raw in enumerate(objects):
        item = {'index': index, 'pointer': _pointer(path+(index,)), 'stable_id': None, 'stable_ids': {},
                'class_label': None, 'class_pointer': None, 'geometry': {'kind': 'none', 'points': None, 'valid': False, 'state': 'absent'},
                'raw': copy.deepcopy(raw), 'warnings': []}
        if not isinstance(raw, dict):
            item['warnings'].append('nonobject_annotation_item'); result['warnings'].append('nonobject_annotation_item'); result['schema_status'] = 'partially_supported'
            result['objects'].append(item); continue
        ids = {k: raw[k] for k in IDENTITY_KEYS if isinstance(raw.get(k), (str, int)) and not isinstance(raw.get(k), bool) and str(raw[k]).strip()}
        item['stable_ids'] = ids
        if ids:
            k = next(iter(ids)); item['stable_id'] = k+':'+str(ids[k])
        for k in ('label', 'event_type', 'eventtype', 'concept'):
            if k in raw:
                item['class_label'] = str(raw[k]) if raw[k] is not None else None
                item['class_pointer'] = item['pointer']+'/'+k; break
        if family == 'CME_C2_sample' and item['class_label'] is None: item['class_label'] = 'CME'
        gkey = 'bbox_polygon_pixel' if 'bbox_polygon_pixel' in raw else ('points' if 'points' in raw else None)
        if gkey:
            points = raw[gkey]; kind = raw.get('shape_type', 'polygon')
            valid = isinstance(points, list) and len(points) >= (2 if kind in ('rectangle', 'line') else 3)
            valid = valid and all(isinstance(p, (list, tuple)) and len(p) == 2 and all(_numeric(x) for x in p) for p in points)
            item['geometry'] = {'kind': kind, 'points': copy.deepcopy(points), 'valid': bool(valid),
                                'state': 'absent' if points is None else ('valid' if valid else 'invalid'),
                                'pointer': item['pointer']+'/'+gkey}
            if points is not None and not valid: item['warnings'].append('invalid_geometry')
        elif family == 'CME_C2_sample':
            params = {k: raw.get(k) for k in ('central_pa', 'width', 'mpa') if k in raw}
            item['geometry'] = {'kind': 'angular_sector', 'points': None, 'parameters': params,
                                'valid': all(_numeric(params.get(k)) for k in ('central_pa', 'width')), 'state': 'parameter_only'}
        result['objects'].append(item)
    result['object_count'] = len(objects)
    if not result['declared_dimensions']['consistent']: result['warnings'].append('declared_dimension_conflict')
    return result


def _bbox(obj):
    g = obj['geometry']
    if not g.get('valid') or not g.get('points'): return None
    xs, ys = zip(*g['points']); return min(xs), min(ys), max(xs), max(ys)


def _iou(a, b):
    a, b = _bbox(a), _bbox(b)
    if a is None or b is None: return 0.0
    aa = max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=aa+bb-inter
    return inter/union if union else 0.0


def _identity_conflict(a, b):
    return any(not _equal(a['stable_ids'][k], b['stable_ids'][k]) for k in a['stable_ids'].keys() & b['stable_ids'].keys())


def _match_objects(left, right, family):
    remaining_l=set(range(len(left)));remaining_r=set(range(len(right)));matches=[];notes=[]
    ambiguous_l=set();ambiguous_r=set()
    def accept(i,j,method,confidence='strong',score=None):
        remaining_l.remove(i);remaining_r.remove(j)
        r={'clean_index':i,'variant_index':j,'clean_pointer':left[i]['pointer'],'variant_pointer':right[j]['pointer'],'method':method,'confidence':confidence}
        if score is not None:r['score']=round(score,8)
        matches.append(r)
    # Unique shared IDs can only bind when the other shared identity fields agree.
    candidates=collections.defaultdict(set);rc=collections.defaultdict(set)
    for key in IDENTITY_KEYS:
        lm=collections.defaultdict(list);rm=collections.defaultdict(list)
        for i,obj in enumerate(left):
            if key in obj['stable_ids']:lm[_canon(obj['stable_ids'][key])].append(i)
        for j,obj in enumerate(right):
            if key in obj['stable_ids']:rm[_canon(obj['stable_ids'][key])].append(j)
        for value in sorted(lm.keys() & rm.keys()):
            if len(lm[value]) != 1 or len(rm[value]) != 1:
                notes.append({'kind':'duplicate_stable_id','field':key,'clean_indices':lm[value],'variant_indices':rm[value]});continue
            i,j=lm[value][0],rm[value][0]
            if _identity_conflict(left[i],right[j]):
                ambiguous_l.add(i);ambiguous_r.add(j);notes.append({'kind':'conflicting_stable_ids','clean_index':i,'variant_index':j});continue
            candidates[i].add(j);rc[j].add(i)
    for i in sorted(candidates):
        if len(candidates[i])==1:
            j=next(iter(candidates[i]))
            if len(rc[j])==1 and i not in ambiguous_l and j not in ambiguous_r:accept(i,j,'unique_stable_id')
    def unique_fingerprint(signature,method,confidence='strong'):
        lm=collections.defaultdict(list);rm=collections.defaultdict(list)
        for i in sorted(remaining_l-ambiguous_l):
            key=signature(left[i])
            if key is not None:lm[key].append(i)
        for j in sorted(remaining_r-ambiguous_r):
            key=signature(right[j])
            if key is not None:rm[key].append(j)
        for key in sorted(lm.keys()&rm.keys()):
            if len(lm[key])==len(rm[key])==1:
                i,j=lm[key][0],rm[key][0]
                if not _identity_conflict(left[i],right[j]):accept(i,j,method,confidence)
            else:
                notes.append({'kind':'nonunique_'+method,'clean_indices':lm[key],'variant_indices':rm[key]})
    unique_fingerprint(lambda o:_canon(o['raw']), 'exact_object')
    unique_fingerprint(lambda o:_canon({'kind':o['geometry']['kind'],'points':o['geometry']['points']}) if o['geometry'].get('valid') and o['geometry'].get('points') else None, 'unique_exact_geometry')
    if family=='CME_C2_sample':
        keys=('speed_2nd_initial','speed_2nd_final','speed_2nd_20R')
        def sig(o):
            raw=o['raw']
            return _canon([raw[k] for k in keys]) if isinstance(raw,dict) and all(_numeric(raw.get(k)) for k in keys) else None
        unique_fingerprint(sig,'unique_unchanged_speed_triplet','candidate')
    # Mutual best overlap, independent of class. A 0.10 margin rejects close ties.
    scores={(i,j):_iou(left[i],right[j]) for i in remaining_l-ambiguous_l for j in remaining_r-ambiguous_r if not _identity_conflict(left[i],right[j])}
    for i in sorted(remaining_l-ambiguous_l):
        ranked=sorted(((v,j) for (ii,j),v in scores.items() if ii==i and j in remaining_r),reverse=True)
        if not ranked or ranked[0][0]<0.5:continue
        best,j=ranked[0];second=ranked[1][0] if len(ranked)>1 else 0
        reverse=sorted(((v,ii) for (ii,jj),v in scores.items() if jj==j and ii in remaining_l),reverse=True)
        rsecond=reverse[1][0] if len(reverse)>1 else 0
        if best-second>=0.1 and reverse[0][1]==i and best-rsecond>=0.1:
            accept(i,j,'mutual_bbox_iou','candidate',best)
    # Where both sides remain, unmatched records may be moved or transformed,
    # so they are NOT automatically declared one removal plus one addition.
    if remaining_l and remaining_r:
        ambiguous_l.update(remaining_l);ambiguous_r.update(remaining_r)
        notes.append({'kind':'unresolved_correspondence','clean_indices':sorted(remaining_l),'variant_indices':sorted(remaining_r)})
    return sorted(matches,key=lambda r:(r['clean_index'],r['variant_index'])),remaining_l,remaining_r,ambiguous_l,ambiguous_r,notes


def compare_annotations(clean, variant, family):
    """Return identity-aware changes plus lossless raw JSON-path evidence.

    ``changes`` uses established or explicitly heuristic object correspondence;
    ``raw_field_diff`` compares source JSON positions with no identity claim.
    A list reorder can consequently have many raw field changes but only one
    semantic annotation_order_changed item. Coordinates remain whole values.
    """
    left=extract_annotation(clean,family);right=extract_annotation(variant,family)
    result={'version':VERSION,'family':family,'tags':[],'changes':[],'object_matches':[],
            'raw_field_diff':_diff(clean,variant),
            'raw_field_diff_interpretation':'Literal source JSON paths; list positions do not establish object identity. Use changes/object_matches for correspondence-aware conclusions.',
            'counts':{'clean_objects':left['object_count'],'variant_objects':right['object_count']},
            'matching_notes':[],'unmatched_clean_indices':[],'unmatched_variant_indices':[],
            'ambiguous_clean_indices':[],'ambiguous_variant_indices':[],
            'interpretation':'Observed differences relative to clean/P0; not an error/ground-truth verdict. Candidate object matches remain heuristic.'}
    tags=set();path=PATHS.get(family)
    if left['schema_status']!='ok' or right['schema_status']!='ok':
        tags.add('unsupported_structure');result['changes']=copy.deepcopy(result['raw_field_diff'])
        result['matching_notes']=[{'clean_schema_status':left['schema_status'],'variant_schema_status':right['schema_status'],'clean_warnings':left['warnings'],'variant_warnings':right['warnings']}]
        tags.update(x['category'] for x in result['changes']);result['tags']=sorted(tags)
        result['counts'].update(matched=0,unmatched_clean=left['object_count'],unmatched_variant=right['object_count'],ambiguous_clean=left['object_count'],ambiguous_variant=right['object_count'],net_object_count_change=right['object_count']-left['object_count'])
        return result
    matches,ul,ur,al,ar,notes=_match_objects(left['objects'],right['objects'],family)
    result.update(object_matches=matches,matching_notes=notes,unmatched_clean_indices=sorted(ul),unmatched_variant_indices=sorted(ur),ambiguous_clean_indices=sorted(al),ambiguous_variant_indices=sorted(ar))
    result['changes']=_diff(clean,variant,skip=path)
    for m in matches:
        a=left['objects'][m['clean_index']];b=right['objects'][m['variant_index']]
        changes=_diff(a['raw'],b['raw'],path+(a['index'],),path+(b['index'],))
        for c in changes:c.update(object_match_method=m['method'],object_match_confidence=m['confidence'])
        result['changes'].extend(changes)
    if matches and any(m['confidence']=='candidate' for m in matches):tags.add('matching_heuristic')
    if al or ar:
        tags.add('matching_ambiguous')
        before_collection=[left['objects'][i]['raw'] for i in sorted(ul)]
        after_collection=[right['objects'][j]['raw'] for j in sorted(ur)]
        if not _equal(before_collection,after_collection):
            result['changes'].append(_change(before_collection,after_collection,path,path,'unresolved_object_collection',clean_indices=sorted(ul),variant_indices=sorted(ur),note='Object identities unresolved; this collection-level evidence does not assign per-object change.'))
    else:
        for i in sorted(ul):result['changes'].append(_change(left['objects'][i]['raw'],ABSENT,path+(i,),path+(i,),'annotation_removed'))
        for j in sorted(ur):result['changes'].append(_change(ABSENT,right['objects'][j]['raw'],path+(j,),path+(j,),'annotation_added'))
    if len(right['objects'])!=len(left['objects']):
        tag='annotation_added' if len(right['objects'])>len(left['objects']) else 'annotation_removed';tags.add(tag)
        result['changes'].append(_change(len(left['objects']),len(right['objects']),path,path,'annotation_count_changed',note='Net collection cardinality change; does not identify which object is new/absent when matching is ambiguous.'))
    order=[m['variant_index'] for m in matches]
    if order != sorted(order):
        tags.add('annotation_order_changed')
        result['changes'].append(_change([m['clean_index'] for m in matches],order,path,path,'annotation_order_changed'))
    tags.update(x['category'] for x in result['changes'] if x['category'] not in {'annotation_count_changed','unresolved_object_collection'})
    if any(x.get('subcategory')=='science_parameters_changed' for x in result['changes']):
        tags.add('science_parameters_changed')
    if any(n.get('kind') in ('duplicate_stable_id','conflicting_stable_ids') for n in notes):tags.add('identity_warning')
    if not result['changes']:tags.add('no_annotation_change')
    result['tags']=sorted(tags)
    result['counts'].update(matched=len(matches),unmatched_clean=len(ul),unmatched_variant=len(ur),ambiguous_clean=len(al),ambiguous_variant=len(ar),net_object_count_change=len(right['objects'])-len(left['objects']))
    return result
