#!/usr/bin/env python3
"""Measure local VLM latency on real image inputs; this is not a quality eval."""
import argparse
import base64
import concurrent.futures
import copy
import hashlib
import io
import json
from pathlib import Path
import statistics
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

FAMILIES = ['AIA_AR_sample', 'HMI_AR_sample', 'CME_C2_sample', 'EPB_seg_sample', 'DMSP_SSUSI']
EVIDENCE_CODES = ['insufficient_context', 'no_supported_defect', 'boundary_mismatch',
                  'label_mismatch', 'unlabeled_candidate', 'unsupported_label', 'coverage_mismatch']
FINITE_REGEX = (r'\{"decision":"(confirm|reject|uncertain)",'
                r'"defect_class_id":"(NONE|L2-G|L2-P|L3-06|L3-07|L3-MIS)",'
                r'"confidence":(0\.0|0\.25|0\.5|0\.75|1\.0),'
                r'"evidence":"(' + '|'.join(EVIDENCE_CODES) + r')",'
                r'"needs_human_review":(true|false)\}')
SCHEMA = {
    'type': 'object',
    'properties': {
        'decision': {'type': 'string', 'enum': ['confirm', 'reject', 'uncertain']},
        'defect_class_id': {'type': 'string', 'enum': ['NONE', 'L2-G', 'L2-P', 'L3-06', 'L3-07', 'L3-MIS']},
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'evidence': {'type': 'string'},
        'needs_human_review': {'type': 'boolean'},
    },
    'required': ['decision', 'defect_class_id', 'confidence', 'evidence', 'needs_human_review'],
    'additionalProperties': False,
}


def jpeg_input(path, edge):
    with Image.open(path) as original:
        original_size = list(original.size)
        image = original.convert('RGB')
        image.thumbnail((edge, edge), Image.Resampling.LANCZOS)
        stream = io.BytesIO()
        image.save(stream, format='JPEG', quality=90)
        payload = stream.getvalue()
        return {'source': str(path), 'original_size': original_size, 'sent_size': list(image.size),
                'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload),
                'url': 'data:image/jpeg;base64,' + base64.b64encode(payload).decode('ascii')}


def schema_valid(value):
    return (isinstance(value, dict) and set(value) == set(SCHEMA['required'])
            and value['decision'] in SCHEMA['properties']['decision']['enum']
            and value['defect_class_id'] in SCHEMA['properties']['defect_class_id']['enum']
            and isinstance(value['confidence'], (float, int)) and not isinstance(value['confidence'], bool)
            and 0 <= value['confidence'] <= 1 and isinstance(value['evidence'], str)
            and isinstance(value['needs_human_review'], bool))


def invoke(case, base_url, api_key):
    compact = case.get('compact_output', False)
    finite = case.get('finite_output', False)
    schema = copy.deepcopy(SCHEMA)
    if compact:
        schema['properties']['evidence']['maxLength'] = 96
    content = [{'type': 'text', 'text': (
        'OPC annotation review runtime pilot, case ' + case['case_id'] + '. '
        'The final image is the current annotation preview; an earlier image, if present, is the raw image. '
        'Do not treat differences from an absent reference as proof. If scientific evidence is insufficient, '
        'return uncertain. ' + (
            'Use one evidence code: ' + ', '.join(EVIDENCE_CODES) + '. No free text. ' if finite else
            'Give brief English evidence within ' + ('12' if compact else '30') + ' words. ')
        + 'Return only the requested JSON.')}]
    content.extend({'type': 'image_url', 'image_url': {'url': item['url']}} for item in case['images'])
    body = {'model': 'Qwen3-VL-8B-Instruct', 'messages': [{'role': 'user', 'content': content}],
            'temperature': 0, 'max_tokens': 128 if finite else (256 if compact else 160),
            'response_format': {'type': 'json_schema', 'json_schema': {
                'name': 'opc_runtime_review', 'strict': True, 'schema': schema}}}
    if finite:
        body.pop('response_format')
        body['structured_outputs'] = {'regex': FINITE_REGEX}
    req = Request(base_url.rstrip('/') + '/chat/completions', data=json.dumps(body).encode(),
                  headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json'})
    started = time.monotonic()
    result = {k: v for k, v in case.items() if k != 'images'}
    result['images'] = [{k: v for k, v in item.items() if k != 'url'} for item in case['images']]
    try:
        with urlopen(req, timeout=180) as response:
            payload = json.load(response)
        choice = payload['choices'][0]
        try:
            parsed = json.loads(choice['message']['content'])
            result.update(valid_json=True, schema_valid=schema_valid(parsed), response=parsed)
        except (ValueError, TypeError):
            result.update(valid_json=False, schema_valid=False,
                          incomplete_content=choice['message'].get('content', '')[:1200])
        result.update(http_success=True, usage=payload.get('usage', {}), finish_reason=choice.get('finish_reason'))
    except HTTPError as exc:
        result.update(http_success=False, http_status=exc.code, schema_valid=False)
    except Exception as exc:
        result.update(http_success=False, error_type=type(exc).__name__, schema_valid=False)
    result['elapsed_seconds'] = round(time.monotonic() - started, 4)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase2', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8001/v1')
    parser.add_argument('--api-key-file', type=Path, default=Path('/staff/tangjiarui/OPC/.qwen3vl-api-key'))
    parser.add_argument('--compact-output', action='store_true', help='Bound evidence length and allow 256 output tokens')
    parser.add_argument('--finite-output', action='store_true', help='Use bounded JSON grammar and enumerated evidence codes')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'output_protocol.json').write_text(json.dumps({
        'compact_output': args.compact_output, 'finite_output': args.finite_output,
        'base_schema': SCHEMA, 'finite_regex': FINITE_REGEX if args.finite_output else None}, indent=2), encoding='utf-8')
    api_key = args.api_key_file.read_text().strip()
    started = time.monotonic()
    samples = []
    for family in FAMILIES:
        selected = 0
        for path in sorted((args.phase2 / 'groups' / family).glob('*.json')):
            group = json.loads(path.read_text())
            tier = next(t for t in group['tiers'] if t['tier'] == 'P1')
            if not tier['data_exists'] or tier['render_status'] != 'ok':
                continue
            raw, preview = Path(tier['data_path']), Path(tier['output_path'])
            samples.append({'family': family, 'base_sample_id': group['base_sample_id'],
                            'single_512': [jpeg_input(preview, 512)],
                            'dual_512': [jpeg_input(raw, 512), jpeg_input(preview, 512)],
                            'dual_768': [jpeg_input(raw, 768), jpeg_input(preview, 768)]})
            selected += 1
            if selected == 2:
                break
        if selected != 2:
            raise RuntimeError('Insufficient valid images for ' + family)
    prepared_seconds = round(time.monotonic() - started, 3)
    warmup_case = {'case_id': 'warmup', 'family': samples[0]['family'], 'images': samples[0]['single_512'],
                   'compact_output': args.compact_output, 'finite_output': args.finite_output}
    warmup = invoke(warmup_case, args.base_url, api_key)
    (args.output / 'warmup.json').write_text(json.dumps(warmup, indent=2), encoding='utf-8')
    if not warmup['http_success']:
        raise RuntimeError('Warmup failed; inspect the sanitized warmup record')
    groups, records = {}, []
    for mode, workers in [('single_512', 1), ('dual_512', 1), ('dual_768', 2)]:
        cases = [{'case_id': mode + ':' + sample['base_sample_id'], 'family': sample['family'],
                  'images': sample[mode], 'compact_output': args.compact_output} for sample in samples]
        for case in cases:
            case['finite_output'] = args.finite_output
        began = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(lambda c: invoke(c, args.base_url, api_key), cases))
        wall = time.monotonic() - began
        times = sorted(row['elapsed_seconds'] for row in rows)
        groups[mode] = {'requests': len(rows), 'concurrency': workers, 'wall_seconds': round(wall, 4),
                        'successful_http': sum(row['http_success'] for row in rows),
                        'valid_structured_outputs': sum(row['schema_valid'] for row in rows),
                        'median_latency_seconds': statistics.median(times),
                        'p95_latency_seconds': times[max(0, int(len(times) * .95 + .999) - 1)],
                        'amortized_seconds_per_request': round(wall / len(rows), 4),
                        'prompt_tokens': sum(row.get('usage', {}).get('prompt_tokens', 0) for row in rows),
                        'completion_tokens': sum(row.get('usage', {}).get('completion_tokens', 0) for row in rows)}
        records.extend(rows)
        (args.output / 'requests.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
        print(mode, json.dumps(groups[mode]), flush=True)
    result = {'model': 'Qwen3-VL-8B-Instruct', 'hardware': 'existing A100 service; runtime bitsandbytes 4bit',
              'prepared_seconds': prepared_seconds,
              'max_output_tokens': 128 if args.finite_output else (256 if args.compact_output else 160),
              'compact_output': args.compact_output,
              'finite_output': args.finite_output,
              'warmup_seconds': warmup['elapsed_seconds'], 'groups': groups,
              'total_elapsed_seconds': round(time.monotonic() - started, 3),
              'limitations': ['Latency pilot only; no scientific accuracy or end-to-end contest score is measured.',
                              'Thirty requests across five families; loaded service, bounded 512/768 pixel inputs.',
                              'A10/DCU performance, model cold loading and official API quotas are not measured.',
                              'Reported throughput excludes data preparation, full rules, scoring and report writing.']}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
