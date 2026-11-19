from pathlib import Path
source = Path(__file__).parent.parent / 'eight_class_20261104' / 'ground_roadshow.py'
text = source.read_text(encoding='utf-8')
changes = {
    "or not fine.intersection({'L2-G','L3-06','L3-07','L3-MIS'})": "",
    "elif 'L2-G' in codes": "if 'L2-G' in codes",
    "for path in (args.roadshow.parent/'visual_review_v4/responses').glob('*.json'):": "for path in []: # Fresh inference does not replay old API pilot answers",
    "frozen classifier-screened raw-RGB localization; model regions remain candidates": "all readable roadshow RGB images freshly forwarded; frozen classifiers only gate candidate emission; model regions remain candidates",
}
for old, new in changes.items():
    assert old in text, old
    text = text.replace(old, new)
text=text.replace('import numpy as np', "import sys\nsys.path.insert(0, '/staff/tangjiarui/OPC/eight_class_20261104/code')\nimport numpy as np")
Path(__file__).with_name('ground_all_refresh.py').write_text(text, encoding='utf-8')
print('Created independently reviewable all-image grounding script')
