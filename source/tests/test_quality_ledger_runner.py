"""Exercise new-package entry, source preservation and independent replay."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class NewPackageRunnerTest(unittest.TestCase):
    def test_new_package_four_states_replay_and_source_immutability(self):
        try:from PIL import Image
        except ImportError:self.skipTest('Pillow required for generated image fixture')
        repo=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);package=root/'EPB_seg_sample_abcdefghij';package.mkdir()
            for i in range(4):
                if i!=3:Image.new('RGB',(100,100),'black').save(package/(str(i)+'.png'))
                annotation=dict(label_header=dict(labelVersion='fixture'),basic_annotation=dict(fileName=str(i)+'.png',imgSize='100x50 pixels' if i==2 else '100x100 pixels'),science_annotation=dict(mask=dict(shapes=[dict(label='__OOV__' if i==1 else 'EPB',points=[[10,10],[30,10],[30,30]])])))
                (package/(str(i)+'.json')).write_text(json.dumps(annotation),encoding='utf-8')
            before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.iterdir()}
            for name in ('first','second'):
                result=subprocess.run([sys.executable,str(repo/'experiments/rescue_20261104/run_roadshow.py'),'--input',str(package),'--output',str(root/name)],env=os.environ.copy(),capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                card=json.loads((root/name/'scorecards.json').read_text(encoding='utf-8'))[0]
                self.assertEqual(card['sampleCount'],4);self.assertEqual(card['confirmedProblemSamples'],3)
                self.assertEqual(card['observedTechnicalScore'],25.)
                self.assertIsNone(card['fullQualityScore'])
            p=Path('packages')/package.name/'samples.jsonl'
            self.assertEqual((root/'first'/p).read_bytes(),(root/'second'/p).read_bytes())
            self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.iterdir()})
