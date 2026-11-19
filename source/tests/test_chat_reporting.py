import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from spaceqa_agent.chat_reporting import enrich_insights, render_report

class ReportScopeTests(unittest.TestCase):
    def test_executed_classifier_does_not_clear_unknown_verdict(self):
        original={'sixDimensions':{'annotationQuality':dict(id='annotationQuality',score=None,state='undetermined',checkCoverage=0)},
                  'axes':[dict(code='L3-07',name='虚标',displayScore=100,coverage={'undeterminedSamples':10})]}
        value=enrich_insights(copy.deepcopy(original))
        self.assertIsNone(value['sixDimensions']['annotation_quality']['score'])
        self.assertEqual(value['axes'][0]['coverage']['undeterminedSamples'],10)
        self.assertIn('模型未检出',value['coverageGaps'][0]['reason'])

    def test_material_dimensions_get_explanations_without_invented_score(self):
        value=enrich_insights({'axes':[],'sixDimensions':{'traceability':dict(id='traceability',name='�',score=None)}})
        row=value['sixDimensions']['documentation_provenance']
        self.assertEqual(row['name'],'文档与来源追溯')
        self.assertIsNone(row['score'])
        self.assertIn('来源链',row['reason'])

    def test_report_escapes_uploaded_text_and_keeps_all_evidence_download(self):
        session={'id':'chat-safe','analysis':{'result':{}},'liveRun':{}}
        insights=enrich_insights({'datasetName':'<script>alert(1)</script>','summary':{'totalSamples':1},
                                 'axes':[dict(code='L0-01',name='文件缺失',confirmedSamples=1,candidateOnlySamples=0)],'sixDimensions':{}})
        content=render_report(session,insights,[dict(defectType='L0-01',stage='confirmed',sampleId='s',basis='<img src=x onerror=alert(1)>',locator={'path':'/a/b.json'})])
        self.assertNotIn('<script>',content)
        self.assertNotIn('<img src=x',content)
        self.assertIn('&lt;script&gt;',content)
        self.assertIn('/evidence.jsonl',content)
        self.assertIn('规则确认',content)

if __name__=='__main__':unittest.main()
