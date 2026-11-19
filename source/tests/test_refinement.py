import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from spaceqa_agent import refinement as r
from spaceqa_agent.review_mcp import ReviewTools,dispatch


def fixture():
    case=dict(case_id="CASE-test",family="EPB_seg_sample",sample_path="data/a.png",label_path="label/a.json",
        annotation={"imgSize":[100,50]},data_facts={"exists":True},image_size=[100,100],
        frame_id="abc",fusion_probabilities=[.8,.6,.9],target=[1,1,1],tier="P4")
    answer=dict(case_id=case["case_id"],checks={c:{"status":"undetermined","reason":"No independent definition"} for c in r.CLASSES},
        defects=[],limitations=[])
    answer["checks"]["L2-F"]={"status":"fail","reason":"decoded 100x100 vs field 100x50"}
    answer["defects"]=[dict(defect_class="L2-F",locator=dict(type="record",annotation_pointer="/imgSize",bbox=None,time_interval=None),
        confidence=.99,severity=dict(metric="height_difference",value=50,unit="pixel"),explanation="Image height mismatch")]
    return case,answer


class ContractTest(unittest.TestCase):
    def test_eight_class_probability_is_validated_and_separate_from_coarse(self):
        from spaceqa_agent.quality_ledger import CLASSES
        c,a=fixture();c['fine_probabilities']={code:.2 for code in CLASSES}
        c['fine_probabilities']['L2-F']=.73
        result=r.validate_review(a,c);d=result['defects'][0]
        self.assertEqual(d['fusion_group_probability'],.9)
        self.assertEqual(d['fusion_class_probability'],.73)
        self.assertEqual(d['weighted_burden'],.73)
        self.assertFalse(d['score_eligible'])
        for mutation in ['missing','nan','range']:
            bad=copy.deepcopy(c)
            if mutation=='missing':bad['fine_probabilities'].pop('L2-T')
            if mutation=='nan':bad['fine_probabilities']['L2-T']=float('nan')
            if mutation=='range':bad['fine_probabilities']['L2-T']=1.1
            with self.assertRaises(ValueError):r.validate_case(bad)

    def test_scope_and_no_truth(self):
        c,a=fixture();v=r.validate_case(c)
        self.assertNotIn("target",v);self.assertNotIn("tier",v)
        row=r.validate_review(a,c);d=row["defects"][0]
        self.assertFalse(d["score_eligible"]);self.assertEqual(d["weighted_burden"],.9)
        self.assertEqual(d["locator"]["record_id"],"/imgSize")

    def test_real_pointer_and_frame_required(self):
        for mutation in ["pointer","bbox","state","severity","confidence"]:
            with self.subTest(mutation=mutation):
                c,a=fixture();d=a["defects"][0]
                if mutation=="pointer":d["locator"]["annotation_pointer"]="/fake"
                if mutation=="bbox":d["locator"].update(type="bbox",bbox=[0,0,100,101])
                if mutation=="state":a["checks"]["L2-F"]["status"]="pass"
                if mutation=="severity":d["severity"]["metric"]="confidence"
                if mutation=="confidence":d["confidence"]=float("nan")
                with self.assertRaises(ValueError):r.validate_review(a,c)

    def test_missing_file_not_spurious(self):
        c,a=fixture();c["data_facts"]["exists"]=False
        a["checks"]["L2-F"]["status"]="undetermined";a["checks"]["L3-07"]["status"]="fail"
        d=a["defects"][0];d["defect_class"]="L3-07"
        d["locator"].update(type="bbox",bbox=[0,0,10,10])
        with self.assertRaises(ValueError):r.validate_review(a,c)

    def test_dedup_denominator_and_incomplete(self):
        c,a=fixture();a["defects"]*=2
        for code,ch in a['checks'].items():
            if code!='L2-F':ch.update(status='pass',reason='Synthetic fully checked scope')
        v=r.validate_review(a,c);self.assertEqual(v["defect_count"],1)
        full=r.aggregate_reviews([v],1);self.assertEqual(full["weighted_burden_per_sample"],.9)
        partial=r.aggregate_reviews([v],2)
        self.assertIsNone(partial["weighted_burden_per_sample"])
        self.assertEqual(partial["unknown_samples"],1)
        # Nine checks remain applicable even when they cannot be computed.
        self.assertEqual(partial["applicable_check_denominator"],18)
        self.assertEqual(partial["completed_checks"],9)
        with self.assertRaises(ValueError):r.aggregate_reviews([v,v],2)

    def test_empty_fail_rejected_and_probability_not_severity(self):
        c,a=fixture();a["defects"]=[]
        with self.assertRaises(ValueError):r.validate_review(a,c)

    def test_burden_reproducible_under_record_reordering(self):
        c,a=fixture();rows=[]
        for i,p in enumerate([1e-16,1.,.1,1e-16,.9,.002]):
            cc=copy.deepcopy(c);aa=copy.deepcopy(a)
            cc['case_id']=aa['case_id']='CASE-order-'+str(i);cc['fusion_probabilities'][2]=p
            rows.append(r.validate_review(aa,cc))
        self.assertEqual(r.aggregate_reviews(rows,len(rows)),r.aggregate_reviews(list(reversed(rows)),len(rows)))

    def test_independent_dimension_severity_recomputed(self):
        c,a=fixture();a['defects'][0]['severity']['value']=999
        v=r.attach_measurements(r.validate_review(a,c),c)
        d=v['defects'][0]
        self.assertEqual(d['severity']['value'],50)
        self.assertEqual(d['model_severity_estimate']['value'],999)
        self.assertEqual(d['severity_source'],'program_recomputed_from_decoded_dimensions')

    def test_mechanical_ledger_recovers_missed_fact_blocks_hallucination(self):
        c,a=fixture();a['checks']['L2-F']['status']='undetermined';a['defects']=[]
        result=r.reconcile_mechanical(r.validate_review(a,c),c)
        self.assertEqual(result['combined_defect_count'],1)
        self.assertEqual(result['program_findings'][0]['severity']['value'],50)
        self.assertEqual(result['final_checks']['L2-F']['status'],'fail')
        c,a=fixture();c['annotation']['imgSize']=[100,100]
        result=r.reconcile_mechanical(r.validate_review(a,c),c)
        self.assertEqual(result['combined_defect_count'],0)
        self.assertEqual(len(result['blocked_mechanical_claims']),1)
        c['data_facts']['exists']=False
        result=r.reconcile_mechanical(r.abstention(c,'API failed'),c)
        self.assertEqual([x['defect_class'] for x in result['program_findings']],['L0-03'])
        self.assertEqual(result['final_checks']['L0-01']['status'],'undetermined')

    def test_api_repair_then_success_and_bounded_failure(self):
        from io import BytesIO
        c,a=fixture();bad=copy.deepcopy(a);bad['defects'][0]['locator']['annotation_pointer']='/imaginary'
        def reply(answer):
            return BytesIO(json.dumps({'choices':[{'message':{'content':json.dumps(answer)}}],
                'usage':{'total_tokens':100}}).encode())
        class Opener:
            def __init__(self):self.n=0
            def open(self,*args,**kwargs):
                self.n+=1;return reply(bad if self.n==1 else a)
        opener=Opener()
        with patch('urllib.request.build_opener',return_value=opener):
            out=r.review_case(c,base_url='http://127.0.0.1:1/v1',api_key='private',model='fixture',retry_limit=1)
        self.assertEqual(out['status'],'reviewed');self.assertEqual(len(out['attempts']),2)
        self.assertEqual(out['usage']['total_tokens'],200)
        self.assertNotIn('private',json.dumps(out))
        class BadOpener:
            def open(self,*args,**kwargs):return reply(bad)
        with patch('urllib.request.build_opener',return_value=BadOpener()):
            out=r.review_case(c,base_url='http://127.0.0.1:1/v1',api_key='',model='fixture',retry_limit=2)
        self.assertEqual(len(out['attempts']),3);self.assertEqual(out['status'],'unavailable')
        self.assertTrue(all(x['status']=='undetermined' for x in out['checks'].values()))

    def test_mcp_protocol_and_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);c,a=fixture();(root/"CASE-test.json").write_text(json.dumps(c),encoding="utf-8")
            service=ReviewTools(root)
            response=dispatch(service,dict(id=1,method="tools/list"))
            self.assertEqual(len(response["result"]["tools"]),7)
            with self.assertRaises(ValueError):service.case("../a")
            service.invoke("validate_review",{"case_id":"CASE-test","answer":a})
            self.assertFalse(service.invoke("aggregate_reviews",{"case_ids":["CASE-test"],"total_samples":1})["complete"])
            messages='\n'.join(json.dumps(x) for x in [dict(jsonrpc="2.0",id=1,method="initialize",params={"protocolVersion":"2025-03-26"}),dict(jsonrpc="2.0",method="notifications/initialized"),dict(jsonrpc="2.0",id=2,method="tools/call",params={"name":"get_case","arguments":{"case_id":"CASE-test"}})])+'\n'
            proc=subprocess.run([sys.executable,"-m","spaceqa_agent.review_mcp","--case-root",str(root)],input=messages,text=True,capture_output=True,check=True)
            lines=[json.loads(x) for x in proc.stdout.splitlines()]
            self.assertEqual(len(lines),2);self.assertEqual(lines[0]["result"]["protocolVersion"],"2025-03-26")
            self.assertFalse(lines[1]["result"]["isError"])

    def test_prepare_current_image_and_irregular_csv_without_truth(self):
        from PIL import Image
        from spaceqa_agent.review_inputs import prepare_case
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);src=root/'source';src.mkdir()
            label=src/'a.json';label.write_text(json.dumps({'imgSize':[100,50]}),encoding='utf-8')
            image=src/'a.png';Image.new('RGB',(100,100),'gray').save(image)
            before=image.read_bytes(),label.read_bytes()
            signals={'fusion_probabilities':[.1,.2,.8],'target':[1,1,1]}
            output=prepare_case(data_path=image,label_path=label,family='EPB_seg_sample',output=root/'case',signals=signals)
            case=json.loads(Path(output['case']).read_text())
            self.assertEqual(case['image_size'],[100,100]);self.assertEqual(len(output['images']),2)
            self.assertNotIn('target',case);self.assertEqual(case['rule_evidence'][0]['severity']['value'],50)
            self.assertEqual(before,(image.read_bytes(),label.read_bytes()))
            data=src/'a.csv';data.write_text('t,H,D,Z\n0,1,2,3\n2,3,4,5\n7,2,3,5\n',encoding='utf-8')
            output=prepare_case(data_path=data,label_path=label,family='ULF_sample',output=root/'csvcase')
            case=json.loads(Path(output['case']).read_text())
            self.assertEqual(case['time_axis']['end'],7)
            self.assertEqual(case['data_facts']['csv_rows'],3)


if __name__=="__main__":unittest.main()
