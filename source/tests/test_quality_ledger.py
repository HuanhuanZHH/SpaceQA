import copy
import unittest
from spaceqa_agent import quality_ledger as q


class QualityLedgerTests(unittest.TestCase):
    def sample(self, **kw):
        return dict(id="P:a", package="P", family="EPB_seg_sample", data_path="/data/a.png",
                    label_path="/labels/a.json", data_exists=True, image_size=[100,100], **kw)

    def analyze(self, annotation, sample=None):
        return q.analyze_sample(sample or self.sample(), annotation, allowed_labels=["EPB"], contract_source="test schema")

    def test_eight_codes_and_alias_dedup(self):
        self.assertEqual(len(q.CLASSES), 8)
        s=self.sample();s["data_exists"]=False
        result=self.analyze({},s);card=q.aggregate([result])
        self.assertEqual(card["confirmedProblemSamples"],1)
        self.assertEqual(card["defects"]["L0-01"]["locatedInstances"],1)
        self.assertIn("L0-03",result["findings"][0]["aliasCodes"])
        self.assertEqual(card["observedTechnicalScore"],0)

    def test_no_annotations_not_spurious_or_missing_proof(self):
        result=self.analyze({"science_annotation":{"mask":{"shapes":[]}}})
        self.assertEqual(result["checks"]["L3-06"]["status"],"undetermined")
        self.assertEqual(result["checks"]["L3-07"]["status"],"undetermined")
        self.assertEqual(result["findings"],[])
        self.assertIsNone(q.aggregate([result])["fullQualityScore"])

    def test_dimension_severity_and_duplicate_scope(self):
        result=self.analyze({"basic_annotation":{"imgSize":"100x50 pixels"},"science_annotation":{"mask":{"imageWidth":100,"imageHeight":50}}})
        self.assertEqual(len(result["findings"]),2)
        self.assertAlmostEqual(result["findings"][0]["severity"]["value"],.5)
        card=q.aggregate([result])
        self.assertEqual(card["defects"]["L2-F"]["confirmedSamples"],1)
        self.assertEqual(len(card["severity"]),1)

    def test_class_location_is_real_and_severity_independent_of_count(self):
        result=self.analyze({"science_annotation":{"mask":{"shapes":[
            {"label":"__OOV__","points":[[10,10],[20,10],[20,20]]},
            {"label":"EPB","points":[[40,40],[50,40],[50,50]]}]}}})
        f=result["findings"][0]
        self.assertEqual(f["locator"]["annotation_pointer"],"/science_annotation/mask/shapes/0/label")
        self.assertEqual(f["locator"]["bbox"],[10,10,20,20])
        self.assertEqual(f["severity"]["value"],.5)
        self.assertEqual(result["checks"]["L2-P"]["status"],"fail")

    def test_in_vocabulary_is_not_semantic_pass(self):
        result=self.analyze({"science_annotation":{"mask":{"shapes":[{"label":"EPB"}]}}})
        self.assertEqual(result["checks"]["L2-P"]["scopeStatus"],"pass")
        self.assertEqual(result["checks"]["L2-P"]["status"],"undetermined")
        card=q.aggregate([result])
        self.assertIsNone(card["defects"]["L2-P"]["proportionEstimate"])
        self.assertEqual(card["defects"]["L2-P"]["proportionBounds"],[0.,1.])

    def test_object_local_dimensions_are_not_image_dimensions(self):
        result=self.analyze({"science_annotation":{"mask":{"shapes":[{"label":"EPB","imageWidth":5,"imageHeight":5}]}}})
        self.assertFalse(result["findings"])
        self.assertEqual(result["checks"]["L2-F"]["status"],"not_applicable")

    def test_candidate_never_scores_or_resolves_unknowns(self):
        result=self.analyze({})
        q.attach_candidates(result,[dict(defect_class="L3-06",locator={"bbox":[1,2,3,4]},scoreEligible=True,confidence=.99)])
        card=q.aggregate([result])
        self.assertEqual(card["confirmedProblemSamples"],0)
        self.assertEqual(card["defects"]["L3-06"]["candidateSamples"],1)
        self.assertEqual(card["defects"]["L3-06"]["statusCounts"]["undetermined"],1)

    def test_rejects_unlocated_candidate_and_duplicate_samples(self):
        result=self.analyze({})
        with self.assertRaises(ValueError):q.attach_candidates(result,[dict(defect_class="L3-07",locator={"path":"x"})])
        with self.assertRaises(ValueError):q.aggregate([result,result])

    def test_ties_are_explicit_and_scores_monotonic_in_density(self):
        clean=self.analyze({});bad=self.analyze({"basic_annotation":{"imgSize":"50x100 pixels"}})
        clean2=copy.deepcopy(clean);clean2.update(sampleId="P:b")
        self.assertGreater(q.aggregate([clean,clean2])["observedTechnicalScore"],q.aggregate([bad,clean2])["observedTechnicalScore"])
        a=q.aggregate([clean]);b=copy.deepcopy(a);b["dirId"]="Q"
        ranked=q.rank([a,b])
        self.assertEqual([c["technicalCompetitionRank"] for c in ranked],[1,1])
        self.assertEqual(len(ranked[0]["tiedDirectoryIds"]),1)

    def test_time_order_diagnostic_does_not_fake_jitter(self):
        result=self.analyze({"ulf_event_annotation":{"matched_events":[{"kind":"Pi2","start_dt":"2020-01-01T01:00:00","end_dt":"2020-01-01T00:00:00"}]}})
        self.assertEqual(result["diagnostics"][0]["seconds"],3600)
        self.assertEqual(result["checks"]["L2-T"]["status"],"undetermined")

    def test_instance_localization_adapter_computes_coverage_without_promotion(self):
        result=self.analyze({})
        def obj(oid,box):
            return dict(id=oid,sample_id="P:a",label="EPB",confidence=.99,
                locator=dict(type="bbox",bbox=box,frame_id="raw-sha",coordinate_system="image_pixels",image_size=[100,100]))
        payload=dict(sample_id="P:a",data_available=True,annotation_parse_status="ok",
            reference=dict(status="ok",independent_of_annotations=True,source="model",objects=[obj("r",[10,10,30,30])]),
            annotations=[obj("a",[10,10,20,30])])
        q.compare_instances(result,payload)
        self.assertEqual(result["findings"][0]["defectType"],"L3-MIS")
        self.assertAlmostEqual(result["findings"][0]["severity"]["value"],.5)
        self.assertFalse(result["findings"][0]["scoreEligible"])
        self.assertEqual(q.aggregate([result])["confirmedProblemSamples"],0)

    def test_eight_protocol_and_missing_file_alias(self):
        from spaceqa_agent import refinement as r
        from spaceqa_agent.review_eight import schema_eight,expand_answer,project_review
        self.assertEqual(len(schema_eight()['properties']['checks']['required']),8)
        case=dict(case_id="CASE-alias",family="AIA_AR_sample",sample_path="data/a.png",label_path="label/a.json",
                  annotation={},data_facts={"exists":False})
        a=dict(case_id="CASE-alias",checks={c:dict(status="undetermined",reason="No reference") for c in q.CLASSES},defects=[],limitations=[])
        projected=project_review(r.reconcile_mechanical(r.validate_review(expand_answer(a),case),case))
        self.assertEqual(set(projected['final_checks']),set(q.CLASSES))
        self.assertEqual(projected['final_checks']['L0-01']['status'],'fail')
        self.assertEqual(projected['program_findings'][0]['defect_class'],'L0-01')
        agg=r.aggregate_reviews([projected],2,classes=q.CLASSES)
        self.assertEqual(agg['applicable_check_denominator'],16)

    def test_eight_mcp_schema_and_validation(self):
        import tempfile,json
        from pathlib import Path
        from spaceqa_agent.review_mcp import ReviewTools,dispatch
        with tempfile.TemporaryDirectory() as folder:
            case=dict(case_id='CASE-eight',family='ULF_sample',annotation={},sample_path='data/a.csv',label_path='label/a.json',data_facts={'exists':True})
            Path(folder,'CASE-eight.json').write_text(json.dumps(case),encoding='utf-8')
            service=ReviewTools(Path(folder),eight_classes=True)
            tools=dispatch(service,dict(id=1,method='tools/list'))['result']['tools']
            schema=next(x['inputSchema']['properties']['answer'] for x in tools if x['name']=='validate_review')
            self.assertEqual(len(schema['properties']['checks']['required']),8)
            answer=dict(case_id='CASE-eight',checks={c:dict(status='undetermined',reason='No reference') for c in q.CLASSES},defects=[],limitations=[])
            out=service.invoke('validate_review',dict(case_id='CASE-eight',answer=answer))
            self.assertEqual(len(out['final_checks']),8)


if __name__=="__main__":unittest.main()
