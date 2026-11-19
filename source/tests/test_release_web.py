import unittest,tempfile,json,csv,hashlib,threading,urllib.request,urllib.error
from pathlib import Path
from spaceqa_agent.release_web import create_server
class ReleaseWebTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.e=self.root/'e.json';self.e.write_text('{"measured":1}');(self.root/'complete.json').write_text(json.dumps({'status':'complete','runId':'r'}));(self.root/'scorecard.json').write_text(json.dumps({'runId':'r','dirId':'d','sampleCount':1,'observedTechnicalScore':0,'fullQualityScore':None}));(self.root/'findings.jsonl').write_text(json.dumps({'findingId':'f','checkId':'C31','sampleId':'s','scoreEligible':True,'findingStage':'confirmed','evidenceRef':['e']})+'\n')
  with (self.root/'evidence_index.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=['evidenceId','path','contentHash']);w.writeheader();w.writerow({'evidenceId':'e','path':'e.json','contentHash':hashlib.sha256(self.e.read_bytes()).hexdigest()})
  self.server=create_server('127.0.0.1',0,[self.root]);self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();self.url='http://127.0.0.1:'+str(self.server.server_port);self.id=self.get('/api/runs')['items'][0]['id']
 def get(self,p):return json.load(urllib.request.urlopen(self.url+p))
 def tearDown(self):self.server.shutdown();self.server.server_close();self.thread.join();self.tmp.cleanup()
 def test_routes_and_filter(self):
  for path in ['/','/chat.html','/demo.html','/evaluation.html','/release.js']:self.assertEqual(urllib.request.urlopen(self.url+path).status,200)
  self.assertEqual(self.get('/api/runs/'+self.id+'/findings?stage=candidate')['total'],0);self.assertEqual(self.get('/api/health')['version'],'0.6.1-release3')
 def test_hash_changed_rejected(self):
  self.assertEqual(self.get('/api/runs/'+self.id+'/evidence/e')['measured'],1);self.e.write_text('{}')
  with self.assertRaises(urllib.error.HTTPError) as error:self.get('/api/runs/'+self.id+'/evidence/e')
  self.assertEqual(error.exception.code,409)
 def test_grounded_response(self):
  req=urllib.request.Request(self.url+'/api/chat',data=json.dumps({'runId':self.id,'question':'summary'}).encode(),headers={'Content-Type':'application/json'});r=json.load(urllib.request.urlopen(req));self.assertEqual(r['runId'],'r');self.assertIn('1 条规则确认',r['answer']);self.assertEqual(r['mode'],'deterministic grounded summary')
