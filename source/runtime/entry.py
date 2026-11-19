import sys,argparse,json
from pathlib import Path
def main():
 args=sys.argv[1:]
 if args and args[0]=='serve-release':
  from spaceqa_agent.release_web import main as serve_release
  return serve_release(args[1:])
 if args and args[0]=='audit' and len(args)==2 and (Path(args[1])/'predictions.jsonl').is_file():
  from audit_evaluation import audit
  result=audit(args[1]);print(json.dumps(result,ensure_ascii=False,indent=2))
  if result['status']!='pass':raise SystemExit(1)
  return
 if args and args[0]=='fresh':
  import compact_infer
  sys.argv=[sys.argv[0]]+args[1:];return compact_infer.main()
 if args and args[0]=='evaluate':
  import pipeline
  try:return pipeline.main(args[1:])
  except Exception as exc:
   import datetime,time,resource
   if '--output' in args:
    out=Path(args[args.index('--output')+1]);log=out/'run.log.jsonl'
    if out.is_dir():
     first=json.loads(log.open().readline()) if log.is_file() else {}
     row=dict(first,event='evaluate',status='failed',timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason=type(exc).__name__+': '+str(exc),retryCount=0,resourcePeak={'processMaxRssKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
     with log.open('a') as fp:fp.write(json.dumps(row)+'\n')
   raise
 if args and args[0]=='manifest':
  print(Path('/opt/spaceqa/package_manifest.json').read_text());return
 if not args or args[0] in ['--help','-h']:
  print('SpaceQA Agent 0.6.1 all-21-expert offline edition\n\nCommands:\n evaluate --input /input/DATASET --family FAMILY --output /output/RUN\n fresh --records /validation/records.jsonl --output /output/experts\n manifest\n scan/audit/serve: standard deterministic agent commands\n\nFull expert evaluation requires a CUDA GPU. All model weights and runtime dependencies are bundled.');return
 from spaceqa_agent.cli import main as cli
 sys.argv=[sys.argv[0]]+args;return cli()
if __name__=='__main__':main()
