"""Read-only experiment progress; contains no credentials or user data bytes."""
import datetime as dt,json,statistics
from pathlib import Path

ROOT=Path('/staff/tangjiarui/OPC/eight_class_20261104')


def read(path,default=None):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def snapshot():
    state=read(ROOT/'localizer_v2/state.json',{})
    result=dict(updated=dt.datetime.now().astimezone().isoformat(),classification=read(ROOT/'pipeline_v2_state.json',{}),
        localization=state,completion=read(ROOT/'completion_state.json',{}),failures={})
    for p in [ROOT/'localizer_v2/failure.json',ROOT/'completion_failure.json',ROOT/'pipeline_v2_state_failure.json']:
        if p.is_file():result['failures'][p.name]=read(p)
    if state.get('stage') in {'training','epoch_finished'}:
        route=state['route'];history=read(ROOT/'localizer_v2'/(route+'_training.json'),[])
        times=[history[i]['seconds']-(history[i-1]['seconds'] if i else 0) for i in range(len(history))]
        if times:
            epoch_seconds=statistics.median(times[-3:]);progress=state['epoch']-1
            if state['stage']=='training':progress+=state['step']/max(1,state['steps'])
            else:progress=state['epoch']
            remaining=(12-progress+(12 if route=='presence_gated' else 0))*epoch_seconds
            result['training_eta']=dict(epoch_seconds=epoch_seconds,remaining_training_seconds=remaining,
                expected_finish=(dt.datetime.now().astimezone()+dt.timedelta(seconds=remaining)).isoformat(),
                excludes='held-out evaluation, legacy comparison, roadshow grounding and report generation')
    review=read(ROOT/'visual_review_v4/receipt.json')
    if review:result['visual_review']=review
    return result


if __name__=='__main__':print(json.dumps(snapshot(),ensure_ascii=False,indent=2))
