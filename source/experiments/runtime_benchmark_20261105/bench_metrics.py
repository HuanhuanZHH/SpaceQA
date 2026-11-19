"""Low overhead actual stage timers for an isolated roadshow benchmark."""
import json
import os
import time
from pathlib import Path

_root = Path(os.environ['SPACEQA_BENCH_ROOT'])
_root.joinpath('timing').mkdir(parents=True, exist_ok=True)
_phase = None
_last = None
_package = None
_package_start = None
_shared_start = None
_forward_start = None
_forward_seconds = 0.0
_forward_batches = 0


def event(kind, **data):
    row = dict(event=kind, phase=_phase, perf_counter=time.perf_counter(), pid=os.getpid(), **data)
    with (_root / 'timing' / (str(os.getpid()) + '.jsonl')).open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def script_enter(phase):
    global _phase, _last
    _phase = phase
    _last = time.perf_counter()
    event('script_enter')


def mark(stage, **data):
    global _last
    now = time.perf_counter()
    event('stage', package=_package, stage=stage, seconds=now - _last, **data)
    _last = time.perf_counter()


def package_begin(package):
    global _package, _package_start, _last, _forward_seconds, _forward_batches
    _package = package
    _package_start = _last = time.perf_counter()
    _forward_seconds = 0.0
    _forward_batches = 0
    event('package_begin', package=package)


def shared_begin(stage):
    global _shared_start
    _shared_start = time.perf_counter()
    event('shared_begin', package=_package, stage=stage)


def shared_end(stage, **data):
    global _last
    now = time.perf_counter()
    event('shared', package=_package, stage=stage, seconds=now - _shared_start, **data)
    _last = time.perf_counter()


def forward_begin(torch):
    global _forward_start
    torch.cuda.synchronize()
    _forward_start = time.perf_counter()


def forward_end(torch):
    global _forward_seconds, _forward_batches
    torch.cuda.synchronize()
    _forward_seconds += time.perf_counter() - _forward_start
    _forward_batches += 1


def package_end(**data):
    global _package, _last
    now = time.perf_counter()
    event('package_end', package=_package, seconds=now - _package_start,
          forward_seconds=_forward_seconds, forward_batches=_forward_batches, **data)
    _package = None
    _last = time.perf_counter()


def done():
    event('script_complete')
