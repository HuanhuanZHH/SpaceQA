"""Small process measurements without optional monitoring dependencies."""
from __future__ import annotations

import os
import platform
import time
from typing import Any


def process_usage() -> dict[str, Any]:
    """Return measured process CPU and lifetime peak RSS, never a GPU estimate.

    The RSS high-water mark includes earlier work in the same process. It must
    not be presented as a check-exclusive allocation or a sum across workers.
    """
    peak = None
    method = "unavailable"
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                        "PagefileUsage", "PeakPagefileUsage")]
            handle = ctypes.windll.kernel32.GetCurrentProcess
            handle.restype = wintypes.HANDLE
            query = ctypes.windll.psapi.GetProcessMemoryInfo
            query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            query.restype = wintypes.BOOL
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if query(handle(), ctypes.byref(counters), counters.cb):
                peak = int(counters.PeakWorkingSetSize)
                method = "windows.PeakWorkingSetSize"
        except (AttributeError, OSError, ValueError):
            pass
    else:
        try:
            import resource
            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak = int(value if platform.system() == "Darwin" else value * 1024)
            method = "getrusage.ru_maxrss"
        except (ImportError, AttributeError, OSError, ValueError):
            pass
    return {"cpuSeconds": round(time.process_time(), 6), "peakRssBytes": peak,
            "peakRssScope": "process_lifetime_high_water", "memoryMethod": method}
