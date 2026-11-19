#!/usr/bin/env python3
"""Compatibility wrapper for the independently implemented run auditor."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spaceqa_agent.audit import *  # noqa: F403 - preserve the historical API

if __name__ == "__main__":
    main()  # noqa: F405
