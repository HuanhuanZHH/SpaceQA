#!/usr/bin/env python3
"""Compatibility entry point for existing OPC scripts and Python imports."""

from pathlib import Path
import sys

# A source checkout works without installation; installed users use ``spaceqa``.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    from spaceqa_agent.cli import main

    main()
else:
    from spaceqa_agent import engine

    # Historical experiment loaders retain the object returned by
    # module_from_spec(), rather than re-reading sys.modules after execution.
    # Populate that object as well as preserving normal-import module aliases.
    globals().update({name: value for name, value in vars(engine).items()
                      if not name.startswith("__")})
    sys.modules[__name__] = engine
