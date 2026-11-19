"""Access bundled, versioned rule assets independently of the working folder."""

from importlib.resources import files
try:
    from importlib.resources.abc import Traversable
except ImportError:  # Python 3.10 exposes the protocol in importlib.abc.
    from importlib.abc import Traversable


def asset(group: str, name: str) -> Traversable:
    """Return a resource in a known asset group, including installed wheels."""
    if group not in {"spec", "agent"} or "/" in name or "\\" in name:
        raise ValueError("Unknown asset group or invalid resource name")
    return files("spaceqa_agent").joinpath("assets", group, name)
