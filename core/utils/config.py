"""Configuration loading utilities."""
from pathlib import Path

import yaml


class DotDict(dict):
    """Dictionary with attribute-style access."""

    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
        if isinstance(val, dict) and not isinstance(val, DotDict):
            val = DotDict(val)
            self[key] = val
        return val

    def __setattr__(self, key, val):
        self[key] = val


def load_config(path: str) -> DotDict:
    """Load a YAML config file and return as DotDict.

    Args:
        path: Path to YAML file.

    Returns:
        DotDict with nested attribute access.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return DotDict(data)
