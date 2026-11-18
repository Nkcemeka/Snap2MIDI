"""Paths for the paper experiments, read from experiments/config.yaml.

Every script in experiments/ takes `--config` (default: experiments/config.yaml,
copied from config.example.yaml) and reads its paths through `load`, so no
path is hard-coded anywhere else.
"""

from pathlib import Path

import yaml

EXPERIMENTS = Path(__file__).resolve().parent
DEFAULT = EXPERIMENTS / "config.yaml"


def _expand(node):
    if isinstance(node, dict):
        return {k: _expand(v) for k, v in node.items()}
    return Path(str(node)).expanduser()


def load(path=DEFAULT) -> dict:
    """The config with every value turned into an expanded Path."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} not found: copy experiments/config.example.yaml "
                         "to experiments/config.yaml and fill in the paths")
    return _expand(yaml.safe_load(path.read_text()))
