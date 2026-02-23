import yaml
from pathlib import Path
from types import SimpleNamespace

def _to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    return d

def load_config(filename):
    path = Path(filename)

    with open(path, 'r') as f:
        return _to_namespace(yaml.safe_load(f))

CONFIG = load_config("configuration/config.yaml")