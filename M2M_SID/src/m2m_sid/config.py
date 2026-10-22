from pathlib import Path
import yaml


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def is_placeholder(value):
    return isinstance(value, str) and value.isupper() and value.endswith(("PATH", "ROOT", "DIR"))


def require_path(value, name):
    if value is None or is_placeholder(value):
        raise ValueError(f"Set {name} in the config file")
    return str(value)


def set_seed(seed):
    import random
    import numpy as np
    import torch
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
