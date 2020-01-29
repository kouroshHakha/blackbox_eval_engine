from typing import Union

from pathlib import Path

from ..base import EvaluationEngineBase
from utils.file import read_yaml
from utils.importlib import import_class

def import_bb_env(env_yaml_str: Union[str, Path]) -> EvaluationEngineBase:
    specs = read_yaml(env_yaml_str)
    env = import_class(specs['bb_engine'])(specs=specs['bb_engine_params'])
    return env

def import_cls(class_str: str):
    """Dummy function for compatibility reasons
    TODO: remove this later."""
    return import_class(class_str)