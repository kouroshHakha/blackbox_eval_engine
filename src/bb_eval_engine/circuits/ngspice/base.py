from typing import Optional, Dict, Any, Sequence

import random

from ...util.design import Design
from ...util.importlib import import_cls
from ..base import CircuitsEngineBase, SpecSeqType
from ..bag_new.base import BagNetEngineBase


class NgspiceEngineBase(BagNetEngineBase):

    def __init__(self, yaml_fname: Optional[str] = None,
                 specs: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        CircuitsEngineBase.__init__(self, yaml_fname, specs, **kwargs)
        _eval_cls_str = self.specs['flow_manager_cls']
        _eval_cls = import_cls(_eval_cls_str)
        self.flow_manager = _eval_cls(**self.specs['flow_manager_params'], **kwargs)
