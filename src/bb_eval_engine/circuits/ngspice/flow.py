from typing import Any, Dict, Type, cast, Sequence

import os
import abc
from pathlib import Path

from .netlist import NgSpiceWrapper
from ..base import FlowManager
from ...util.importlib import import_cls
from ...util.design import Design


class NgspiceFlowManager(FlowManager, abc.ABC):

    def __init__(self, ngspice_config: Dict[str, Any], **kwargs):
        num_process = kwargs.get('num_workers', 1)
        self.verbose = kwargs.get('verbose', False)

        self.ngspice_lut = {}
        for name, config in ngspice_config.items():
            cls = cast(Type[NgSpiceWrapper], import_cls(config['ngspice_cls']))
            self.ngspice_lut[name] = cls(num_process=num_process, design_netlist=config['netlist'])

    def update_netlist_model_paths(self, design: Design,
                                   path_base_names: Sequence[str],
                                   name: str) -> Dict[str, str]:
        try:
            model_path = Path(os.environ['SIM_MODEL'])
        except KeyError:
            raise EnvironmentError('$SIM_MODEL is not specified for running ngspice')

        ngspice = self.ngspice_lut[name]
        dsn_id = design['id']
        updated_dict = {}
        for k in path_base_names:
            updated_dict[k] = str(ngspice.get_design_folder(dsn_id) / f'{k}.csv')
        updated_dict['include'] = model_path.resolve()
        updated_dict['id'] = dsn_id

        return updated_dict
