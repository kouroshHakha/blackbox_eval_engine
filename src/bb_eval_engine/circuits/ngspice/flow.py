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

        try:
            self.sim_model = Path(kwargs['sim_model'])
        except KeyError:
            try:
                self.sim_model = Path(os.environ['SIM_MODEL'])
            except KeyError:
                raise EnvironmentError('sim_model is not defined in the env spec file and variable'
                                       'SIM_MODEL is also not specified for running ngspice')

        self.ngspice_lut = {}
        for name, config in ngspice_config.items():
            cls = cast(Type[NgSpiceWrapper], import_cls(config['ngspice_cls']))
            self.ngspice_lut[name] = cls(num_process=num_process, design_netlist=config['netlist'])

    def update_netlist_model_paths(self, design: Design,
                                   path_base_names: Sequence[str],
                                   name: str) -> Dict[str, str]:


        ngspice = self.ngspice_lut[name]
        dsn_id = design['id']
        updated_dict = {}
        for k in path_base_names:
            updated_dict[k] = str(ngspice.get_design_folder(dsn_id) / f'{k}.csv')
        updated_dict['include'] = self.sim_model.resolve()
        updated_dict['id'] = dsn_id

        return updated_dict
