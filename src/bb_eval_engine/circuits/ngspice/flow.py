from typing import Any, Dict, Type, cast, Sequence

import os
import abc
from copy import copy
from pathlib import Path

from .netlist import NgSpiceWrapper
from ..base import FlowManager
from ...util.importlib import import_cls
from ...data.design import Design


class NgspiceFlowManager(FlowManager, abc.ABC):

    def __init__(self, ngspice_config: Dict[str, Any], **kwargs):
        FlowManager.__init__(self, **kwargs)
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

    def get_netlist_params(self, design: Design,
                           path_base_names: Sequence[str],
                           name: str) -> Dict[str, str]:
        """Returns a dictionary of parameters compatible with jinja2 template formatting.
        Parameters
        ----------
        design: Design
            The design object.
            Note: Design object should already have the interpretation of circuit related parameters
            in its spec dictionary (i.e. rload=100, cload=20f)
        path_base_names: Sequence[str]
            Base names used for save directory paths in the template of that wrapper.
        name: str
            Name of the ngspice wrapper to be used.

        Returns
        -------
        updated_dict: Dict[str, str]
            Dictionary of sim_parameters to be usd via jinja2 rendering engine.
        """

        ngspice = self.ngspice_lut[name]
        for k in path_base_names:
            design[k] = str(ngspice.get_design_folder(design['id']) / f'{k}.csv')
        design['include'] = str(self.sim_model.resolve())
        updated_dict = copy(design.specs)
        updated_dict['id'] = design['id']
        updated_dict.update(design.value_dict)

        return updated_dict
